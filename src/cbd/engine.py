from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.image_utils import normalized_cxcywh_to_original_xywh
from sam3.model.box_ops import box_cxcywh_to_xyxy, fast_diag_box_iou, fast_diag_generalized_box_iou

from .common import build_center_targets, label_to_target_type, resolve_input_size
from .dataset import CBDDataset, cbd_collate_fn
from .model import CBDBoxModel, CBDModelOutput


def resolve_device(config: dict) -> torch.device:
    preferred = str(config.get("hardware", {}).get("device", "cuda"))
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_autocast(device: torch.device, training_config: dict):
    if device.type != "cuda":
        return contextlib.nullcontext()
    precision = str(training_config.get("mixed_precision", "bf16")).lower()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def resolve_loss_weights(training_config: dict) -> dict[str, float]:
    default = {
        "box_l1": 5.0,
        "box_giou": 2.0,
        "center_ce": 1.0,
        "heatmap_bce": 1.0,
        "type_ce": 1.0,
    }
    for key, value in training_config.get("loss_weights", {}).items():
        default[str(key)] = float(value)
    return default


def compute_box_losses(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    weights = loss_weights or {"box_l1": 5.0, "box_giou": 2.0}
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    loss_bbox = F.l1_loss(pred_boxes, target_boxes)
    loss_giou = (1.0 - fast_diag_generalized_box_iou(pred_xyxy, target_xyxy)).mean()
    return {
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
        "loss": float(weights.get("box_l1", 5.0)) * loss_bbox + float(weights.get("box_giou", 2.0)) * loss_giou,
    }


def compute_box_metric_tensors(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    iou = fast_diag_box_iou(pred_xyxy, target_xyxy)
    center_error = (pred_boxes[:, :2] - target_boxes[:, :2]).abs().mean(dim=-1)
    box_error = (pred_boxes - target_boxes).abs().mean(dim=-1)
    return {
        "mean_iou": iou,
        "center_error": center_error,
        "box_error": box_error,
    }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_cbd_losses(
    model_output: CBDModelOutput,
    batch: dict,
    training_config: dict,
) -> dict[str, torch.Tensor]:
    loss_weights = resolve_loss_weights(training_config)
    loss_dict = compute_box_losses(model_output.pred_boxes, batch["target_box"], loss_weights=loss_weights)

    zero = loss_dict["loss_bbox"].new_zeros(())
    type_loss = zero
    type_accuracy = zero
    center_ce = zero
    heatmap_bce = zero

    if model_output.type_logits is not None:
        valid_type_mask = batch["target_type_label"] >= 0
        if valid_type_mask.any():
            type_loss = F.cross_entropy(
                model_output.type_logits[valid_type_mask],
                batch["target_type_label"][valid_type_mask],
            )
            type_accuracy = (
                model_output.type_logits[valid_type_mask].argmax(dim=-1)
                == batch["target_type_label"][valid_type_mask]
            ).float().mean()

    if model_output.center_cell_logits is not None and model_output.grid_size is not None:
        grid_h, grid_w = model_output.grid_size
        center_indices, center_heatmaps = build_center_targets(
            batch["target_box"],
            grid_h,
            grid_w,
            sigma=float(training_config.get("center_heatmap_sigma", 1.0)),
        )
        center_ce = F.cross_entropy(model_output.center_cell_logits, center_indices)
        heatmap_bce = F.binary_cross_entropy_with_logits(model_output.center_heatmap_logits, center_heatmaps)

    total_loss = loss_dict["loss"]
    total_loss = total_loss + loss_weights["center_ce"] * center_ce
    total_loss = total_loss + loss_weights["heatmap_bce"] * heatmap_bce
    total_loss = total_loss + loss_weights["type_ce"] * type_loss

    return {
        **loss_dict,
        "loss_center_ce": center_ce,
        "loss_heatmap_bce": heatmap_bce,
        "loss_type_ce": type_loss,
        "type_accuracy": type_accuracy,
        "loss_total": total_loss,
    }


class CBDTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.output_config = config.get("output", {})
        self.device = resolve_device(config)

        self.model = CBDBoxModel(self.model_config).to(self.device)
        weight_decay = float(self.training_config.get("weight_decay", 1e-4))
        backbone_params = self.model.backbone_trainable_parameters()
        backbone_ids = {id(parameter) for parameter in backbone_params}
        new_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        if backbone_params:
            optimizer_params = []
            if new_params:
                optimizer_params.append(
                    {
                        "params": new_params,
                        "lr": float(self.training_config.get("new_layers_lr", self.training_config.get("learning_rate", 1e-4))),
                        "weight_decay": weight_decay,
                    }
                )
            optimizer_params.append(
                {
                    "params": backbone_params,
                    "lr": float(self.training_config.get("backbone_lr", 1e-5)),
                    "weight_decay": weight_decay,
                }
            )
        else:
            optimizer_params = [
                {
                    "params": [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    "lr": float(self.training_config.get("learning_rate", 1e-4)),
                    "weight_decay": weight_decay,
                }
            ]

        self.optimizer = AdamW(optimizer_params)
        scheduler_name = str(self.training_config.get("lr_scheduler", "cosine")).lower()
        self.scheduler = (
            CosineAnnealingLR(self.optimizer, T_max=max(1, int(self.training_config.get("num_epochs", 20))))
            if scheduler_name == "cosine"
            else None
        )
        self.input_size = resolve_input_size(self.model_config)
        self.output_dir = Path(self.output_config.get("output_dir", "outputs/bsafe_cbd"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_stats_path = self.output_dir / "val_stats.jsonl"
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def build_dataset(self, split: str, apply_augmentation: bool = False) -> CBDDataset:
        augmentation_level = 0
        if apply_augmentation and split == self.data_config.get("train_split", "train"):
            augmentation_level = int(self.training_config.get("augmentation_level", 0))
        return CBDDataset.from_config(
            self.config,
            split,
            augmentation_level=augmentation_level,
        )

    def build_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = self.build_dataset(
            split,
            apply_augmentation=shuffle and split == self.data_config.get("train_split", "train"),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.training_config.get("batch_size", 2)),
            shuffle=shuffle,
            num_workers=int(self.training_config.get("num_workers", 0)),
            pin_memory=bool(self.config.get("hardware", {}).get("dataloader_pin_memory", True)),
            collate_fn=cbd_collate_fn,
        )

    def save_checkpoint(self, filename: str) -> Path:
        path = self.output_dir / filename
        torch.save({"model": self.model.state_dict(), "config": self.config}, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

    def run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {
            "loss": 0.0,
            "loss_bbox": 0.0,
            "loss_giou": 0.0,
            "loss_center_ce": 0.0,
            "loss_heatmap_bce": 0.0,
            "loss_type_ce": 0.0,
            "mean_iou": 0.0,
            "center_error": 0.0,
            "box_error": 0.0,
            "type_accuracy": 0.0,
        }
        per_type_sums = {
            name: {"mean_iou": 0.0, "center_error": 0.0, "box_error": 0.0, "count": 0}
            for name in ("soft", "hard")
        }
        count = 0
        accumulation_steps = max(1, int(self.training_config.get("gradient_accumulation_steps", 1)))
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc="Train" if train else "Eval")
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, self.device)
            with make_autocast(self.device, self.training_config):
                model_output = self.model(batch["rgb"], batch["masks"])
                loss_dict = compute_cbd_losses(model_output, batch, self.training_config)
                metric_tensors = compute_box_metric_tensors(model_output.pred_boxes.detach(), batch["target_box"])
                loss = loss_dict["loss_total"]

            if train:
                (loss / accumulation_steps).backward()
                if step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                        float(self.training_config.get("max_grad_norm", 1.0)),
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            count += 1
            totals["loss"] += float(loss.detach().item())
            for key in ("loss_bbox", "loss_giou", "loss_center_ce", "loss_heatmap_bce", "loss_type_ce", "type_accuracy"):
                totals[key] += float(loss_dict[key].detach().item())
            for key in ("mean_iou", "center_error", "box_error"):
                totals[key] += float(metric_tensors[key].mean().detach().item())

            for type_label, type_name in enumerate(("soft", "hard")):
                mask = batch["target_type_label"] == type_label
                if not mask.any():
                    continue
                per_type_sums[type_name]["count"] += int(mask.sum().item())
                for metric_name in ("mean_iou", "center_error", "box_error"):
                    per_type_sums[type_name][metric_name] += float(metric_tensors[metric_name][mask].sum().detach().item())

            pbar.set_postfix(loss=f"{totals['loss'] / count:.4f}", iou=f"{totals['mean_iou'] / count:.4f}")

        if train and count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                float(self.training_config.get("max_grad_norm", 1.0)),
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        stats = {key: value / max(1, count) for key, value in totals.items()}
        for type_name, values in per_type_sums.items():
            per_type_count = max(1, values["count"])
            stats[f"{type_name}_count"] = float(values["count"])
            stats[f"{type_name}_mean_iou"] = values["mean_iou"] / per_type_count
            stats[f"{type_name}_center_error"] = values["center_error"] / per_type_count
            stats[f"{type_name}_box_error"] = values["box_error"] / per_type_count
        return stats

    def log_val_stats(self, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]) -> None:
        payload = {"epoch": epoch}
        payload.update({f"train_{key}": value for key, value in train_stats.items()})
        payload.update({f"val_{key}": value for key, value in val_stats.items()})
        with open(self.val_stats_path, "a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def train(self) -> dict[str, float]:
        train_loader = self.build_dataloader(self.data_config.get("train_split", "train"), shuffle=True)
        val_loader = self.build_dataloader(self.data_config.get("val_split", "val"), shuffle=False)
        num_epochs = int(self.training_config.get("num_epochs", 20))
        patience = int(self.training_config.get("early_stopping_patience", num_epochs))
        epochs_without_improvement = 0

        best_stats = {}
        for epoch in range(1, num_epochs + 1):
            train_stats = self.run_epoch(train_loader, train=True)
            with torch.no_grad():
                val_stats = self.run_epoch(val_loader, train=False)
            self.log_val_stats(epoch, train_stats, val_stats)
            self.save_checkpoint("last_cbd.pt")
            if val_stats["loss"] < self.best_val_loss:
                self.best_val_loss = val_stats["loss"]
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint("best_cbd.pt")
                best_stats = {"epoch": epoch, **val_stats}
            else:
                epochs_without_improvement += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if epochs_without_improvement >= patience:
                break
        return best_stats

    def evaluate(self, split: str, weights_path: str | Path) -> dict[str, float]:
        self.load_checkpoint(weights_path)
        loader = self.build_dataloader(split, shuffle=False)
        with torch.no_grad():
            return self.run_epoch(loader, train=False)

    def predict_batch(self, batch: dict) -> CBDModelOutput:
        self.model.eval()
        batch = move_batch_to_device(batch, self.device)
        with torch.no_grad():
            with make_autocast(self.device, self.training_config):
                return self.model(batch["rgb"], batch["masks"])

    def predict_dataset_index(self, dataset: CBDDataset, sample_index: int) -> dict:
        record = dataset.get_record(sample_index)
        sample = dataset[sample_index]
        batch = cbd_collate_fn([sample])
        model_output = self.predict_batch(batch)
        pred_boxes = model_output.pred_boxes.detach().cpu()[0]
        orig_h, orig_w = sample["original_size"]

        prediction = {
            "clip_id": record.clip_id,
            "split": dataset.split,
            "pred_box_norm_cxcywh": [float(value) for value in pred_boxes.tolist()],
            "pred_bbox_xywh": list(normalized_cxcywh_to_original_xywh(pred_boxes.tolist(), orig_w, orig_h)),
            "sample": sample,
            "record": record,
            "dataset": dataset,
        }

        target_box = sample.get("target_box")
        if target_box is not None:
            target_box = target_box.detach().cpu()
            prediction["target_box_norm_cxcywh"] = [float(value) for value in target_box.tolist()]
            prediction["target_bbox_xywh"] = list(normalized_cxcywh_to_original_xywh(target_box.tolist(), orig_w, orig_h))
            prediction["iou"] = float(
                compute_box_metric_tensors(pred_boxes.unsqueeze(0), target_box.unsqueeze(0))["mean_iou"][0].item()
            )

        target_type_name = sample.get("target_type_name")
        if target_type_name is not None:
            prediction["target_type"] = target_type_name

        target_type_label = sample.get("target_type_label")
        if target_type_label is not None:
            prediction["target_type_label"] = int(target_type_label.item())

        if model_output.type_logits is not None:
            type_probs = model_output.type_logits.detach().cpu().softmax(dim=-1)[0]
            pred_type_label = int(type_probs.argmax().item())
            prediction["pred_type_label"] = pred_type_label
            prediction["pred_type_name"] = label_to_target_type(pred_type_label)
            prediction["pred_type_probs"] = {
                label_to_target_type(index): float(prob.item()) for index, prob in enumerate(type_probs)
            }
            prediction["pred_type_confidence"] = float(type_probs[pred_type_label].item())
        if model_output.center_cell_logits is not None:
            center_cell_probs = model_output.center_cell_logits.detach().cpu()[0].softmax(dim=-1)
            pred_center_cell_index = int(center_cell_probs.argmax().item())
            prediction["pred_center_cell_confidence"] = float(center_cell_probs[pred_center_cell_index].item())
            prediction["pred_center_cell_index"] = pred_center_cell_index
            if model_output.grid_size is not None:
                _, grid_w = model_output.grid_size
                prediction["pred_center_cell_row_col"] = [
                    pred_center_cell_index // grid_w,
                    pred_center_cell_index % grid_w,
                ]
        if model_output.center_heatmap_logits is not None:
            prediction["pred_center_heatmap"] = model_output.center_heatmap_logits.detach().cpu().sigmoid()[0]
        if model_output.attention_map is not None:
            prediction["pred_attention_map"] = model_output.attention_map.detach().cpu()[0]
        return prediction

    def predict_record(self, split: str, clip_id: str) -> dict:
        dataset = self.build_dataset(split, apply_augmentation=False)
        sample_index = next(index for index, record in enumerate(dataset.records) if record.clip_id == clip_id)
        return self.predict_dataset_index(dataset, sample_index)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)
