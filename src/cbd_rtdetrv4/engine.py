from __future__ import annotations

import contextlib
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler

from .common import DEFAULT_OUTPUT_DIR, resolve_output_dir, resolve_training_settings
from .dataset import CBDRTDetrV4Dataset
from .metrics import (
    bbox_iou_xywh,
    build_metrics_payload,
    flatten_metrics_payload,
    write_metrics_json,
    write_predictions_jsonl,
    xyxy_to_xywh,
)
from .model import build_student_bundle, build_teacher_model
from .preprocess import RTDETRv4Collator, prepare_inference_image


def _merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_dict(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r") as handle:
        payload = yaml.safe_load(handle) or {}

    merged: dict[str, Any] = {}
    includes = list(payload.get("__include__", []) or [])
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = (config_path.parent / include_path).resolve()
        _merge_dict(merged, load_config(include_path))

    payload = dict(payload)
    payload.pop("__include__", None)
    return _merge_dict(merged, payload)


def slurm_distributed_enabled() -> bool:
    return int(os.environ.get("SLURM_NTASKS", "1")) > 1


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def gather_object_list(local_value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [local_value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_value)
    return gathered


def reduce_mean(value: float, count: float) -> float:
    if not dist.is_initialized():
        return value / max(1.0, count)
    tensor = torch.tensor([value, count], dtype=torch.float64, device=torch.device("cuda"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item() / max(1.0, tensor[1].item()))


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return moved


def sizes_hw_to_wh(sizes: torch.Tensor) -> torch.Tensor:
    return sizes[:, [1, 0]]


def make_autocast(device: torch.device, training_config: dict[str, Any]):
    if device.type != "cuda":
        return contextlib.nullcontext()
    precision = str(training_config.get("mixed_precision", "bf16")).lower()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def temporarily_set_training(module: torch.nn.Module | None, enabled: bool):
    if module is None:
        yield
        return
    previous = module.training
    module.train(enabled)
    try:
        yield
    finally:
        module.train(previous)


def _compute_encoder_grad_percentage(model: torch.nn.Module) -> float:
    total_l1 = 0.0
    encoder_l1 = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().abs().sum().item()
        total_l1 += value
        if name.startswith("encoder.encoder") or name.startswith("module.encoder.encoder"):
            encoder_l1 += value
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * encoder_l1 / total_l1


def _distillation_is_adaptive(criterion: torch.nn.Module) -> bool:
    params = getattr(criterion, "distill_adaptive_params", None) or {}
    return bool(params.get("enabled", False))


class CBDRTDetrV4Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.teacher_config = config.get("teacher", {})
        self.training_config = resolve_training_settings(self.model_config, config.get("training", {}))
        self.output_config = config.get("output", {})
        self.hardware_config = config.get("hardware", {})

        self.multi_gpu = slurm_distributed_enabled()
        if self.multi_gpu:
            self.rank, self.world_size, local_rank = setup_distributed()
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            preferred_device = str(self.hardware_config.get("device", "cuda"))
            if preferred_device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.output_dir = Path(self.output_config.get("output_dir", DEFAULT_OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        bundle = build_student_bundle(
            self.model_config,
            self.training_config,
            self.output_dir,
            self.teacher_config,
        )
        self.model = bundle.model.to(self.device)
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False,
            )
        self.unwrapped_model = self.model.module if self.multi_gpu else self.model
        self.criterion = bundle.criterion.to(self.device)
        self.postprocessor = bundle.postprocessor.to(self.device) if hasattr(bundle.postprocessor, "to") else bundle.postprocessor
        self.student_build_config_path = bundle.build_config_path
        self.student_initial_load_report = bundle.initial_load_report

        self.teacher_model = None
        self.collator = RTDETRv4Collator()

        backbone_parameters = []
        detector_parameters = []
        for name, parameter in self.unwrapped_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_parameters.append(parameter)
            else:
                detector_parameters.append(parameter)

        param_groups = []
        if detector_parameters:
            param_groups.append(
                {
                    "params": detector_parameters,
                    "lr": float(self.training_config["detector_lr"]),
                }
            )
        if backbone_parameters:
            param_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": float(self.training_config["backbone_lr"]),
                }
            )

        self.optimizer = AdamW(
            param_groups,
            weight_decay=float(self.training_config["weight_decay"]),
        )

        precision = str(self.training_config.get("mixed_precision", "bf16")).lower()
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and precision == "fp16",
        )
        self.scheduler = None
        self.val_stats_path = self.output_dir / "val_stats.jsonl"
        self.best_metric = float("-inf")
        self.best_epoch = 0

    def ensure_teacher_model(self) -> torch.nn.Module:
        if self.teacher_model is None:
            self.teacher_model = build_teacher_model(self.model_config, self.teacher_config).to(self.device)
            self.teacher_model.eval()
        return self.teacher_model

    def build_dataset(self, split: str, train: bool) -> CBDRTDetrV4Dataset:
        return CBDRTDetrV4Dataset.from_config(self.config, split=split, train=train)

    def build_dataloader(self, split: str, train: bool) -> DataLoader:
        dataset = self.build_dataset(split=split, train=train)
        sampler = None
        shuffle = train
        if self.multi_gpu:
            sampler = DistributedSampler(dataset, shuffle=train)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=int(self.training_config["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(self.training_config["num_workers"]),
            pin_memory=bool(self.hardware_config.get("dataloader_pin_memory", True)),
            collate_fn=self.collator,
        )

    def save_checkpoint(self, filename: str, metric: float | None = None) -> Path:
        path = self.output_dir / filename
        torch.save(
            {
                "model": self.unwrapped_model.state_dict(),
                "config": self.config,
                "best_metric": self.best_metric,
                "metric": metric,
            },
            path,
        )
        return path

    def load_checkpoint(self, weights_path: str | Path) -> None:
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.unwrapped_model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and "best_metric" in checkpoint:
            self.best_metric = float(checkpoint["best_metric"])

    def build_scheduler(self, num_training_steps: int):
        scheduler_name = str(self.training_config.get("lr_scheduler", "cosine")).lower()
        if scheduler_name == "none":
            return None
        warmup_steps = int(self.training_config.get("warmup_steps", 0))
        return get_scheduler(
            scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, num_training_steps),
        )

    def _update_distill_weight(self, epoch: int, grad_percentages: list[float]) -> None:
        params = getattr(self.criterion, "distill_adaptive_params", None) or {}
        if not params.get("enabled", False):
            return
        current_weight = float(self.criterion.weight_dict.get("loss_distill", 0.0))
        avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0
        new_weight = current_weight

        if avg_percentage < 1e-6:
            default_weight = params.get("default_weight")
            if default_weight is not None:
                new_weight = float(default_weight)
        elif epoch >= int(self.training_config.get("stage_stop_epoch", self.training_config["num_epochs"])):
            default_weight = params.get("default_weight")
            if default_weight is not None:
                new_weight = float(default_weight)
        else:
            rho = float(params["rho"])
            delta = float(params["delta"])
            lower_bound = rho - delta
            upper_bound = rho + delta
            if not (lower_bound <= avg_percentage <= upper_bound):
                target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                if current_weight > 1e-6:
                    p_current = avg_percentage / 100.0
                    p_target = target_percentage / 100.0
                    numerator = p_target * (1.0 - p_current)
                    denominator = p_current * (1.0 - p_target)
                    if abs(denominator) >= 1e-9:
                        ratio = max(numerator / denominator, 0.1)
                        new_weight = min(max(current_weight * ratio, current_weight / 10.0), current_weight * 10.0)

        self.criterion.weight_dict["loss_distill"] = float(new_weight)

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        self.criterion.train()
        teacher_model = self.ensure_teacher_model()
        teacher_model.eval()

        if self.multi_gpu and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        accumulation_steps = max(1, int(self.training_config.get("gradient_accumulation_steps", 1)))
        max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))

        loss_sum = 0.0
        count = 0.0
        grad_percentages: list[float] = []
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"].to(self.device)
            targets = move_targets_to_device(batch["targets"], self.device)

            with torch.no_grad():
                teacher_features = teacher_model(pixel_values)

            with make_autocast(self.device, self.training_config):
                outputs = self.model(
                    pixel_values,
                    targets=targets,
                    teacher_encoder_output=teacher_features,
                )
                loss_dict = self.criterion(
                    outputs,
                    targets,
                    epoch=epoch,
                    step=step - 1,
                    global_step=(epoch - 1) * len(loader) + (step - 1),
                    epoch_step=len(loader),
                )
                loss = sum(loss_dict.values())

            scaled_loss = loss / accumulation_steps
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if _distillation_is_adaptive(self.criterion):
                grad_percentages.append(_compute_encoder_grad_percentage(self.model))

            if step % accumulation_steps == 0:
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.unwrapped_model.parameters() if parameter.requires_grad],
                    max_grad_norm,
                )
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            loss_sum += float(loss.detach().item())
            count += 1.0

        if count % accumulation_steps != 0:
            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.unwrapped_model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

        self._update_distill_weight(epoch, grad_percentages)
        train_loss = reduce_mean(loss_sum, count)
        return {"loss": train_loss, "distill_weight": float(self.criterion.weight_dict.get("loss_distill", 0.0))}

    def build_prediction_records(
        self,
        batch: dict[str, Any],
        detections: list[dict[str, torch.Tensor]],
        *,
        split: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, detection in enumerate(detections):
            all_detections = []
            boxes = detection["boxes"].detach().cpu()
            scores = detection["scores"].detach().cpu()
            labels = detection["labels"].detach().cpu()
            for score, label, box_xyxy in zip(scores, labels, boxes):
                bbox_xywh = xyxy_to_xywh(box_xyxy.tolist())
                all_detections.append(
                    {
                        "label": int(label.item()),
                        "score": float(score.item()),
                        "bbox_xywh": [float(value) for value in bbox_xywh],
                    }
                )

            top_detection = all_detections[0] if all_detections else None
            target_box = [float(value) for value in batch["target_boxes_xywh"][index].tolist()]
            pred_box = top_detection["bbox_xywh"] if top_detection is not None else None
            iou = bbox_iou_xywh(pred_box, target_box)

            records.append(
                {
                    "split": split,
                    "image_id": int(batch["image_ids"][index]),
                    "annotation_id": int(batch["annotation_ids"][index]),
                    "file_name": batch["file_names"][index],
                    "image_path": batch["image_paths"][index],
                    "original_size": [int(value) for value in batch["original_sizes"][index].tolist()],
                    "target_type_name": batch["target_type_names"][index],
                    "target_bbox_xywh": target_box,
                    "pred_bbox_xywh": pred_box,
                    "pred_score": None if top_detection is None else float(top_detection["score"]),
                    "iou": float(iou),
                    "detections": all_detections,
                }
            )
        return records

    def evaluate_loader(
        self,
        loader: DataLoader,
        *,
        split: str,
        write_artifacts: bool = False,
        artifact_prefix: str | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
        self.model.eval()
        self.criterion.eval()
        loss_sum = 0.0
        count = 0.0
        local_records: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(self.device)
                targets = move_targets_to_device(batch["targets"], self.device)
                # RT-DETRv4 only emits auxiliary decoder outputs in train mode,
                # but the criterion expects them even for validation loss.
                decoder = getattr(self.unwrapped_model, "decoder", None)
                with temporarily_set_training(decoder, True):
                    with make_autocast(self.device, self.training_config):
                        loss_outputs = self.model(pixel_values, targets=targets)
                        loss_dict = self.criterion(
                            loss_outputs,
                            targets,
                            epoch=0,
                            step=0,
                            global_step=0,
                            epoch_step=max(1, len(loader)),
                        )
                        loss = sum(loss_dict.values())

                with make_autocast(self.device, self.training_config):
                    outputs = self.model(pixel_values, targets=targets)
                detections = self.postprocessor(outputs, sizes_hw_to_wh(batch["original_sizes"].to(self.device)))
                local_records.extend(self.build_prediction_records(batch, detections, split=split))
                loss_sum += float(loss.detach().item())
                count += 1.0

        average_loss = reduce_mean(loss_sum, count)
        gathered = gather_object_list(local_records)
        records: list[dict[str, Any]] = []
        for shard in gathered:
            records.extend(shard)
        records.sort(key=lambda record: record["annotation_id"])

        payload = build_metrics_payload(records, split=split, loss=average_loss)
        flat_stats = flatten_metrics_payload(payload)

        if write_artifacts and is_main_process():
            prefix = artifact_prefix or f"eval_{split}"
            metrics_path = self.output_dir / f"{prefix}_metrics.json"
            predictions_path = self.output_dir / f"{prefix}_predictions.jsonl"
            write_metrics_json(metrics_path, payload)
            write_predictions_jsonl(predictions_path, records)

        return flat_stats, records, payload

    def log_val_stats(self, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]) -> None:
        if not is_main_process():
            return
        payload = {"epoch": epoch}
        payload.update({f"train_{key}": value for key, value in train_stats.items()})
        payload.update({f"val_{key}": value for key, value in val_stats.items()})
        with open(self.val_stats_path, "a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def train(self) -> dict[str, float]:
        train_split = self.data_config.get("train_split", "train")
        val_split = self.data_config.get("val_split", "val")
        train_loader = self.build_dataloader(train_split, train=True)
        val_loader = self.build_dataloader(val_split, train=False)

        accumulation_steps = max(1, int(self.training_config.get("gradient_accumulation_steps", 1)))
        steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        num_epochs = int(self.training_config["num_epochs"])
        self.scheduler = self.build_scheduler(steps_per_epoch * num_epochs)

        patience = int(self.training_config.get("early_stopping_patience", num_epochs))
        epochs_without_improvement = 0
        best_stats: dict[str, float] = {}

        for epoch in range(1, num_epochs + 1):
            train_stats = self.train_epoch(train_loader, epoch)
            val_stats, _, _ = self.evaluate_loader(val_loader, split=val_split, write_artifacts=False)
            self.log_val_stats(epoch, train_stats, val_stats)

            if is_main_process():
                self.save_checkpoint("last_cbd_rtdetrv4.pt", metric=val_stats["map50_95"])

            current_metric = float(val_stats[str(self.training_config.get("best_metric", "map50_95"))])
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                epochs_without_improvement = 0
                best_stats = {"epoch": float(epoch), **val_stats}
                if is_main_process():
                    self.save_checkpoint("best_cbd_rtdetrv4.pt", metric=current_metric)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return best_stats

    def evaluate(self, split: str, weights_path: str | Path) -> dict[str, float]:
        self.load_checkpoint(weights_path)
        loader = self.build_dataloader(split, train=False)
        prefix = f"eval_{Path(weights_path).stem}_{split}"
        stats, _, _ = self.evaluate_loader(loader, split=split, write_artifacts=True, artifact_prefix=prefix)
        return stats

    def predict_dataset_index(
        self,
        dataset: CBDRTDetrV4Dataset,
        sample_index: int,
        *,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        if score_threshold is None:
            score_threshold = float(self.training_config.get("score_threshold", 0.0))
        sample = dataset[sample_index]
        batch = self.collator([sample])
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch["pixel_values"].to(self.device))
            detections = self.postprocessor(outputs, sizes_hw_to_wh(batch["original_sizes"].to(self.device)))
        detection = detections[0]
        all_detections = []
        for score, label, box_xyxy in zip(
            detection["scores"].detach().cpu(),
            detection["labels"].detach().cpu(),
            detection["boxes"].detach().cpu(),
        ):
            score_value = float(score.item())
            if score_value < float(score_threshold):
                continue
            all_detections.append(
                {
                    "label": int(label.item()),
                    "score": score_value,
                    "bbox_xywh": [float(value) for value in xyxy_to_xywh(box_xyxy.tolist())],
                }
            )

        top_detection = all_detections[0] if all_detections else None
        target_box = [float(value) for value in sample["target_bbox_xywh"]]
        pred_box = top_detection["bbox_xywh"] if top_detection is not None else None
        return {
            "mode": "bsafe_sample",
            "split": dataset.split,
            "image_id": sample["image_id"],
            "annotation_id": sample["annotation_id"],
            "file_name": sample["file_name"],
            "image_path": sample["image_path"],
            "original_size": list(sample["original_size"]),
            "target_bbox_xywh": target_box,
            "target_type_name": sample["target_type_name"],
            "pred_bbox_xywh": pred_box,
            "pred_score": None if top_detection is None else float(top_detection["score"]),
            "iou": bbox_iou_xywh(pred_box, target_box),
            "detections": all_detections,
        }

    def predict_image_path(
        self,
        image_path: str | Path,
        *,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        prepared = prepare_inference_image(
            image_path,
            model_config=self.model_config,
            training_config=self.training_config,
        )
        if score_threshold is None:
            score_threshold = prepared["score_threshold"]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(prepared["pixel_values"].to(self.device))
            detections = self.postprocessor(
                outputs,
                sizes_hw_to_wh(torch.tensor([prepared["original_size"]], dtype=torch.long, device=self.device)),
            )

        detection = detections[0]
        all_detections = []
        for score, label, box_xyxy in zip(
            detection["scores"].detach().cpu(),
            detection["labels"].detach().cpu(),
            detection["boxes"].detach().cpu(),
        ):
            score_value = float(score.item())
            if score_value < float(score_threshold):
                continue
            all_detections.append(
                {
                    "label": int(label.item()),
                    "score": score_value,
                    "bbox_xywh": [float(value) for value in xyxy_to_xywh(box_xyxy.tolist())],
                }
            )

        top_detection = all_detections[0] if all_detections else None
        return {
            "mode": "image_path",
            "file_name": prepared["file_name"],
            "image_path": prepared["image_path"],
            "original_size": list(prepared["original_size"]),
            "pred_bbox_xywh": None if top_detection is None else top_detection["bbox_xywh"],
            "pred_score": None if top_detection is None else float(top_detection["score"]),
            "detections": all_detections,
        }
