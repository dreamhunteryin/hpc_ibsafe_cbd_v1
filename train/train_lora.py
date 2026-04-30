#!/usr/bin/env python3

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_scheduler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3 import build_sam3_image_model
from sam3.data import CammaSam3Dataset
from sam3.lora import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    save_lora_weights,
)
from sam3.model.model_misc import SAM3Output
from sam3.train import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data import collate_fn_api
from sam3.train.loss import Boxes, CORE_LOSS_KEY, IABCEMdetr, Masks, Sam3LossWrapper


def build_lora_config(lora_config: dict) -> LoRAConfig:
    return LoRAConfig(
        rank=int(lora_config.get("rank", 16)),
        alpha=int(lora_config.get("alpha", 32)),
        dropout=float(lora_config.get("dropout", 0.1)),
        target_modules=lora_config.get("target_modules"),
        target_module_match=str(lora_config.get("target_module_match", "exact")),
        apply_to_vision_encoder=bool(lora_config.get("apply_to_vision_encoder", False)),
        apply_to_text_encoder=bool(lora_config.get("apply_to_text_encoder", True)),
        apply_to_geometry_encoder=bool(lora_config.get("apply_to_geometry_encoder", False)),
        apply_to_detr_encoder=bool(lora_config.get("apply_to_detr_encoder", True)),
        apply_to_detr_decoder=bool(lora_config.get("apply_to_detr_decoder", True)),
        apply_to_decoder_text_attention=lora_config.get("apply_to_decoder_text_attention"),
        apply_to_mask_decoder=bool(lora_config.get("apply_to_mask_decoder", False)),
    )


def validate_training_configuration(lora_config: dict, training_config: dict) -> None:
    if bool(lora_config.get("apply_to_mask_decoder", False)) and not bool(
        training_config.get("use_mask_loss", True)
    ):
        raise ValueError(
            "Invalid config: lora.apply_to_mask_decoder=true requires "
            "training.use_mask_loss=true so the mask head receives supervision."
        )


def distributed_env_detected() -> bool:
    return (
        all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        or all(name in os.environ for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))
    )


def setup_distributed():
    if all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif all(name in os.environ for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")):
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        raise RuntimeError(
            "Distributed environment variables not found. Launch with srun/torchrun, "
            "or run without DDP in single-GPU mode."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
        return obj
    return obj


class Trainer:
    def __init__(self, config: dict, multi_gpu: bool = False, class_names_override: list[str] | None = None):
        self.config = config
        self.model_config = config.get("model", {})
        self.data_config = config.get("data", {})
        self.lora_config = config.get("lora", {})
        self.training_config = config.get("training", {})
        self.evaluation_config = config.get("evaluation", {})
        self.output_config = config.get("output", {})
        self.hardware_config = config.get("hardware", {})
        self.class_names_override = class_names_override
        self.multi_gpu = multi_gpu

        if self.multi_gpu:
            rank, world_size, local_rank = setup_distributed()
            self.rank = rank
            self.world_size = world_size
            self.local_rank = local_rank
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.rank = 0
            self.world_size = 1
            preferred_device = self.hardware_config.get("device", "cuda")
            if preferred_device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.last_eval_step = -1
        self._set_seed(int(self.training_config.get("seed", 42)))
        self._setup_amp()
        validate_training_configuration(self.lora_config, self.training_config)

        print_rank0("Building SAM3 model...")
        self.model = build_sam3_image_model(
            checkpoint_path=self.model_config.get("checkpoint_path"),
            device=self.device.type,
            eval_mode=False,
            compile=bool(self.hardware_config.get("use_compile", False)),
            bpe_path=self.model_config.get("bpe_path"),
            load_from_HF=bool(self.model_config.get("load_from_hf", True)),
        )

        print_rank0("Applying LoRA...")
        self.lora_cfg = build_lora_config(self.lora_config)
        self.model = apply_lora_to_model(self.model, self.lora_cfg)

        stats = count_parameters(self.model)
        print_rank0(
            f"Trainable params: {stats['trainable_parameters']:,} "
            f"({stats['trainable_percentage']:.2f}%)"
        )

        self.model.to(self.device)
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        self._unwrapped_model = self.model.module if self.multi_gpu else self.model
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.training_config.get("learning_rate", 5e-5)),
            betas=(
                float(self.training_config.get("adam_beta1", 0.9)),
                float(self.training_config.get("adam_beta2", 0.999)),
            ),
            eps=float(self.training_config.get("adam_epsilon", 1e-8)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )

        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            focal=True,
        )
        o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
        normalization = "global" if self.multi_gpu else "local"
        use_mask_loss = bool(self.training_config.get("use_mask_loss", True))
        loss_fns_find = [
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
        ]
        if use_mask_loss:
            loss_fns_find.append(
                Masks(
                    weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                    focal_alpha=0.25,
                    focal_gamma=2.0,
                    compute_aux=False,
                )
            )

        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns_find,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization=normalization,
            normalize_by_valid_object_num=False,
        )

        self.out_dir = Path(self.output_config.get("output_dir", "outputs/endoscapes_lora"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_stats_path = self.out_dir / "val_stats.jsonl"
        self.scheduler = None

    def _set_seed(self, seed: int):
        rank_seed = seed + self.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

    def _setup_amp(self):
        precision = str(self.training_config.get("mixed_precision", "none")).lower()
        self.amp_enabled = self.device.type == "cuda" and precision in {"fp16", "bf16"}
        self.amp_dtype = None
        self.scaler = None
        if not self.amp_enabled:
            return
        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler()

    def _autocast_context(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _prepare_find_targets(self, input_batch):
        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]
        for targets in find_targets:
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(self.device)
        return find_targets

    def _should_skip_batch(self, batch_dict) -> bool:
        local_has_batch = batch_dict is not None
        if not self.multi_gpu:
            return not local_has_batch

        status = torch.tensor(int(local_has_batch), device=self.device, dtype=torch.int64)
        status_min = status.clone()
        status_max = status.clone()
        dist.all_reduce(status_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(status_max, op=dist.ReduceOp.MAX)
        if status_min.item() != status_max.item():
            raise RuntimeError(
                "Distributed batch mismatch: some ranks received an empty-query batch while others did not. "
                "Use include_negatives=true or single-GPU training for this setup."
            )
        return not local_has_batch

    def _attach_matcher_indices(self, outputs_list, find_targets):
        with SAM3Output.iteration_mode(
            outputs_list,
            iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
        ) as outputs_iter:
            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                for outputs in stage_outputs:
                    outputs["indices"] = self.matcher(outputs, stage_targets)
                    if "aux_outputs" in outputs:
                        for aux_out in outputs["aux_outputs"]:
                            aux_out["indices"] = self.matcher(aux_out, stage_targets)

    def _compute_total_loss(self, input_batch):
        outputs_list = self.model(input_batch)
        find_targets = self._prepare_find_targets(input_batch)
        self._attach_matcher_indices(outputs_list, find_targets)
        loss_dict = self.loss_wrapper(outputs_list, find_targets)
        return loss_dict[CORE_LOSS_KEY]

    def _sync_gradients(self):
        if not self.multi_gpu or self.world_size <= 1:
            return
        for param in self.model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self.world_size)

    def _optimizer_step(self, max_grad_norm: float):
        if self.scaler is not None:
            if max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    def _save_lora_file(self, filename: str):
        if not is_main_process():
            return
        model_to_save = self.model.module if self.multi_gpu else self.model
        save_lora_weights(model_to_save, str(self.out_dir / filename), config=self.lora_cfg)

    def _save_checkpoint(self, global_step: int):
        if not is_main_process():
            return
        checkpoint_dir = self.out_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = self.model.module if self.multi_gpu else self.model
        save_lora_weights(model_to_save, str(checkpoint_dir / "lora_weights.pt"), config=self.lora_cfg)
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        keep_limit = int(self.training_config.get("save_total_limit", 0))
        if keep_limit <= 0 or not is_main_process():
            return

        checkpoints = []
        for path in self.out_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))

        checkpoints.sort(key=lambda item: item[0])
        while len(checkpoints) > keep_limit:
            _, old_path = checkpoints.pop(0)
            shutil.rmtree(old_path, ignore_errors=True)

    def _log_validation_stats(self, epoch: int, train_loss: float, val_loss: float):
        if not is_main_process():
            return
        with open(self.val_stats_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                + "\n"
            )

    def _run_validation(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", disable=not is_main_process())
            for batch_dict in val_pbar:
                if self._should_skip_batch(batch_dict):
                    continue
                input_batch = move_to_device(batch_dict["input"], self.device)
                with self._autocast_context():
                    total_loss = self._compute_total_loss(input_batch)
                total_val_loss += float(total_loss.detach().item())
                total_batches += 1

        if self.multi_gpu:
            reduced = torch.tensor([total_val_loss, float(total_batches)], device=self.device, dtype=torch.float64)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            total_val_loss = reduced[0].item()
            total_batches = int(reduced[1].item())

        self.model.train()
        return total_val_loss / max(1, total_batches)

    def train(self):
        data_root = self.data_config["dataset_root"]
        dataset_name = self.data_config.get("dataset_name", "Endoscapes-Seg201-CBD")
        annotation_file = self.data_config.get("annotation_file", "annotation_coco.json")
        class_names = self.class_names_override or self.data_config.get("class_names")
        include_negatives = bool(self.data_config.get("include_negatives", True))
        bbox_anchor = self.data_config.get("bbox_anchor")
        use_mask_loss = bool(self.training_config.get("use_mask_loss", True))

        print_rank0(f"Loading training data from {data_root} ({dataset_name})...")
        train_ds = CammaSam3Dataset(
            dataset_root=data_root,
            dataset_name=dataset_name,
            split=self.data_config.get("train_split", "train"),
            annotation_file=annotation_file,
            selected_class_names=class_names,
            include_negatives=include_negatives,
            augment=bool(self.training_config.get("augment", False)),
            bbox_anchor=bbox_anchor,
        )
        if use_mask_loss and not train_ds.has_segmentation_masks:
            raise ValueError(
                f"Dataset {dataset_name!r} contains no segmentation masks in {annotation_file!r}. "
                "Set training.use_mask_loss=false and lora.apply_to_mask_decoder=false for box-only training."
            )

        val_ds = CammaSam3Dataset(
            dataset_root=data_root,
            dataset_name=dataset_name,
            split=self.data_config.get("val_split", "val"),
            annotation_file=annotation_file,
            selected_class_names=class_names,
            include_negatives=include_negatives,
            augment=False,
            bbox_anchor=bbox_anchor,
        )

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=use_mask_loss)

        train_sampler = None
        val_sampler = None
        if self.multi_gpu:
            train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=get_rank(), shuffle=True)
            val_sampler = DistributedSampler(val_ds, num_replicas=self.world_size, rank=get_rank(), shuffle=False)

        batch_size = int(self.training_config.get("batch_size", 2))
        num_workers = int(self.training_config.get("num_workers", 0))
        pin_memory = bool(self.hardware_config.get("dataloader_pin_memory", True))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        accumulation_steps = max(1, int(self.training_config.get("gradient_accumulation_steps", 1)))
        max_grad_norm = float(self.training_config.get("max_grad_norm", 0.0))
        epochs = int(self.training_config.get("num_epochs", 1))
        logging_steps = max(1, int(self.training_config.get("logging_steps", 10)))
        eval_steps = int(self.training_config.get("eval_steps", 0))
        save_steps = int(self.training_config.get("save_steps", 0))

        updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
        total_training_steps = max(1, epochs * updates_per_epoch)
        self.scheduler = get_scheduler(
            name=self.training_config.get("lr_scheduler", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=int(self.training_config.get("warmup_steps", 0)),
            num_training_steps=total_training_steps,
        )

        print_rank0(f"Training for {epochs} epochs ({total_training_steps} optimizer steps)")
        optimizer_step_count = 0

        for epoch in range(epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_losses = []
            accumulation_count = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not is_main_process())

            for batch_dict in train_pbar:
                if self._should_skip_batch(batch_dict):
                    continue

                accumulation_count += 1
                should_step = accumulation_count == accumulation_steps
                sync_context = (
                    self.model.no_sync() if self.multi_gpu and not should_step else contextlib.nullcontext()
                )

                input_batch = move_to_device(batch_dict["input"], self.device)
                with sync_context:
                    with self._autocast_context():
                        total_loss = self._compute_total_loss(input_batch)
                        scaled_loss = total_loss / accumulation_steps

                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                loss_value = float(total_loss.detach().item())
                train_losses.append(loss_value)

                if should_step:
                    self._optimizer_step(max_grad_norm)
                    optimizer_step_count += 1
                    accumulation_count = 0

                    if is_main_process() and self.global_step % logging_steps == 0:
                        recent_loss = sum(train_losses[-logging_steps:]) / min(len(train_losses), logging_steps)
                        train_pbar.set_postfix(
                            {
                                "loss": f"{recent_loss:.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            }
                        )

                    if eval_steps > 0 and self.global_step % eval_steps == 0:
                        val_loss = self._run_validation(val_loader)
                        mean_train_loss = sum(train_losses) / max(1, len(train_losses))
                        print_rank0(f"Step {self.global_step}: train_loss={mean_train_loss:.4f}, val_loss={val_loss:.4f}")
                        self._log_validation_stats(epoch, mean_train_loss, val_loss)
                        self.last_eval_step = self.global_step
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_lora_file("best_lora_weights.pt")

                    if save_steps > 0 and self.global_step % save_steps == 0:
                        self._save_checkpoint(self.global_step)

            if accumulation_count > 0:
                self._sync_gradients()
                self._optimizer_step(max_grad_norm)
                optimizer_step_count += 1

            if self.last_eval_step != self.global_step:
                val_loss = self._run_validation(val_loader)
                mean_train_loss = sum(train_losses) / max(1, len(train_losses))
                print_rank0(f"Epoch {epoch + 1}: train_loss={mean_train_loss:.4f}, val_loss={val_loss:.4f}")
                self._log_validation_stats(epoch, mean_train_loss, val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_lora_file("best_lora_weights.pt")

            self._save_lora_file("last_lora_weights.pt")

        if self.multi_gpu:
            dist.barrier()

        print_rank0(f"Finished training after {optimizer_step_count} optimizer steps")
        print_rank0(f"Best validation loss: {self.best_val_loss:.4f}")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Minimal SAM3 LoRA finetuning for Endoscapes")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Force DDP. Under torchrun/srun, DDP is also auto-detected from the environment.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Optional class-name override. Can be passed as a list or a single comma-separated string.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    class_names_override = args.class_names
    if class_names_override and len(class_names_override) == 1 and "," in class_names_override[0]:
        class_names_override = [part.strip() for part in class_names_override[0].split(",") if part.strip()]

    distributed = distributed_env_detected()
    if args.multi_gpu and not distributed:
        raise RuntimeError(
            "--multi-gpu was requested, but distributed environment variables were not found. "
            "Launch with srun/torchrun, or omit --multi-gpu for single-GPU execution."
        )

    use_multi_gpu = args.multi_gpu or distributed
    trainer = Trainer(config=config, multi_gpu=use_multi_gpu, class_names_override=class_names_override)
    try:
        trainer.train()
    finally:
        if use_multi_gpu:
            cleanup_distributed()


if __name__ == "__main__":
    main()
