from __future__ import annotations

import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .common import (
    normalize_student_variant,
    resolve_training_settings,
    resolve_upstream_repo_path,
    resolve_upstream_variant_config,
)


@dataclass
class RTDETRv4StudentBundle:
    model: nn.Module
    criterion: nn.Module
    postprocessor: nn.Module
    build_config_path: Path
    initial_load_report: dict[str, Any] | None = None


@dataclass(frozen=True)
class _UpstreamSymbols:
    YAMLConfig: type
    DINOv3TeacherModel: type
    remove_module_prefix: Any


_UPSTREAM_SYMBOL_CACHE: dict[Path, _UpstreamSymbols] = {}


def _load_upstream_symbols(repo_path: Path) -> _UpstreamSymbols:
    repo_path = repo_path.resolve()
    cached = _UPSTREAM_SYMBOL_CACHE.get(repo_path)
    if cached is not None:
        return cached

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    import engine  # noqa: F401
    from engine.core import YAMLConfig
    from engine.rtv4.dinov3_teacher import DINOv3TeacherModel
    from engine.solver._solver import remove_module_prefix

    symbols = _UpstreamSymbols(
        YAMLConfig=YAMLConfig,
        DINOv3TeacherModel=DINOv3TeacherModel,
        remove_module_prefix=remove_module_prefix,
    )
    _UPSTREAM_SYMBOL_CACHE[repo_path] = symbols
    return symbols


def _patch_criterion_for_missing_teacher(criterion: nn.Module) -> nn.Module:
    original_loss_distillation = criterion.loss_distillation

    def patched_loss_distillation(self, outputs, targets, indices, num_boxes, **kwargs):
        student_feature_map = outputs.get("student_distill_output")
        teacher_feature_map = outputs.get("teacher_encoder_output")
        if student_feature_map is None or teacher_feature_map is None:
            reference = outputs.get("pred_boxes")
            if reference is None:
                reference = outputs.get("pred_logits")
            device = reference.device if reference is not None else torch.device("cpu")
            return {"loss_distill": torch.tensor(0.0, device=device, requires_grad=bool(reference is not None))}
        return original_loss_distillation(outputs, targets, indices, num_boxes, **kwargs)

    criterion.loss_distillation = types.MethodType(patched_loss_distillation, criterion)
    return criterion


def _unwrap_external_checkpoint(checkpoint: Any, remove_module_prefix) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
        ema_state = checkpoint["ema"].get("module")
        if isinstance(ema_state, dict):
            checkpoint = ema_state
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported RT-DETRv4 checkpoint format.")

    return remove_module_prefix(checkpoint)


def load_compatible_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    current = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    missed: list[str] = []
    skipped: list[str] = []

    for name, tensor in current.items():
        source = state_dict.get(name)
        if source is None:
            missed.append(name)
            continue
        if tuple(source.shape) != tuple(tensor.shape):
            skipped.append(name)
            continue
        matched[name] = source

    module.load_state_dict(matched, strict=False)
    return {
        "matched_count": len(matched),
        "missed_count": len(missed),
        "skipped_count": len(skipped),
        "missed_keys": missed,
        "skipped_keys": skipped,
    }


def _build_upstream_config_payload(
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: Path,
    include_teacher: bool,
    teacher_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_config_path = resolve_upstream_variant_config(model_config)
    recipe_settings = resolve_training_settings(model_config, training_config)
    teacher_config = dict(teacher_config or {})

    payload: dict[str, Any] = {
        "__include__": [str(variant_config_path)],
        "num_classes": 1,
        "remap_mscoco_category": False,
        "output_dir": str(output_dir),
        "use_focal_loss": True,
        "eval_spatial_size": [int(recipe_settings["image_size"]), int(recipe_settings["image_size"])],
        "HGNetv2": {
            "pretrained": False,
        },
        "PostProcessor": {
            "num_top_queries": int(training_config.get("num_top_queries", 300)),
        },
        "RTv4Criterion": {
            "weight_dict": {
                "loss_distill": float(recipe_settings["distill_loss_weight"]),
            },
            "distill_adaptive_params": recipe_settings["distill_adaptive_params"],
        },
    }
    distill_teacher_dim = teacher_config.get("distill_teacher_dim")
    if distill_teacher_dim is not None:
        payload["HybridEncoder"] = {
            "distill_teacher_dim": int(distill_teacher_dim),
        }
    if include_teacher:
        payload["teacher_model"] = {
            "type": "DINOv3TeacherModel",
            "dinov3_repo_path": str(teacher_config["dinov3_repo_path"]),
            "dinov3_weights_path": str(teacher_config["dinov3_weights_path"]),
            "patch_size": int(teacher_config.get("patch_size", 16)),
            "mean": list(teacher_config.get("mean", [0.485, 0.456, 0.406])),
            "std": list(teacher_config.get("std", [0.229, 0.224, 0.225])),
        }
        dinov3_model_type = teacher_config.get("dinov3_model_type")
        if dinov3_model_type is not None:
            payload["teacher_model"]["dinov3_model_type"] = str(dinov3_model_type)
    return payload


def _write_upstream_config(payload: dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        return Path(handle.name)


def build_student_bundle(
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    teacher_config: dict[str, Any] | None = None,
) -> RTDETRv4StudentBundle:
    model_config = dict(model_config or {})
    training_config = dict(training_config or {})
    output_dir = Path(output_dir or "outputs/bsafe_cbd_rtdetrv4")
    repo_path = resolve_upstream_repo_path(model_config)
    _ = normalize_student_variant(model_config.get("student_variant"))
    symbols = _load_upstream_symbols(repo_path)

    payload = _build_upstream_config_payload(
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
        include_teacher=False,
        teacher_config=teacher_config,
    )
    config_path = _write_upstream_config(payload)
    upstream_cfg = symbols.YAMLConfig(str(config_path))
    model = upstream_cfg.model
    criterion = _patch_criterion_for_missing_teacher(upstream_cfg.criterion)
    postprocessor = upstream_cfg.postprocessor

    load_report = None
    student_init_path = model_config.get("student_init_path")
    if student_init_path:
        checkpoint = torch.load(student_init_path, map_location="cpu")
        external_state = _unwrap_external_checkpoint(checkpoint, symbols.remove_module_prefix)
        load_report = load_compatible_state_dict(model, external_state)

    return RTDETRv4StudentBundle(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        build_config_path=config_path,
        initial_load_report=load_report,
    )


def build_teacher_model(
    model_config: dict[str, Any] | None = None,
    teacher_config: dict[str, Any] | None = None,
) -> nn.Module:
    model_config = dict(model_config or {})
    teacher_config = dict(teacher_config or {})
    repo_path = resolve_upstream_repo_path(model_config)
    if "dinov3_repo_path" not in teacher_config or "dinov3_weights_path" not in teacher_config:
        raise ValueError(
            "Missing `teacher.dinov3_repo_path` or `teacher.dinov3_weights_path` for RT-DETRv4 distillation."
        )
    symbols = _load_upstream_symbols(repo_path)
    teacher_kwargs: dict[str, Any] = {
        "dinov3_repo_path": str(teacher_config["dinov3_repo_path"]),
        "dinov3_weights_path": str(teacher_config["dinov3_weights_path"]),
        "patch_size": int(teacher_config.get("patch_size", 16)),
        "mean": tuple(teacher_config.get("mean", [0.485, 0.456, 0.406])),
        "std": tuple(teacher_config.get("std", [0.229, 0.224, 0.225])),
    }
    dinov3_model_type = teacher_config.get("dinov3_model_type")
    if dinov3_model_type is not None:
        teacher_kwargs["dinov3_model_type"] = str(dinov3_model_type)
    return symbols.DINOv3TeacherModel(**teacher_kwargs)
