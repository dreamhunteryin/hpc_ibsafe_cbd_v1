from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


CBD_CLASS_NAME = "CBD"
CBD_CATEGORY_ID = 0
DEFAULT_OUTPUT_DIR = "outputs/bsafe_cbd_rtdetrv4"
DEFAULT_TARGET_TYPE_ORDER = ("soft", "hard")
DEFAULT_SCORE_THRESHOLD = 0.0

SUPPORTED_STUDENT_VARIANTS = ("s", "m", "l", "x")
SUPPORTED_RECIPE_PRESETS = ("minimal", "upstream_lite", "near_upstream")

VARIANT_TO_UPSTREAM_CONFIG = {
    "s": "configs/rtv4/rtv4_hgnetv2_s_coco.yml",
    "m": "configs/rtv4/rtv4_hgnetv2_m_coco.yml",
    "l": "configs/rtv4/rtv4_hgnetv2_l_coco.yml",
    "x": "configs/rtv4/rtv4_hgnetv2_x_coco.yml",
}

VARIANT_DEFAULTS = {
    "s": {
        "backbone_lr": 2.0e-4,
        "detector_lr": 4.0e-4,
        "weight_decay": 1.0e-4,
        "num_epochs": 132,
        "warmup_steps": 2000,
        "distill_loss_weight": 5.0,
        "distill_adaptive_params": {
            "enabled": True,
            "rho": 11.0,
            "delta": 1.0,
            "default_weight": 20.0,
        },
        "stage_stop_epoch": 120,
    },
    "m": {
        "backbone_lr": 4.0e-5,
        "detector_lr": 4.0e-4,
        "weight_decay": 1.0e-4,
        "num_epochs": 102,
        "warmup_steps": 2000,
        "distill_loss_weight": 5.0,
        "distill_adaptive_params": {
            "enabled": True,
            "rho": 3.5,
            "delta": 0.25,
            "default_weight": 15.0,
        },
        "stage_stop_epoch": 90,
    },
    "l": {
        "backbone_lr": 2.5e-5,
        "detector_lr": 5.0e-4,
        "weight_decay": 1.25e-4,
        "num_epochs": 58,
        "warmup_steps": 2000,
        "distill_loss_weight": 15.0,
        "distill_adaptive_params": {
            "enabled": True,
            "rho": 2.0,
            "delta": 0.1,
            "default_weight": 15.0,
        },
        "stage_stop_epoch": 50,
    },
    "x": {
        "backbone_lr": 5.0e-6,
        "detector_lr": 5.0e-4,
        "weight_decay": 1.25e-4,
        "num_epochs": 58,
        "warmup_steps": 2000,
        "distill_loss_weight": 20.0,
        "distill_adaptive_params": {
            "enabled": True,
            "rho": 2.0,
            "delta": 0.25,
            "default_weight": 20.0,
        },
        "stage_stop_epoch": 50,
    },
}

RECIPE_PRESET_DEFAULTS = {
    "minimal": {
        "image_size": 640,
        "horizontal_flip": True,
        "photometric_distort": False,
        "color_jitter": True,
        "zoom_out": False,
        "iou_crop": False,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
    },
    "upstream_lite": {
        "image_size": 640,
        "horizontal_flip": True,
        "photometric_distort": True,
        "color_jitter": True,
        "zoom_out": True,
        "iou_crop": True,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
    },
    "near_upstream": {
        "image_size": 640,
        "horizontal_flip": True,
        "photometric_distort": True,
        "color_jitter": True,
        "zoom_out": True,
        "iou_crop": True,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
    },
}


def normalize_target_types(target_type: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if target_type is None:
        return DEFAULT_TARGET_TYPE_ORDER
    if isinstance(target_type, str):
        normalized = target_type.strip().lower()
        if normalized in {"", "*", "all"}:
            return DEFAULT_TARGET_TYPE_ORDER
        if "," in normalized:
            return tuple(part.strip() for part in normalized.split(",") if part.strip())
        return (normalized,)
    return tuple(str(part).strip().lower() for part in target_type if str(part).strip())


def normalize_student_variant(value: str | None) -> str:
    normalized = str(value or "s").strip().lower()
    if normalized not in SUPPORTED_STUDENT_VARIANTS:
        raise ValueError(
            f"Unsupported RT-DETRv4 student variant: {value!r}. "
            f"Expected one of: {', '.join(SUPPORTED_STUDENT_VARIANTS)}."
        )
    return normalized


def normalize_recipe_preset(value: str | None) -> str:
    normalized = str(value or "minimal").strip().lower()
    if normalized not in SUPPORTED_RECIPE_PRESETS:
        raise ValueError(
            f"Unsupported RT-DETRv4 recipe preset: {value!r}. "
            f"Expected one of: {', '.join(SUPPORTED_RECIPE_PRESETS)}."
        )
    return normalized


def resolve_output_dir(config: dict) -> Path:
    return Path(config.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR))


def resolve_upstream_repo_path(model_config: dict[str, Any] | None = None) -> Path:
    model_config = dict(model_config or {})
    raw_path = model_config.get("rtdetrv4_repo_path")
    if raw_path is None:
        raise ValueError(
            "Missing `model.rtdetrv4_repo_path`. "
            "Point it at a local checkout of the official RT-DETRv4 repository."
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RT-DETRv4 repo path does not exist: {path}")
    if not (path / "engine").exists():
        raise FileNotFoundError(
            f"RT-DETRv4 repo path is missing the upstream `engine/` package: {path}"
        )
    return path


def resolve_upstream_variant_config(model_config: dict[str, Any] | None = None) -> Path:
    model_config = dict(model_config or {})
    repo_path = resolve_upstream_repo_path(model_config)
    variant = normalize_student_variant(model_config.get("student_variant"))
    config_path = repo_path / VARIANT_TO_UPSTREAM_CONFIG[variant]
    if not config_path.exists():
        raise FileNotFoundError(f"Missing upstream RT-DETRv4 config: {config_path}")
    return config_path


def _merge_missing(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if key not in target or target[key] is None:
            target[key] = deepcopy(value)
    return target


def resolve_training_settings(
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = dict(model_config or {})
    resolved = deepcopy(dict(training_config or {}))
    variant = normalize_student_variant(model_config.get("student_variant"))
    preset = normalize_recipe_preset(resolved.get("recipe_preset"))
    resolved["recipe_preset"] = preset
    resolved["student_variant"] = variant
    _merge_missing(resolved, VARIANT_DEFAULTS[variant])
    _merge_missing(resolved, RECIPE_PRESET_DEFAULTS[preset])
    resolved.setdefault("early_stopping_patience", 6)
    resolved.setdefault("best_metric", "map50_95")
    resolved.setdefault("lr_scheduler", "cosine")
    resolved.setdefault("mixed_precision", "bf16")
    resolved.setdefault("max_grad_norm", 1.0)
    resolved.setdefault("gradient_accumulation_steps", 1)
    resolved.setdefault("augment", True)
    resolved.setdefault("batch_size", 2)
    resolved.setdefault("num_workers", 4)
    return resolved
