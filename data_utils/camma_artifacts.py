from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CammaArtifactPaths:
    output_root: Path
    inference_root: Path
    overlays_dir: Path
    predictions_coco_path: Path
    legacy_predictions_coco_path: Path
    eval_root: Path
    gt_coco_path: Path
    cgf1_gt_coco_path: Path
    cgf1_predictions_coco_path: Path
    results_path: Path
    legacy_results_path: Path


def _slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.strip().lower())
    return slug.strip("_") or "camma"


def normalize_split_name(split: str) -> str:
    split_normalized = split.strip().lower()
    if split_normalized in {"val", "valid", "validation"}:
        return "val"
    if split_normalized in {"train", "test"}:
        return split_normalized
    raise ValueError(f"Unsupported split: {split!r}. Expected one of: train, val, test.")


def resolve_camma_split(
    data_config: dict,
    requested_split: str | None = None,
    default_key: str = "test_split",
) -> str:
    if requested_split is not None:
        return normalize_split_name(requested_split)

    fallback_default = "test" if default_key == "test_split" else "val"
    configured_split = str(data_config.get(default_key, fallback_default))
    return normalize_split_name(configured_split)


def resolve_output_root(config: dict, dataset_name: str | None = None) -> Path:
    configured_output = config.get("output", {}).get("output_dir")
    if configured_output:
        return Path(configured_output)

    dataset_slug = _slugify_dataset_name(dataset_name or "camma")
    return Path("outputs") / dataset_slug


def build_camma_artifact_paths(
    config: dict,
    split_name: str,
    dataset_name: str | None = None,
    overlay_root: str | Path | None = None,
) -> CammaArtifactPaths:
    normalized_split = normalize_split_name(split_name)
    output_root = resolve_output_root(config, dataset_name=dataset_name)
    inference_root = output_root / "inference" / normalized_split
    default_overlays_dir = inference_root / "overlays"
    overlays_dir = Path(overlay_root) if overlay_root is not None else default_overlays_dir
    eval_root = output_root / "evaluation" / normalized_split
    return CammaArtifactPaths(
        output_root=output_root,
        inference_root=inference_root,
        overlays_dir=overlays_dir,
        predictions_coco_path=inference_root / "predictions_coco.json",
        legacy_predictions_coco_path=output_root / f"eval_predictions_{normalized_split}.json",
        eval_root=eval_root,
        gt_coco_path=eval_root / "gt_coco.json",
        cgf1_gt_coco_path=eval_root / "cgf1_gt_coco.json",
        cgf1_predictions_coco_path=eval_root / "cgf1_predictions_coco.json",
        results_path=eval_root / "results.json",
        legacy_results_path=output_root / f"eval_results_{normalized_split}.json",
    )


def prediction_path_candidates(paths: CammaArtifactPaths) -> list[Path]:
    candidates = []
    seen = set()
    for path in (paths.predictions_coco_path, paths.legacy_predictions_coco_path):
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
