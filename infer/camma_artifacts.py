from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CammaArtifactPaths:
    output_dir: Path
    split_name: str
    dataset_name: str
    inference_root: Path
    eval_root: Path
    overlays_dir: Path
    gt_coco_path: Path
    cgf1_gt_coco_path: Path
    predictions_coco_path: Path
    legacy_predictions_coco_path: Path
    cgf1_predictions_coco_path: Path
    results_path: Path
    legacy_results_path: Path


def read_json(path: str | Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def resolve_camma_split(data_config: dict, requested_split: str | None, *, default_key: str) -> str:
    if requested_split:
        return str(requested_split)
    return str(data_config.get(default_key, data_config.get("split", "test")))


def _safe_dataset_name(dataset_name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in dataset_name)


def build_camma_artifact_paths(
    *,
    config: dict,
    split_name: str,
    dataset_name: str,
    overlay_root: str | Path | None = None,
) -> CammaArtifactPaths:
    output_dir = Path(config.get("output", {}).get("output_dir", "outputs/camma_lora"))
    split_name = str(split_name)
    dataset_name = str(dataset_name)
    dataset_slug = _safe_dataset_name(dataset_name)

    inference_root = output_dir / "inference" / split_name
    eval_root = output_dir / "evaluation" / split_name
    overlays_dir = Path(overlay_root) if overlay_root is not None else inference_root / "overlays"

    return CammaArtifactPaths(
        output_dir=output_dir,
        split_name=split_name,
        dataset_name=dataset_name,
        inference_root=inference_root,
        eval_root=eval_root,
        overlays_dir=overlays_dir,
        gt_coco_path=eval_root / f"{dataset_slug}_gt_coco.json",
        cgf1_gt_coco_path=eval_root / f"{dataset_slug}_cgf1_gt_coco.json",
        predictions_coco_path=inference_root / "predictions_coco.json",
        legacy_predictions_coco_path=output_dir / f"eval_predictions_{split_name}.json",
        cgf1_predictions_coco_path=eval_root / "cgf1_predictions_coco.json",
        results_path=eval_root / "results.json",
        legacy_results_path=output_dir / f"eval_results_{split_name}.json",
    )
