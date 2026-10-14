"""Evaluate a YOLO checkpoint and save class-level validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import load_data_config, normalize_split_paths, resolve_device


def count_instances(data_path: Path, split: str) -> dict[int, int]:
    config, root = load_data_config(data_path)
    paths = normalize_split_paths(config[split], root)
    counts: dict[int, int] = {}
    for image_dir in paths:
        label_dir = image_dir.parent / "labels" if image_dir.name == "images" else image_dir / "labels"
        if not label_dir.is_dir():
            continue
        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    class_id = int(float(parts[0]))
                    counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/data.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/validation"))
    parser.add_argument("--name", default="dhaka_traffic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}")
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                        device=resolve_device(args.device), project=str(args.project), name=args.name,
                        plots=True, save_json=False)
    output_dir = Path(results.save_dir)
    counts = count_instances(args.data, args.split)
    class_ids = list(getattr(results.box, "ap_class_index", []))
    ap = np.asarray(getattr(results.box, "ap", []))
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name", "validation_instances", "AP50", "AP50_95"])
        for row_index, class_id in enumerate(class_ids):
            name = model.names.get(class_id, str(class_id)) if isinstance(model.names, dict) else model.names[class_id]
            ap50_95 = float(ap[row_index].mean()) if ap.ndim == 2 and row_index < len(ap) else ""
            ap50 = float(ap[row_index][0]) if ap.ndim == 2 and row_index < len(ap) else ""
            writer.writerow([class_id, name, counts.get(int(class_id), 0), ap50, ap50_95])
    summary = {key: float(value) for key, value in results.results_dict.items() if isinstance(value, (int, float, np.number))}
    summary["split"] = args.split
    summary["instances_by_class"] = {str(k): v for k, v in sorted(counts.items())}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
