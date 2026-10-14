"""Train the historical YOLOv8m baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/data.yaml"))
    parser.add_argument("--model", default="yolov8m.pt", help="Pretrained checkpoint or Ultralytics model name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA device index")
    parser.add_argument("--seed", type=int, default=0, help="Historical training seed")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="dhaka_traffic_baseline")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset config not found: {args.data}. Run scripts/prepare_data.sh first.")
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=resolve_device(args.device), seed=args.seed, deterministic=True,
        save=True, save_period=20, pretrained=True, plots=True, val=True,
        lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=0.05, cls=0.5, dfl=1.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, close_mosaic=10,
        resume=args.resume, project=str(args.project), name=args.name,
    )


if __name__ == "__main__":
    main()
