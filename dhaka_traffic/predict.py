"""Run YOLO detection on an image, directory, or video."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Image, directory of images, or video file")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="dhaka_traffic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}")
    if not args.source.exists():
        raise SystemExit(f"Prediction source not found: {args.source}")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.predict(source=str(args.source), imgsz=args.imgsz, conf=args.conf,
                  device=resolve_device(args.device), save=True,
                  project=str(args.project), name=args.name)


if __name__ == "__main__":
    main()
