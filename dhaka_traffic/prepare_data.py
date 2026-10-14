"""Convert Pascal VOC XML annotations into the historical five-fold YOLO set."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .classes import CLASSES, CLASS_TO_ID

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_annotation(xml_path: Path, image_path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[int]]:
    """Read one VOC file and return normalized YOLO boxes and class IDs."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not read annotation {xml_path}: {exc}") from exc

    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0
    if width <= 0 or height <= 0:
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"Missing image dimensions in {xml_path} and cannot read {image_path}") from exc

    rows: list[tuple[int, float, float, float, float]] = []
    labels: list[int] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        if raw_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {raw_name!r} in {xml_path}; expected one of the 21 project classes")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Object without bndbox in {xml_path}")
        try:
            xmin, ymin, xmax, ymax = [float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError as exc:
            raise ValueError(f"Invalid box coordinates in {xml_path}") from exc
        # VOC coordinates may extend a little past image edges; clip them before normalization.
        xmin, xmax = max(0.0, min(xmin, width)), max(0.0, min(xmax, width))
        ymin, ymax = max(0.0, min(ymin, height)), max(0.0, min(ymax, height))
        if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
            raise ValueError(f"Non-finite box coordinates in {xml_path}")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Degenerate box in {xml_path}: {(xmin, ymin, xmax, ymax)}")
        class_id = CLASS_TO_ID[raw_name]
        rows.append((class_id, (xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                     (xmax - xmin) / width, (ymax - ymin) / height))
        labels.append(class_id)
    return rows, labels


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def prepare(source: Path, output: Path, folds: int = 5, seed: int = 42) -> Path:
    import yaml
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source}")
    if folds < 2:
        raise ValueError("folds must be at least 2")

    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found directly inside {source}")
    xml_by_stem = {p.stem: p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".xml"}
    missing = [p.name for p in images if p.stem not in xml_by_stem]
    if missing:
        raise ValueError(f"Missing XML annotation for {len(missing)} image(s), for example: {missing[0]}")

    parsed: dict[str, list[tuple[int, float, float, float, float]]] = {}
    all_labels: list[int] = []
    groups: list[str] = []
    stratify_labels: list[int] = []
    class_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    for image in images:
        rows, labels = parse_annotation(xml_by_stem[image.stem], image)
        parsed[image.name] = rows
        all_labels.extend(labels)
        class_counts.update(labels)
        image_counts.update(set(labels))
        # Match the original notebook: stratify annotated objects while keeping each image grouped.
        stratify_labels.extend(labels)
        groups.extend([image.name] * len(labels))
    if not stratify_labels:
        raise ValueError("No annotated objects found in the XML files")

    least_count = min(Counter(stratify_labels).values())
    if least_count < folds:
        print(f"Warning: the rarest class has {least_count} object(s), fewer than {folds} folds; stratification may be uneven.", file=sys.stderr)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    try:
        splits = list(splitter.split([[0]] * len(stratify_labels), stratify_labels, groups))
    except ValueError as exc:
        # Recent scikit-learn versions reject rare classes with support below n_splits;
        # older versions warned and continued. Preserve grouped, deterministic folds.
        print(f"Warning: stratified grouped splitting is unavailable ({exc}); falling back to GroupKFold.", file=sys.stderr)
        try:
            splits = list(GroupKFold(n_splits=folds).split([[0]] * len(stratify_labels), stratify_labels, groups))
        except ValueError as fallback_exc:
            raise ValueError(f"Could not create {folds} grouped folds: {fallback_exc}") from fallback_exc
    image_fold: dict[str, int] = {}
    for fold_id, (_, held_out_rows) in enumerate(splits):
        for row_index in held_out_rows:
            image_fold[groups[row_index]] = fold_id
    if set(image_fold) != {p.name for p in images}:
        absent = sorted({p.name for p in images} - set(image_fold))
        raise ValueError(f"Some images have no annotated objects and could not be split: {absent[:3]}")

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    per_fold_objects: dict[int, Counter[int]] = defaultdict(Counter)
    per_fold_images: Counter[int] = Counter()
    for image in images:
        fold_id = image_fold[image.name]
        fold_dir = output / f"fold_{fold_id}"
        image_dir, label_dir = fold_dir / "images", fold_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / image.name).symlink_to(image)
        label_path = label_dir / f"{image.stem}.txt"
        with label_path.open("w", encoding="utf-8") as stream:
            for class_id, x_center, y_center, box_width, box_height in parsed[image.name]:
                stream.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                per_fold_objects[fold_id][class_id] += 1
        per_fold_images[fold_id] += 1

    train_folds = list(range(folds - 1))
    data_config = {
        "path": str(output),
        "train": [f"fold_{i}/images" for i in train_folds],
        "val": f"fold_{folds - 1}/images",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    config_path = output / "data.yaml"
    config_path.write_text(yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    _write_csv(output / "class_distribution.csv", ["class_id", "class_name", "objects", "images"],
               [[i, name, class_counts[i], image_counts[i]] for i, name in enumerate(CLASSES)])
    _write_csv(output / "fold_distribution.csv", ["fold", "images", "objects"],
               [[i, per_fold_images[i], sum(per_fold_objects[i].values())] for i in range(folds)])
    print(f"Prepared {len(images)} images and {len(all_labels)} boxes across {folds} grouped folds.")
    print(f"Dataset config: {config_path}")
    return config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing image files and matching Pascal VOC XML files")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Output directory (must be empty or absent)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        prepare(args.source, args.output, args.folds, args.seed)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
