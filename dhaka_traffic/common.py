"""Shared command-line helpers."""

from __future__ import annotations

from pathlib import Path

def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_data_config(path: Path) -> tuple[dict, Path]:
    import yaml

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(config.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return config, root


def normalize_split_paths(value: str | list[str], root: Path) -> list[Path]:
    values = [value] if isinstance(value, str) else value
    paths = []
    for item in values:
        path = Path(item)
        paths.append(path if path.is_absolute() else root / path)
    return paths
