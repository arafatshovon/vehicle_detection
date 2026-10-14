#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/images-and-voc-xmls [output-directory]" >&2
  exit 2
fi
SOURCE_DIR="$1"
OUTPUT_DIR="${2:-$ROOT_DIR/data/processed}"
cd "$ROOT_DIR"
python -m dhaka_traffic.prepare_data --source "$SOURCE_DIR" --output "$OUTPUT_DIR"
