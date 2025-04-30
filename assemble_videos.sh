#!/usr/bin/env bash
set -euo pipefail

# Default values
FRAMERATE=30
OUTPUT_DIR="out/videos"

usage() {
  cat <<EOF
Usage: $0 [-o output_dir] [-r framerate] [frames_dir1 frames_dir2 ...]

Assemble PNG frames into MP4 videos using ffmpeg.

Options:
  -o DIR    Directory to write .mp4 files (default: out/videos)
  -r RATE   Input framerate (frames per second, default: 30)
  -h        Show this help message

If no frames_dir arguments are given, the script will process all subdirectories
matching out/*/frames.
EOF
  exit 1
}

# Parse options
while getopts ":o:r:h" opt; do
  case $opt in
    o) OUTPUT_DIR="$OPTARG" ;;
    r) FRAMERATE="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done
shift $((OPTIND-1))

# Gather frames directories
if [ $# -gt 0 ]; then
  FRAMES_DIRS=("$@")
else
  # default to out/*/frames
  FRAMES_DIRS=(out/*/frames)
fi

mkdir -p "$OUTPUT_DIR"

for fd in "${FRAMES_DIRS[@]}"; do
  if [ -d "$fd" ]; then
    parent=$(basename "$(dirname "$fd")")
    out_file="${OUTPUT_DIR}/${parent}.mp4"
    echo "📽  Assembling '$fd' → '$out_file' at ${FRAMERATE}fps"

    ffmpeg -y \
      -framerate "$FRAMERATE" \
      -pattern_type glob \
      -i "${fd}/*.png" \
      -c:v libx264 \
      -pix_fmt yuv420p \
      "$out_file"

    echo "✅  Created $out_file"
  else
    echo "⚠️  Skipping missing directory: $fd" >&2
  fi
done