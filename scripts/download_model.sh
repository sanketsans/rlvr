#!/usr/bin/env bash
# Download either Qwen3-4B-Base or Qwen3-4B-Instruct dynamically
set -euo pipefail

# 1. Validate that exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Error: Missing argument." >&2
    echo "Usage: $0 [base|instruct]" >&2
    exit 1
fi

# 2. Assign and normalize the argument to lowercase
VARIANT=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# 3. Determine MODEL_ID and OUT_DIR based on the input string
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${VARIANT}" = "base" ]; then
    MODEL_ID="Qwen/Qwen3-4B-Base"
    OUT_DIR="${ROOT}/models/Qwen3-4B-Base"
elif [ "${VARIANT}" = "instruct" ]; then
    MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
    OUT_DIR="${ROOT}/models/Qwen3-4B-Instruct"
else
    echo "Error: Invalid argument '${1}'." >&2
    echo "Must be either 'base' or 'instruct'." >&2
    exit 1
fi

# 4. Execute the download execution blocks
mkdir -p "${OUT_DIR}"

echo "Downloading ${MODEL_ID} -> ${OUT_DIR}"

conda run -n olmo --no-capture-output env \
  MODEL_ID="${MODEL_ID}" OUT_DIR="${OUT_DIR}" \
  python -c "
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ['MODEL_ID'],
    local_dir=os.environ['OUT_DIR'],
    local_dir_use_symlinks=False,
)
print(f'Downloaded to: {path}')
"

echo "Done. Set: export MODEL_PATH=${OUT_DIR}"