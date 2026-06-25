#!/usr/bin/env bash
# Download a Qwen checkpoint into models/ for local training & eval.
#
# Usage: scripts/download_model.sh [base|instruct|0.5b-instruct]
#   base          -> Qwen/Qwen3-4B-Base
#   instruct      -> Qwen/Qwen3-4B-Instruct-2507
#   0.5b-instruct -> Qwen/Qwen2.5-0.5B-Instruct   (default for fast iteration)
#
# Runs in the active Python env (see README "Install"). Requires huggingface_hub,
# which is installed by `pip install -e .`.
set -euo pipefail

# 1. Require exactly one argument (the variant to download).
if [ "$#" -ne 1 ]; then
    echo "Error: missing argument." >&2
    echo "Usage: $0 [base|instruct|0.5b-instruct]" >&2
    exit 1
fi

# 2. Normalize the argument to lowercase.
VARIANT=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# 3. Map the variant to a HF repo id and a local output directory.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${VARIANT}" = "base" ]; then
    MODEL_ID="Qwen/Qwen3-4B-Base"
    OUT_DIR="${ROOT}/models/Qwen3-4B-Base"
elif [ "${VARIANT}" = "instruct" ]; then
    MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
    OUT_DIR="${ROOT}/models/Qwen3-4B-Instruct"
elif [ "${VARIANT}" = "0.5b-instruct" ]; then
    MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
    OUT_DIR="${ROOT}/models/Qwen2.5-0.5B-Instruct"
else
    echo "Error: invalid argument '${1}'." >&2
    echo "Must be one of: base | instruct | 0.5b-instruct" >&2
    exit 1
fi

# 4. Download the snapshot into OUT_DIR.
mkdir -p "${OUT_DIR}"
echo "Downloading ${MODEL_ID} -> ${OUT_DIR}"

MODEL_ID="${MODEL_ID}" OUT_DIR="${OUT_DIR}" python -c "
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ['MODEL_ID'],
    local_dir=os.environ['OUT_DIR'],
    local_dir_use_symlinks=False,
)
print(f'Downloaded to: {path}')
"

echo "Done. Now set: export MODEL_PATH=${OUT_DIR}"
