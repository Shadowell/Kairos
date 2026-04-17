#!/usr/bin/env bash
# One-shot bootstrap for a fresh AutoDL GPU instance.
# Run from /root/autodl-tmp/Kairos after scp has landed the code + data
# tarballs into /root/autodl-tmp/.
#
# Usage:
#   bash scripts/autodl_bootstrap.sh <path-to-dataset-tarball>
#
# Example:
#   bash scripts/autodl_bootstrap.sh /root/autodl-tmp/bv_1min_2y.tar.gz
#
# What it does (each step is idempotent):
#   1. Unset AutoDL academic-turbo proxy (known to hijack hf-mirror.com).
#   2. Create .venv, pip install -e '.[train]', pin numpy<2.
#   3. Configure HF_ENDPOINT=hf-mirror.com + HF_HOME on data disk.
#   4. Extract the dataset tarball into ./data/crypto/ and verify meta.json.
#   5. Run a short smoke training (KAIROS_SMOKE=1 + KAIROS_PRESET=crypto-1min).
#      If that passes, print the exact command to start the real training.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-dataset-tarball>" >&2
    echo "Example: $0 /root/autodl-tmp/bv_1min_2y.tar.gz" >&2
    exit 1
fi

DATA_TAR="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- 1. proxy hygiene ----
echo "[1/5] clearing AutoDL academic-turbo proxies (if any)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
env | grep -iE 'proxy=' || echo "  no proxy env — good"

# ---- 2. venv + deps ----
echo "[2/5] preparing venv + installing Kairos[train]"
if [ ! -d .venv ]; then
    python -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip
pip install -e '.[train]'
# Torch wheels in Kairos pin against numpy 1.x ABI; 2.x causes subtle
# _ARRAY_API crashes.
pip install 'numpy<2'

# ---- 3. HuggingFace mirror config ----
echo "[3/5] setting HF_ENDPOINT + HF_HOME"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/hf_cache"
mkdir -p "$HF_HOME"

# Persist to ~/.bashrc so subsequent sessions keep the config.
if ! grep -q "HF_ENDPOINT=https://hf-mirror.com" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'EOF'
# Kairos: route HuggingFace through the cn mirror and cache on the data disk.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
EOF
fi

# ---- 4. extract dataset ----
echo "[4/5] extracting dataset from $DATA_TAR"
mkdir -p data/crypto
tar xzf "$DATA_TAR" -C data/crypto
# figure out the extracted folder name (strip .tar.gz)
ds_name="$(basename "$DATA_TAR" .tar.gz)"
DATASET_DIR="data/crypto/${ds_name}"
if [ ! -f "$DATASET_DIR/meta.json" ]; then
    echo "error: $DATASET_DIR/meta.json missing; dataset not extracted as expected" >&2
    exit 1
fi
echo "  dataset at $DATASET_DIR"
python -c "
import json, pathlib, pickle
meta = json.loads(pathlib.Path('$DATASET_DIR/meta.json').read_text())
print('  market:', meta.get('market'), 'exog_dim:', len(meta.get('exog_cols', [])))
for split in ('train', 'val', 'test'):
    with open('$DATASET_DIR/' + split + '_data.pkl', 'rb') as f:
        d = pickle.load(f)
    n = sum(len(v) for v in d.values())
    print(f'  {split}: {len(d)} symbols, {n} rows')
"

# ---- 5. smoke training ----
echo "[5/5] running smoke training (1 epoch, 200 steps)"
export KAIROS_SMOKE=1
export KAIROS_PRESET=crypto-1min
export KAIROS_DATASET="$DATASET_DIR"

torchrun --standalone --nproc_per_node=1 \
    -m kairos.training.train_predictor

cat <<EOF

Smoke OK. To start the real training, run inside tmux:

    tmux new -s train
    cd $REPO_ROOT && source .venv/bin/activate
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export KAIROS_PRESET=crypto-1min
    export KAIROS_DATASET=$DATASET_DIR
    torchrun --standalone --nproc_per_node=1 \\
        -m kairos.training.train_predictor 2>&1 | tee train.log

Detach: Ctrl+B then D.
Re-attach from a new SSH: tmux attach -t train

When training finishes, download the checkpoint from your Mac:

    scp -P <PORT> -r root@<HOST>:$REPO_ROOT/artifacts/checkpoints/predictor \\
        /Users/jie.feng/wlb/Kairos/artifacts/checkpoints/

EOF
