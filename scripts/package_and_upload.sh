#!/usr/bin/env bash
# Package the Kairos code + a prepared dataset bundle and scp them to
# an AutoDL instance. Meant to be run from the repo root on macOS.
#
# Usage:
#   scripts/package_and_upload.sh <SSH_PORT> <HOST> [DATASET_DIR]
#
# Examples:
#   scripts/package_and_upload.sh 12345 connect.westa.seetacloud.com
#   scripts/package_and_upload.sh 12345 123.45.67.89 data/crypto/bv_1min_2y
#
# Defaults:
#   DATASET_DIR = data/crypto/bv_1min_2y  (the 2-year BTC+ETH 1min bundle)
#
# Output uploaded to remote:
#   /root/autodl-tmp/kairos_code.tar.gz
#   /root/autodl-tmp/<dataset>.tar.gz
#
# The remote side should then run scripts/autodl_bootstrap.sh.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <SSH_PORT> <HOST> [DATASET_DIR]" >&2
    exit 1
fi

PORT="$1"
HOST="$2"
DATASET_DIR="${3:-data/crypto/bv_1min_2y}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "error: dataset dir not found: $DATASET_DIR" >&2
    exit 1
fi

DATASET_NAME="$(basename "$DATASET_DIR")"
DATASET_PARENT="$(dirname "$DATASET_DIR")"

CODE_TAR="/tmp/kairos_code.tar.gz"
DATA_TAR="/tmp/${DATASET_NAME}.tar.gz"

# ---- 1. pack code (via git archive to guarantee zero macOS metadata) ----
echo "[1/3] packing code via git archive -> $CODE_TAR"
git archive --format=tar.gz -o "$CODE_TAR" HEAD

# ---- 2. pack dataset (pkl + meta.json) ----
echo "[2/3] packing dataset $DATASET_DIR -> $DATA_TAR"
COPYFILE_DISABLE=1 tar --no-xattrs \
    --exclude='._*' --exclude='.DS_Store' \
    -czf "$DATA_TAR" \
    -C "$DATASET_PARENT" "$DATASET_NAME"

ls -lh "$CODE_TAR" "$DATA_TAR"

# ---- 3. upload ----
echo "[3/3] scp -> root@$HOST:/root/autodl-tmp/"
scp -P "$PORT" "$CODE_TAR" "$DATA_TAR" \
    "root@$HOST:/root/autodl-tmp/"

cat <<EOF

Uploaded. Next step on the AutoDL box:

    ssh -p $PORT root@$HOST
    cd /root/autodl-tmp
    tar xzf kairos_code.tar.gz -C Kairos --strip-components=0 \
        || { mkdir -p Kairos && tar xzf kairos_code.tar.gz -C Kairos; }
    cd Kairos
    bash scripts/autodl_bootstrap.sh ../${DATASET_NAME}.tar.gz

EOF
