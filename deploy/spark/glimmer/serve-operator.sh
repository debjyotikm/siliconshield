#!/usr/bin/env bash
# Serve base Muse Glimmer + trained operator adapter on DGX Spark (loopback only).
# Usage: serve-operator.sh <workspace> <adapter-dir> <model-name> [port]
set -euo pipefail
WORKSPACE="${1:?workspace root, e.g. /path/to/glimmer-workspace}"
ADAPTER="${2:?adapter path relative to workspace, e.g. runs/operator/best/adapter}"
NAME="${3:?unique model identity for this adapter}"
PORT="${4:-18320}"
IMAGE="${GLIMMER_IMAGE:-siliconshield-glimmer}"
SOURCE="${GLIMMER_SOURCE:-source}"

docker rm -f siliconshield-glimmer-serve >/dev/null 2>&1 || true
docker run -d --name siliconshield-glimmer-serve \
  --gpus all --network host \
  -v "$WORKSPACE":/workspace \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH="/workspace/$SOURCE" \
  -e SILICONSHIELD_GPU_LOCK="${GLIMMER_GPU_LOCK:-/workspace/runtime/glimmer-gpu.lock}" \
  --entrypoint python3 "$IMAGE" \
  -m siliconshield.research_loop.controller.serve \
  --model /workspace/models/muse-glimmer-30b \
  --adapter "/workspace/$ADAPTER" \
  --name "$NAME" --host 127.0.0.1 --port "$PORT"
echo "serving $NAME on 127.0.0.1:$PORT (container: siliconshield-glimmer-serve)"
