#!/bin/bash
set -e

MODE="normal"
if [[ "$1" == "--deep" ]]; then
  MODE="deep"
fi

echo "🧹 Cleaning caches (mode: $MODE)..."

# Root & home caches
rm -rf ~/.cache /root/.cache ~/.huggingface 2>/dev/null || true

# Workspace caches
rm -rf /workspace/.cache/* 2>/dev/null || true
rm -rf /workspace/hf_cache/* 2>/dev/null || true
rm -rf /workspace/tmp/* 2>/dev/null || true
rm -rf /workspace/wandb/* 2>/dev/null || true

# Torch / pip cache
rm -rf /workspace/envs/qwen/qwenvenv/lib/python*/site-packages/torch/.nv_fuser_cache 2>/dev/null || true
pip cache purge || true

# Deep mode: wipe models + runs
# if [[ "$MODE" == "deep" ]]; then
#   echo "⚠️ Deep mode enabled: removing models and runs!"
#   rm -rf /workspace/models/* 2>/dev/null || true
  # rm -rf /workspace/projects/nutrition-table/runs/* 2>/dev/null || true
# fi

# Recreate structure
mkdir -p /workspace/.cache/hf/hub
mkdir -p /workspace/.cache/hf/transformers
mkdir -p /workspace/.cache/hf/datasets
mkdir -p /workspace/.cache/tmp
mkdir -p /workspace/.cache/wandb
mkdir -p /workspace/hf_cache
mkdir -p /workspace/tmp
mkdir -p /workspace/wandb

echo "✅ Cleanup done. Current disk usage:"
df -h /
df -h /workspace

