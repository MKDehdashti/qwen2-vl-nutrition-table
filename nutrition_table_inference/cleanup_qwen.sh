#!/bin/bash

set -e

BASE="model/Qwen2-VL-7B"
FT="$BASE/fine_tuned"

echo "=============================================================="
echo " 1) CLEANING fine_tuned/"
echo "=============================================================="

# keep ONLY the safetensors + index + configs
find "$FT" -type f ! -name "model-*.safetensors" \
                  ! -name "model.safetensors.index.json" \
                  ! -name "config.json" \
                  ! -name "generation_config.json" \
                  -exec rm -f {} \;

echo "fine_tuned cleaned."

echo ""
echo "=============================================================="
echo " 2) CLEANING BASE MODEL ROOT"
echo "=============================================================="

# remove weight shards accidentally copied into base
find "$BASE" -maxdepth 1 -type f -name "model-*.safetensors" -exec rm -f {} \;
find "$BASE" -maxdepth 1 -type f -name "model.safetensors.index.json" -exec rm -f {} \;

# remove tokenizer duplicates in base (fine_tuned should not override tokenizer)
find "$BASE" -maxdepth 1 -type f -name "tokenizer.json" -exec rm -f {} \;
find "$BASE" -maxdepth 1 -type f -name "tokenizer_config.json" -exec rm -f {} \;
find "$BASE" -maxdepth 1 -type f -name "vocab.json" -exec rm -f {} \;
find "$BASE" -maxdepth 1 -type f -name "merges.txt" -exec rm -f {} \;

# remove stray configs that belong to fine_tuned
find "$BASE" -maxdepth 1 -type f -name "generation_config.json" -exec rm -f {} \;
find "$BASE" -maxdepth 1 -type f -name "config.json" -exec rm -f {} \;

echo "Base cleaned."

echo ""
echo "=============================================================="
echo " 3) VERIFYING REQUIRED DIRECTORIES"
echo "=============================================================="

REQUIRED_DIRS=(
  "image_processor"
  "processor"
  "vision_tower"
)

for D in "${REQUIRED_DIRS[@]}"; do
  if [ ! -d "$BASE/$D" ]; then
     echo "⚠️  WARNING: missing $BASE/$D"
  else
     echo "✔ found $BASE/$D"
  fi
done

echo ""
echo "=============================================================="
echo " 4) VERIFYING REQUIRED BASE FILES"
echo "=============================================================="

REQUIRED_FILES=(
  "tokenizer.model"
  "preprocessor_config.json"
  "vision_config.json"
  "chat_template.json"
)

for F in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$BASE/$F" ]; then
     echo "⚠️  WARNING: missing $BASE/$F"
  else
     echo "✔ found $BASE/$F"
  fi
done

echo ""
echo "=============================================================="
echo " 🎉 CLEANUP FINISHED"
echo "=============================================================="

tree "$BASE" -L 3
