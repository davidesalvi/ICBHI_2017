#!/bin/bash

FEATURE_SETS=("MelSpec" "LogSpec" "MFCC" "LFCC")
MODEL_ARCHS=("ResNet" "LCNN")
BINARY_CLASS=("True" "False")
WIN_LENS=(5.0 10.0)

for FEATURE_SET in "${FEATURE_SETS[@]}"; do
  for MODEL_ARCH in "${MODEL_ARCHS[@]}"; do
    for BINARY in "${BINARY_CLASS[@]}"; do
      for WIN_LEN in "${WIN_LENS[@]}"; do
        echo "Running experiment with --feature_set=$FEATURE_SET --model_arch=$MODEL_ARCH --binary_classification=$BINARY --win_len=$WIN_LEN"
        python main.py \
          --feature_set "$FEATURE_SET" \
          --model_arch "$MODEL_ARCH" \
          --binary_classification "$BINARY" \
          --win_len "$WIN_LEN"
      done
    done
  done
done
