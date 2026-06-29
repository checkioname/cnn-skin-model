#!/usr/bin/env bash
set -euo pipefail

# Benchmark comparativo: VGG16 vs ViT vs Swin
# Uso: bash benchmark.sh [epochs]
# Exemplo: bash benchmark.sh 50

EPOCHS="${1:-50}"
BASE="python main.py training.epochs=${EPOCHS} training.batch_size=32 dp=true"

echo "=========================================="
echo " Benchmark: VGG16 (CNN classica)"
echo "=========================================="
$BASE model=vgg16 training.lr=0.01 training.optimizer=sgd training.scheduler=plateau

echo ""
echo "=========================================="
echo " Benchmark: ViT (Vision Transformer)"
echo "=========================================="
$BASE model=vit training.lr=0.0001 training.optimizer=adam training.scheduler=cosine

echo ""
echo "=========================================="
echo " Benchmark: Swin Transformer"
echo "=========================================="
$BASE model=swin training.lr=0.0001 training.optimizer=adam training.scheduler=cosine

echo ""
echo "=========================================="
echo " Benchmark concluido!"
echo " Veja os resultados no MLflow:"
echo " mlflow ui --port 5000"
echo "=========================================="
