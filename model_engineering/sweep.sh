#!/bin/bash
# Paraleliza 3 experimentos em GPUs separadas
# Uso: bash sweep.sh [epochs]
# Cada experimento usa 1 GPU (sem DataParallel), rodando em background.
#
# Monitoramento:
#   tail -f logs/vgg16.log
#   tail -f logs/swin.log
#   tail -f logs/resnet152.log
#   ps aux | grep python

EPOCHS=${1:-50}
mkdir -p logs

BASE="training.epochs=$EPOCHS training.optimizer=adam training.lr=0.0001 training.weight_decay=0.0001 training.unfreeze_blocks=2 training.pos_weight=1.96 data.preprocessed_dir=preprocessed"

echo "=== Iniciando sweep em paralelo ==="
echo "GPU0: VGG16    → logs/vgg16.log"
echo "GPU1: Swin     → logs/swin.log"
echo "GPU2: ResNet152 → logs/resnet152.log"
echo ""

# GPU 0: VGG16
CUDA_VISIBLE_DEVICES=0 python main.py model=vgg16 $BASE \
  > logs/vgg16.log 2>&1 &
echo "[GPU0] VGG16 PID: $!"

# GPU 1: Swin
CUDA_VISIBLE_DEVICES=1 python main.py model=swin $BASE \
  > logs/swin.log 2>&1 &
echo "[GPU1] Swin PID: $!"

# GPU 2: ResNet152
CUDA_VISIBLE_DEVICES=2 python main.py model=resnet152 $BASE \
  > logs/resnet152.log 2>&1 &
echo "[GPU2] ResNet152 PID: $!"

echo ""
echo "Monitorar todos: tail -f logs/vgg16.log logs/swin.log logs/resnet152.log"
echo "Aguardar: wait"
