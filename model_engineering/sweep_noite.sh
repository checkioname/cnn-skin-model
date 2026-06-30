#!/bin/bash
# Sweep completo — 15 hipóteses, ~6h total (EarlyStopping corta mais cedo)
# Uso: bash sweep_noite.sh
# Monitorar: tail -f sweep_noite.log

LOG=sweep_noite.log
BASE="training.epochs=50 data.preprocessed_dir=preprocessed dp=true"

echo "=== INICIO: $(date) ===" | tee $LOG

run() {
    local name="$1"
    shift
    echo "" | tee -a $LOG
    echo "==========================================" | tee -a $LOG
    echo "[$(date)] $name" | tee -a $LOG
    echo "Comando: python model_engineering/main.py $@" | tee -a $LOG
    echo "==========================================" | tee -a $LOG
    python model_engineering/main.py "$@" >> $LOG 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✓ $name concluído" | tee -a $LOG
    else
        echo "[$(date)] ✗ $name FALHOU (exit=$exit_code)" | tee -a $LOG
    fi
}

# ==========================================
# RODADA 1 — VGG16: otimizador + unfreeze + pos_weight
# ==========================================

run "01-VGG16-Adam-unfreeze1-pos0" \
    model=vgg16 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=1 training.pos_weight=0

run "02-VGG16-Adam-unfreeze1-pos1.96" \
    model=vgg16 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=1 training.pos_weight=1.96

run "03-VGG16-Adam-unfreeze2-pos0" \
    model=vgg16 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=2 training.pos_weight=0

run "04-VGG16-Adam-unfreeze2-pos1.96" \
    model=vgg16 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=2 training.pos_weight=1.96

run "05-VGG16-Adam-unfreezeALL-pos1.96" \
    model=vgg16 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=-1 training.pos_weight=1.96

# ==========================================
# RODADA 2 — VGG16: SGD + LR alto
# ==========================================

run "06-VGG16-SGD-unfreeze2-pos1.96" \
    model=vgg16 training.optimizer=sgd training.lr=0.01 \
    training.unfreeze_blocks=2 training.pos_weight=1.96

run "07-VGG16-SGD-unfreezeALL-pos1.96" \
    model=vgg16 training.optimizer=sgd training.lr=0.001 \
    training.unfreeze_blocks=-1 training.pos_weight=1.96

run "08-VGG16-Adam-LR3e4-unfreeze2-pos1.96" \
    model=vgg16 training.optimizer=adam training.lr=0.0003 \
    training.unfreeze_blocks=2 training.pos_weight=1.96

# ==========================================
# RODADA 3 — Swin Transformer
# ==========================================

run "09-Swin-Adam-unfreeze1-pos1.96" \
    model=swin training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=1 training.pos_weight=1.96

run "10-Swin-Adam-unfreeze2-pos1.96" \
    model=swin training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=2 training.pos_weight=1.96

run "11-Swin-Adam-unfreezeALL-pos1.96" \
    model=swin training.optimizer=adam training.lr=0.00005 \
    training.unfreeze_blocks=-1 training.pos_weight=1.96

# ==========================================
# RODADA 4 — ResNet152
# ==========================================

run "12-ResNet-Adam-unfreeze1-pos1.96" \
    model=resnet152 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=1 training.pos_weight=1.96

run "13-ResNet-Adam-unfreeze2-pos1.96" \
    model=resnet152 training.optimizer=adam training.lr=0.0001 \
    training.unfreeze_blocks=2 training.pos_weight=1.96

run "14-ResNet-Adam-unfreezeALL-pos1.96" \
    model=resnet152 training.optimizer=adam training.lr=0.00005 \
    training.unfreeze_blocks=-1 training.pos_weight=1.96

# ==========================================
# RODADA 5 — VGG16: exploratório (controle)
# ==========================================

run "15-VGG16-Adam-LR1e3-unfreeze2-pos1.96" \
    model=vgg16 training.optimizer=adam training.lr=0.001 \
    training.unfreeze_blocks=2 training.pos_weight=1.96

# ==========================================

echo "" | tee -a $LOG
echo "=== FIM: $(date) ===" | tee -a $LOG
echo "=== Resultados em MLflow: ===" | tee -a $LOG
echo "mlflow ui --port 5000 --backend-store-uri sqlite:///../cnn-skin-model-runs/mlflow/mlflow.db" | tee -a $LOG
