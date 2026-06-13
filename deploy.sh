#!/usr/bin/env bash
set -e

usage() {
    echo "Uso: ./deploy.sh <usuario>@<host> [diretorio_remoto]"
    echo ""
    echo "Exemplo: ./deploy.sh king@100.100.100.100 ~/cnn-skin-model"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-~/cnn-skin-model}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Sincronizando projeto para $SSH_TARGET:$REMOTE_DIR ==="

rsync -avz --progress \
    --exclude '*.jpg' \
    --exclude '*.png' \
    --exclude '**/__pycache__' \
    --exclude '.git' \
    --exclude 'runs' \
    --exclude 'data_engineering/main' \
    "$LOCAL_DIR/" "$SSH_TARGET:$REMOTE_DIR"

echo ""
echo "=== Instalando dependencias no servidor ==="
ssh "$SSH_TARGET" "cd $REMOTE_DIR/model_engineering && pip install -r requirements.txt"

echo ""
echo "=== Deploy concluido! ==="
echo ""
echo "Comandos uteis:"
echo "  Treinar:        ssh $SSH_TARGET 'cd $REMOTE_DIR/model_engineering && python main.py -e 50 -m vgg16'"
echo "  Shell remoto:   ssh $SSH_TARGET"
echo "  Sync rapido:    rsync -avz --exclude='*.jpg' --exclude='*.png' $LOCAL_DIR/ $SSH_TARGET:$REMOTE_DIR"
