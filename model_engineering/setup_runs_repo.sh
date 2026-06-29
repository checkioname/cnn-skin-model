#!/bin/bash
# Configura o repositório psoriasis-runs para versionamento de checkpoints/logs
#
# Uso:
#   ./setup_runs_repo.sh                               # configura com caminho padrão
#   ./setup_runs_repo.sh /caminho/absoluto
#
# Pré-requisitos:
#   1. gh auth login (ou token GITHUB_TOKEN)
#   2. git-lfs instalado

set -e

GITHUB_USER="${GITHUB_USER:-checkioname}"
RUNS_DIR="${1:-$HOME/Documents/PsoriasisEngineering/cnn-skin-model-runs}"

echo "=== Setup do Repositório de Runs ==="
echo "Local: $RUNS_DIR"
echo ""

# 1. Clonar se não existir
if [ -d "$RUNS_DIR/.git" ]; then
    echo "[OK] $RUNS_DIR já clonado"
else
    echo "[...] Clonando para $RUNS_DIR..."
    gh repo clone checkioname/cnn-skin-model-runs "$RUNS_DIR" 2>/dev/null || \
        git clone git@github.com:checkioname/cnn-skin-model-runs.git "$RUNS_DIR"
fi

cd "$RUNS_DIR"

# 2. Configurar Git LFS
echo "[...] Configurando Git LFS..."
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.pkl"

if [ -f ".gitattributes" ]; then
    git add .gitattributes
    if ! git diff --cached --quiet; then
        git commit -m "chore: track .pt/.pth/.pkl via Git LFS"
        git push
    fi
fi

echo ""
echo "=== Setup concluído! ==="
echo ""
echo "O results_dir padrão já aponta para: $RUNS_DIR"
echo "  MLflow:  mlflow ui --port 5000 --backend-store-uri $RUNS_DIR/mlflow"
echo "  TB:      tensorboard --logdir $RUNS_DIR/tensorboard"
echo ""
echo "Para ver resultados localmente:"
echo "  cd $RUNS_DIR && git pull && tensorboard --logdir $RUNS_DIR"
