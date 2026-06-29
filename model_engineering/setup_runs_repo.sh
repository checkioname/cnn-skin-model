#!/bin/bash
# Configura o repositório psoriasis-runs para versionamento de checkpoints/logs
#
# Uso:
#   ./setup_runs_repo.sh                     # configura com caminho padrão
#   ./setup_runs_repo.sh /caminho/alternativo
#
# Pré-requisitos:
#   1. gh auth login (ou token GITHUB_TOKEN)
#   2. git-lfs instalado

set -e

REPO_NAME="psoriasis-runs"
GITHUB_USER="${GITHUB_USER:-checkioname}"
RUNS_DIR="${1:-$HOME/psoriasis-runs}"

echo "=== Setup do Repositório de Runs ==="
echo "Repo:  $GITHUB_USER/$REPO_NAME"
echo "Local: $RUNS_DIR"
echo ""

# 1. Criar no GitHub
if gh repo view "$GITHUB_USER/$REPO_NAME" &>/dev/null; then
    echo "[OK] Repositório $GITHUB_USER/$REPO_NAME já existe no GitHub"
else
    echo "[...] Criando repositório $GITHUB_USER/$REPO_NAME no GitHub..."
    gh repo create "$GITHUB_USER/$REPO_NAME" --private --description "Checkpoints e logs do modelo CNN psoríase"
    echo "[OK] Repositório criado"
fi

# 2. Clonar localmente se não existir
if [ -d "$RUNS_DIR/.git" ]; then
    echo "[OK] $RUNS_DIR já clonado"
else
    echo "[...] Clonando para $RUNS_DIR..."
    gh repo clone "$GITHUB_USER/$REPO_NAME" "$RUNS_DIR"
fi

cd "$RUNS_DIR"

# 3. Configurar Git LFS
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

# 4. .gitkeep para manter a estrutura
mkdir -p runs
touch runs/.gitkeep
git add runs/.gitkeep
if ! git diff --cached --quiet; then
    git commit -m "chore: estrutura inicial runs/"
    git push
fi

echo ""
echo "=== Setup concluído! ==="
echo ""
echo "Para usar no treino:"
echo "  python main.py runs_repo.path=$RUNS_DIR runs_repo.auto_push=true"
echo ""
echo "Para ver resultados localmente:"
echo "  cd $RUNS_DIR && git pull && tensorboard --logdir ."
