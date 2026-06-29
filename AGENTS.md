# AGENTS.md - Coding Guidelines for Psoriasis Engineering Project

## Project Overview

This is a dual-component machine learning project:
- **data_engineering/**: Go CLI for data preprocessing (augmentation, background removal)
- **model_engineering/**: Python/PyTorch for CNN model training

## Build/Lint/Test Commands

### Python (model_engineering)

```bash
# Install dependencies
pip install -r model_engineering/requirements.txt

# Run all tests
pytest model_engineering/

# Run a single test file
pytest model_engineering/EarlyStopping_test.py

# Run a specific test function
pytest model_engineering/EarlyStopping_test.py::test_initialization

# Run with verbose output
pytest -v model_engineering/

# Train model (example)
python model_engineering/main.py model=vgg16 training.epochs=10
```

### Docker

```bash
# Build image
docker compose build

# Build image
docker compose build

# Gerar CSV com paths relativos (imagens em /data/dermatite/, /data/psoriasis/)
docker compose run --rm -v /caminho/das/imagens:/data preprocess

# Single GPU
docker compose run --rm -v /caminho/das/imagens:/data:ro train

# DataParallel (multi-GPU)
docker compose --profile dp run --rm -v /caminho/das/imagens:/data:ro train-dp

# DDP (3 GPUs)
docker compose --profile ddp run --rm -v /caminho/das/imagens:/data:ro train-ddp
```

### Training Modes

```bash
# Single GPU (baseline)
python main.py -e 50 -m vgg16

# DataParallel (multi-GPU, single process)
python main.py -e 50 -m resnet152 --dp --batch-size 64 --num-workers 8

# DistributedDataParallel (multi-GPU, torchrun)
torchrun --nproc_per_node=3 main.py -e 50 -m vgg16 --batch-size 64 --num-workers 8
```


#### Configurações (Hydra)

```bash
# Todos os parâmetros via config.yaml ou linha de comando:
python main.py model=vgg16 training.epochs=50 training.scheduler=cosine
```

| Parâmetro | Default | Descrição |
|---|---|---|
| `model` | obrigatório | vgg16, resnet152, vit, swin |
| `training.epochs` | 50 | Número de épocas |
| `training.batch_size` | 32 | Batch size (por GPU em DDP) |
| `training.num_workers` | 4 | Workers do DataLoader |
| `training.lr` | 0.01 | Learning rate |
| `training.optimizer` | sgd | sgd, adam |
| `training.scheduler` | plateau | plateau, step, cosine |
| `training.patience` | 7 | Patience do EarlyStopping |
| `data.preprocessed_dir` | "" | Se preenchido, pula preprocessing online |
| `runs_repo.path` | "" | Caminho para repo Git LFS de runs |
| `dp` | false | DataParallel (multi-GPU single process) |

#### Schedulers disponíveis
- `plateau` — ReduceLROnPlateau (step com `test_loss`)
- `step` — StepLR (step por época)
- `cosine` — CosineAnnealingLR (T_max = epochs)

#### Métricas salvas ao final do treino (`runs/*/metrics.json`)
```json
{
  "model": "vgg16",
  "parallelism": "DDP",
  "epochs": 50,
  "total_time_seconds": 1234.56,
  "avg_throughput_img_per_sec": 85.2,
  "gpu_count": 3
}
```

### Go (data_engineering)

```bash
# Install dependencies
cd data_engineering && go mod download

# Build
cd data_engineering && go build -o main ./cmd/main.go

# Run (with subcommands)
cd data_engineering && ./main gendataset -p /path/to/data
cd data_engineering && ./main removebg -indir /input -outdir /output
cd data_engineering && ./main augment -indir /input -outdir /output
cd data_engineering && ./main stratify
```

## Code Style Guidelines

### Python Conventions

#### Naming
- **Classes**: PascalCase (e.g., `EarlyStopping`, `SetupModel`)
- **Functions/Variables**: snake_case (e.g., `train_model`, `class_to_idx`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`)
- **Private methods**: prefix with underscore (e.g., `_initialize_model`)

#### Imports
```python
# Standard library first
import sys
import os
import argparse

# Third-party libraries
import numpy as np
import torch

# Local application imports (relative)
from application.callbacks import EarlyStopping
from domain.SetupModel import SetupModel
```

#### Type Annotations
- Use type hints for function parameters and return values:
```python
def setup_model(self, device: torch.device) -> tuple:
    ...
```

#### Error Handling
- Use explicit exception handling with informative messages:
```python
try:
    model, loss_fn, optimizer, scheduler = setup.setup_model(device)
except ValueError as e:
    print(f"Model setup failed: {e}")
    raise
```

#### Class Structure
- Use `__init__` for initialization
- Use `__call__` for callable classes (like callbacks)
- Keep classes focused on single responsibility

#### Comments
- Write comments in Portuguese (project convention)
- Use docstrings for public methods:
```python
def setup_model(self, device):
    """Configura o modelo, a função de perda, o otimizador e o scheduler."""
```

### Go Conventions

#### Naming
- **Packages**: lowercase, short (e.g., `dataset`, `augmentation`)
- **Functions/Variables**: camelCase (e.g., `generateCsvFromDir`)
- **Types/Interfaces**: PascalCase (e.g., `BgRemover`)
- **Constants**: PascalCase or camelCase depending on scope

#### Imports
```go
import (
    "flag"
    "fmt"
    "os"

    "github.com/checkioname/cnn-skin-model/internal/augmentation"
    "github.com/checkioname/cnn-skin-model/internal/dataset"
)
```

#### Error Handling
- Check errors explicitly:
```go
if err != nil {
    fmt.Println("Erro:", err)
    os.Exit(1)
}
```

#### CLI Structure
- Use `flag` package for subcommands
- Follow existing pattern in `cmd/main.go`

### Testing Guidelines

#### Python (pytest)
- Test files: `*_test.py` naming convention
- Test functions: `test_*` prefix
- Use assertions:
```python
def test_initialization():
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    assert early_stopping.patience == 3
```

#### Go
- Test files: `*_test.go` naming convention
- Use `testing` package
- Table-driven tests recommended

### Project Structure

```
cnn-skin-model/
├── data_engineering/          # Go CLI
│   ├── cmd/                   # Entry points
│   ├── internal/              # Private packages
│   │   ├── augmentation/      # Image augmentation
│   │   ├── bgremover/         # Background removal
│   │   ├── dataset/           # Dataset handling
│   │   └── utils/             # Utilities
│   ├── go.mod
│   └── main                   # Compiled binary
│
├── model_engineering/         # Python ML
│   ├── application/           # Application layer
│   │   ├── callbacks/        # Training callbacks
│   │   ├── cmd/              # Training commands
│   │   ├── dataset/          # Data loading
│   │   ├── preprocessing/    # Image preprocessing
│   │   └── utils/            # Utilities
│   ├── domain/               # Domain models
│   │   ├── Vgg16.py
│   │   ├── ResNet152.py
│   │   ├── Vit.py
│   │   └── Swim.py
│   ├── infrastructure/       # K8s configs
│   ├── main.py               # Entry point
│   └── requirements.txt
│
└── README.md
```

### Common Patterns

#### Device Management
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### Model Checkpointing
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, path)
```

#### Logging with TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(save_path)
writer.add_scalar("Loss/train", loss, epoch)
```

## Melhorias Futuras

### Repositório de Resultados (Runs) ✓

Checkpoints e logs podem ser salvos em um repositório Git LFS separado para versionamento centralizado.

#### Configuração

```bash
# Setup automático (cria repo GitHub + clone local + Git LFS)
cd model_engineering
bash setup_runs_repo.sh

# Ou manual:
# 1. gh repo create checkioname/psoriasis-runs --private
# 2. gh repo clone checkioname/psoriasis-runs ~/psoriasis-runs
# 3. cd ~/psoriasis-runs && git lfs track "*.pt" "*.pth" "*.pkl"
```

#### Uso no treino

```bash
# Salvar no repositório versionado + commit automático
python main.py runs_repo.path=$HOME/psoriasis-runs runs_repo.auto_push=true

# Salvar local (comportamento padrão)
python main.py
```

#### Ver resultados localmente

```bash
cd ~/psoriasis-runs && git pull && tensorboard --logdir .
```

#### Como funciona

- `RunsRepo` (`application/utils/runs_repo.py`) gerencia clone/init, commit e push
- `main.py` detecta `runs_repo.path` na config Hydra e usa `runs_repo.run_dir()` para `save_path` e `SummaryWriter`
- Ao final do treino (main process apenas), faz `git add -A && git commit -m "feat: ..." && git push`
- Arquivos `.pt`, `.pth`, `.pkl` rastreados por Git LFS (binários grandes não poluem o repo)



### Multi-GPU ✓

```bash
# Single GPU (baseline)
python main.py -e 50 -m vgg16

# DataParallel (multi-GPU, single process)
python main.py -e 50 -m resnet152 --dp --batch-size 64 --num-workers 8

# DistributedDataParallel (multi-GPU, torchrun)
torchrun --nproc_per_node=3 main.py -e 50 -m vgg16 --batch-size 64 --num-workers 8
```

#### Otimizações aplicadas
- **AMP** (`autocast` + `GradScaler`) no loop de treino
- **DDP**: `DistributedSampler` + `DistributedDataParallel` via `torchrun`
- **DataLoader**: `num_workers=4`, `pin_memory=True`, `prefetch_factor=4`, `persistent_workers=True`
- **Grad-CAM espaçado**: executa a cada `epochs // 5` épocas
- **Preprocessing offline**: opcional via `data.preprocessed_dir` (elimina CPU-bound do DataLoader)

### Scheduler Flexível ✓

`SetupModel.SCHEDULERS` contém os 3 schedulers (`plateau`, `step`, `cosine`).
A escolha é feita via Hydra: `training.scheduler=cosine`.
Cada scheduler recebe os parâmetros corretos automaticamente:
- `plateau` → `ReduceLROnPlateau(optimizer, mode='min', patience=5)`
- `step` → `StepLR(optimizer, step_size=5, gamma=0.1)`
- `cosine` → `CosineAnnealingLR(optimizer, T_max=epochs)`

### VSCode Configuration

The project uses VSCode with pytest enabled. See `.vscode/settings.json`:
- Python path includes `./application/cmd`
- Pytest is the test runner
