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
python model_engineering/main.py -e 10 -m vgg16
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

#### Flags
| Flag | Default | Descrição |
|---|---|---|
| `-e` | obrigatório | Número de épocas |
| `-m` | obrigatório | Arquitetura: vgg16, resnet152, vit, swin |
| `--batch-size` | 32 | Tamanho do batch (por GPU) |
| `--num-workers` | 4 | Workers do DataLoader |
| `--dp` | off | Ativa DataParallel (multi-GPU single process) |

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

### Repositório de Resultados (Runs)

Atualmente os checkpoints e logs do TensorBoard são salvos em `model_engineering/runs/` localmente no servidor. Para centralizar e versionar os resultados, recomenda-se:

1. Criar um repositório separado (ex.: `psoriasis-runs`) com Git LFS para arquivos `.pt`
2. No servidor, clonar esse repo e apontar o `save_path` e `SummaryWriter` para dentro dele
3. Ao final do treino, fazer commit + push automático
4. Localmente, puxar e abrir TensorBoard: `git pull && tensorboard --logdir runs/`

Isso elimina rsync manual e mantém histórico de todas as execuções.

### Aceleração Multi-GPU

O treino atual é single-GPU. Para escalar para múltiplas GPUs:

#### Gargalos Identificados

| Gargalo | Estado Atual | Impacto |
|---|---|---|
| **DataLoader workers** | `num_workers=1` | CPU subutilizada, GPU espera dados |
| **Mixed Precision (AMP)** | Código comentado/removido | Perde ~2x speedup em GPUs Turing+ |
| **Batch size** | Fixo 32 | Não escala com mais GPUs |
| **Paralelismo** | Sem `DataParallel`/`DDP` | Apenas 1 GPU utilizada |
| **Prefetch** | Nenhum | Pipeline CPU-GPU não otimizado |
| **Pin Memory** | Não configurado | Transferência CPU→GPU lenta |
| **OpenCV Preprocessing** | CPU-bound por imagem | Deep Learning preprocessing (denoise+equalize) bloqueia DataLoader |
| **Grad-CAM** | Executado toda época | Inferência extra que pode ser espaçada |

#### Plano de Otimização

**1. Baixo esforço / Alto impacto:**

```python
# DataLoader (PreProcessing.py)
train_loader = DataLoader(..., batch_size=batch_size * num_gpus,
                          num_workers=4, pin_memory=True, prefetch_factor=4)
```

- `num_workers=4`: paraleliza carregamento + preprocessing das imagens
- `pin_memory=True`: acelera transferência CPU→GPU
- `prefetch_factor=4`: pré-carrega batches enquanto GPU processa

**2. Mixed Precision (AMP):**

```python
scaler = GradScaler()
with autocast():
    pred = self.model(X)
    loss = loss_fn(pred, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Ativar AMP dá ~1.5-2x mais throughput em GPUs com Tensor Cores (RTX 30xx+, A-series, V100).

**3. DistributedDataParallel (DDP) — Multi-GPU:**

```bash
# torchrun lida com init_process_group automaticamente
torchrun --nproc_per_node=N main.py -e 50 -m vgg16
```

Mudanças no código:
- `SetupModel._initialize_model`: após criar o modelo, wrap com `DDP(model)`
- `Batch size`: escalar linearmente com número de GPUs (ex.: 32 por GPU → 128 com 4 GPUs)
- `WeightedRandomSampler`: `DistributedSampler` substitui para evitar duplicação entre GPUs
- `Loss`: usar `mean` reduction (já é o padrão do `BCELoss`)

**4. Otimização do Preprocessing (CPU-bound):**

O `OpenCVPreprocessing` (denoise + equalização) é a operação mais pesada do DataLoader. Opções:
- Mover para um dataset pré-processado offline (salvar versões denoised/equalizadas no disco)
- Usar `torchvision.transforms.v2` com suporte nativo a GPU
- Aumentar `num_workers` proporcional ao número de cores CPU

**5. Grad-CAM espaçado:**

Em vez de toda época, executar a cada N épocas (ex.: `if epoch % 5 == 0`). O `writer.add_image` já aceita qualquer step, não precisa ser sequencial.

**6. Estimativa de Speedup:**

| Configuração | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|
| Baseline (atual) | 1x | — | — |
| + AMP + workers | 1.8x | — | — |
| + DDP + AMP + workers | — | 3.2x | 5.5x |

*Estimativas conservadoras. O ganho real depende do modelo (VGG16 escala melhor que ViT/Swin em multi-GPU devido à comunicação entre GPUs).*

### Scheduler Flexível

O `SetupModel` atualmente cria `ReduceLROnPlateau` enquanto cada modelo específico (Vgg16, ResNet152, etc.) cria `StepLR` próprio que é descartado. Para permitir troca fácil de scheduler via CLI:

```python
# SetupModel.py
SCHEDULERS = {
    'plateau': lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5),
    'step':    lambda opt: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1),
    'cosine':  lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs),
}

# main.py
parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'step', 'cosine'])
```

### VSCode Configuration

The project uses VSCode with pytest enabled. See `.vscode/settings.json`:
- Python path includes `./application/cmd`
- Pytest is the test runner
