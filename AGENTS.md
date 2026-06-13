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

# Gerar CSV com paths relativos (imagens em /data/dermatite/, /data/psoriasis/)
docker compose run --rm -v /caminho/das/imagens:/data preprocess /data -o dataset.csv

# Treinar (monta o CSV gerado e as imagens)
docker compose run --rm \
  -v /caminho/das/imagens:/data:ro \
  -v ./model_engineering/dataset.csv:/app/dataset.csv:ro \
  train -e 50 -m vgg16

# Ou em background
docker compose up -d train
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

### VSCode Configuration

The project uses VSCode with pytest enabled. See `.vscode/settings.json`:
- Python path includes `./application/cmd`
- Pytest is the test runner
