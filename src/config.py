# src/config.py
from pathlib import Path

# Root directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42
MODEL_FILENAME = "cnn_model.pth"

# Ensure folders exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
