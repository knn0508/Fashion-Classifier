from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "fashion-dataset"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_TEXT_MODEL = "bert-base-uncased"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MAX_TEXT_LENGTH = 128
DEFAULT_TOP_K = 10
DEFAULT_SEED = 42
