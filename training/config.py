import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Model
BASE_MODEL = "TrevorJS/gemma-4-E4B-it-uncensored"
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LOAD_IN_4BIT = True

# Paths
DATA_DIR = PROJECT_ROOT / "data"
SCRAPED_DIR = DATA_DIR / "scraped"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
ADAPTER_NAME = "persona-v1"
OUTPUT_DIR = ADAPTERS_DIR / ADAPTER_NAME

# Synthetic data generation
SYNTHETIC_MODEL = os.environ.get("SYNTHETIC_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Scraping API keys
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
