import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path("/mnt/data/multilingual-audio-models")

# Directory configurations
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "data"
RUNNER_DIR = PROJECT_ROOT / "runners"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist (optional but good practice)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNNER_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Device Settings
DEVICE = "cuda"

# Audio processing settings
SAMPLE_RATE = 16000  # Default sampling rate for most ASR models
MAX_AUDIO_SECONDS = 30
AUDIO_FORMATS = [".wav", ".flac", ".mp3", ".ogg"]

# Inference / Generation settings
BATCH_SIZE = 1
DTYPE = "float16"
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
DO_SAMPLE = False

# Experiment-specific settings (e.g. languages)
# You can populate this list with the languages you are evaluating (e.g., ["en", "es", "fr", "hi"])
TARGET_LANGUAGES = [
    "en",  # English
    # Add other language codes as needed
]
