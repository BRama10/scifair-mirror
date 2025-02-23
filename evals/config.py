# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BENCHMARK_CONFIGS = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "cache_file": CACHE_DIR / "math500.json"
    },
    "aime24": {
        "dataset": "ScaleFrontierData/aime24",
        "split": "train",
        "cache_file": CACHE_DIR / "aime24.json"
    },
    "gpqa-diamond": {
        "dataset": "Idavidrein/gpqa",
        "subset": "gpqa_diamond",
        "split": "train",
        "cache_file": CACHE_DIR / "gpqa_diamond.json"
    }
}