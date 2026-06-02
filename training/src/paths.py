from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
BACKBONE_DIR = MODELS_DIR / "backbone"
CLASSIFIER_PATH = MODELS_DIR / "auvin.onnx"
VOICES_DIR = MODELS_DIR / "voices"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
CONFIG_PATH = ROOT / "training" / "configs" / "auvin.yaml"
LOGS_DIR = ROOT / "training" / "logs"

OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

BACKBONE_FILES = [
    "melspectrogram.onnx",
    "embedding_model.onnx",
    "silero_vad.onnx",
]
