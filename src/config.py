"""Fixed pipeline parameters (document in the report as well)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROFILES = PROJECT_ROOT / "data" / "profiles"
DATA_TEST = PROJECT_ROOT / "data" / "test"

SAMPLE_RATE = 16_000
FRAME_MS = 25.0
HOP_MS = 10.0
N_MELS = 26
N_MFCC = 13
INCLUDE_C0 = False  # use MFCC 1..13 (timbre-focused); keep fixed
PRE_EMPHASIS = 0.97
VAD_ENERGY_RATIO = 0.05  # keep frames above this fraction of max frame energy
MIN_SPEECH_FRAMES = 10

# Cosine similarity threshold for acceptance (tune via evaluate / metrics --sweep)
DEFAULT_THRESHOLD = 0.92

SPEAKER_NAMES = {
    "speaker_01": "Участник 1",
    "speaker_02": "Участник 2",
    "speaker_03": "Участник 3",
}
