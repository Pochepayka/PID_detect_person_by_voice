"""WAV load / save with resampling to the project sample rate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import SAMPLE_RATE


def load_wav(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono in [-1, 1], resampled to target_sr."""
    path = Path(path)
    sr, data = wavfile.read(path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
        peak = np.max(np.abs(audio)) + 1e-12
        if peak > 1.5:
            audio = audio / peak

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        audio = signal.resample(audio, n_out).astype(np.float32)
        sr = target_sr

    return audio, sr


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save float mono audio as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    wavfile.write(path, sr, pcm)
