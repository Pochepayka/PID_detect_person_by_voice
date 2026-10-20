"""Normalization and simple energy-based VAD."""

from __future__ import annotations

import numpy as np

from .config import FRAME_MS, HOP_MS, MIN_SPEECH_FRAMES, PRE_EMPHASIS, SAMPLE_RATE, VAD_ENERGY_RATIO


def pre_emphasize(audio: np.ndarray, coef: float = PRE_EMPHASIS) -> np.ndarray:
    """y[n] = x[n] - a * x[n-1] — boost high frequencies (classic speech prep)."""
    if len(audio) < 2:
        return audio.astype(np.float32)
    out = np.empty_like(audio, dtype=np.float32)
    out[0] = audio[0]
    out[1:] = audio[1:] - coef * audio[:-1]
    return out


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) + 1e-12
    return (audio / peak).astype(np.float32)


def frame_signal(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: float = FRAME_MS,
    hop_ms: float = HOP_MS,
) -> np.ndarray:
    """Return overlapping frames shaped (n_frames, frame_len)."""
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len = int(round(sr * hop_ms / 1000.0))
    if len(audio) < frame_len:
        pad = np.zeros(frame_len, dtype=np.float32)
        pad[: len(audio)] = audio
        return pad[np.newaxis, :]

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = audio[start : start + frame_len]
    return frames


def energy_vad_mask(frames: np.ndarray, ratio: float = VAD_ENERGY_RATIO) -> np.ndarray:
    """Boolean mask: keep frames with energy >= ratio * max_energy."""
    energy = np.sum(frames**2, axis=1)
    thr = ratio * float(np.max(energy) + 1e-12)
    return energy >= thr


def apply_vad(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Keep only speech-like segments by energy VAD on frames, then concatenate.
    Falls back to full signal if too few speech frames.
    """
    frames = frame_signal(audio, sr=sr)
    mask = energy_vad_mask(frames)
    if int(mask.sum()) < MIN_SPEECH_FRAMES:
        return audio.astype(np.float32)

    hop_len = int(round(sr * HOP_MS / 1000.0))
    frame_len = frames.shape[1]
    kept: list[np.ndarray] = []
    for i, keep in enumerate(mask):
        if keep:
            start = i * hop_len
            kept.append(audio[start : start + frame_len])
    if not kept:
        return audio.astype(np.float32)
    return np.concatenate(kept).astype(np.float32)


def prepare_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full preprocessing chain before feature extraction."""
    x = normalize_peak(audio.astype(np.float32))
    x = apply_vad(x, sr=sr)
    x = pre_emphasize(x)
    x = normalize_peak(x)
    return x
