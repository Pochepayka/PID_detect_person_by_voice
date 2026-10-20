"""
MFCC feature extraction from scratch (numpy / scipy FFT).

Pipeline (explainable on defense):
  frames -> Hamming -> FFT power spectrum -> Mel filterbank -> log -> DCT -> MFCC
Links to lectures: FFT/spectrum (L13), timbre via spectral envelope (L4), sampling (L3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import FRAME_MS, HOP_MS, INCLUDE_C0, N_MELS, N_MFCC, SAMPLE_RATE
from .preprocess import frame_signal, prepare_audio


def hz_to_mel(hz: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    n_fft: int,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular Mel filterbank matrix of shape (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = min(left + 1, n_freqs - 1)
        if right == center:
            right = min(center + 1, n_freqs - 1)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    # Avoid all-zero filters
    fb_sum = fb.sum(axis=1, keepdims=True)
    fb_sum[fb_sum == 0] = 1.0
    return fb / fb_sum


def dct_ii(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Type-II DCT along last axis (orthonormal-ish scaling sufficient for matching)."""
    n = x.shape[-1]
    k = np.arange(n_coeffs)[:, None]
    n_idx = np.arange(n)[None, :]
    basis = np.cos(np.pi / n * (n_idx + 0.5) * k)
    return x @ basis.T


def power_spectrum(frames: np.ndarray) -> np.ndarray:
    """Hamming-windowed |FFT|^2 for each frame."""
    frame_len = frames.shape[1]
    n_fft = 1 << int(np.ceil(np.log2(frame_len)))
    window = np.hamming(frame_len).astype(np.float32)
    windowed = frames * window
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float64)
    return power, n_fft


def mfcc_from_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_mfcc: int = N_MFCC,
    include_c0: bool = INCLUDE_C0,
    do_prepare: bool = True,
) -> np.ndarray:
    """
    Compute MFCC matrix of shape (n_frames, n_mfcc).
    If include_c0 is False, returns coefficients 1..n_mfcc (skips energy-like C0).
    """
    if do_prepare:
        audio = prepare_audio(audio, sr=sr)

    frames = frame_signal(audio, sr=sr, frame_ms=FRAME_MS, hop_ms=HOP_MS)
    power, n_fft = power_spectrum(frames)
    fb = mel_filterbank(n_fft, sr=sr, n_mels=n_mels)
    mel_energies = np.maximum(power @ fb.T, 1e-10)
    log_mel = np.log(mel_energies)

    # Need n_mfcc+1 coeffs if we drop C0
    n_take = n_mfcc if include_c0 else n_mfcc + 1
    coeffs = dct_ii(log_mel, n_take)
    if include_c0:
        return coeffs[:, :n_mfcc].astype(np.float32)
    return coeffs[:, 1 : n_mfcc + 1].astype(np.float32)


def voiceprint_from_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Voiceprint = mean MFCC vector over speech frames (L2-normalized)."""
    if mfcc.size == 0:
        raise ValueError("Empty MFCC matrix")
    mean_vec = mfcc.mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean_vec) + 1e-12)
    return mean_vec / norm


def extract_voiceprint(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full path: audio -> MFCC -> mean voiceprint."""
    mfcc = mfcc_from_audio(audio, sr=sr)
    return voiceprint_from_mfcc(mfcc)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)
