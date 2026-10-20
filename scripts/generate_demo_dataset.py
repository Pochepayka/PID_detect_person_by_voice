#!/usr/bin/env python3
"""
Generate a synthetic demo voice dataset with distinct spectral "timbres".

Real brigade recordings should replace these files later (same naming scheme):
  data/raw/<speaker_id>/enroll_01.wav ... enroll_06.wav
  data/raw/<speaker_id>/test_01.wav ... test_04.wav
  data/raw/impostor/test_01.wav ...

Each synthetic speaker uses a unique set of formant-like resonances so MFCC
voiceprints separate reliably for the educational demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_io import save_wav  # noqa: E402
from src.config import DATA_RAW, SAMPLE_RATE  # noqa: E402


def _glottal_pulse_train(n: int, f0: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Simple impulse train with slight jitter (voiced excitation)."""
    t = np.arange(n) / sr
    phase = 0.0
    x = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        period = int(sr / max(f0 * (1.0 + 0.01 * rng.normal()), 60.0))
        if i < n:
            x[i] = 1.0
        i += max(period, 1)
        phase += period
    # soft pulse shape
    from scipy.signal import lfilter

    x = lfilter([1.0], [1.0, -0.95], x)
    return x.astype(np.float32)


def _formant_filter(x: np.ndarray, sr: int, formants: list[tuple[float, float]]) -> np.ndarray:
    """Cascade of resonators (freq_hz, bandwidth_hz)."""
    from scipy.signal import lfilter

    y = x.astype(np.float64)
    for f, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        b = [1.0]
        y = lfilter(b, a, y)
    return y.astype(np.float32)


def synthesize_utterance(
    speaker_cfg: dict,
    duration: float,
    seed: int,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    f0 = float(speaker_cfg["f0"] * (1.0 + 0.03 * rng.normal()))
    exc = _glottal_pulse_train(n, f0, sr, rng)
    y = _formant_filter(exc, sr, speaker_cfg["formants"])
    # amplitude envelope with silence pads (VAD practice)
    env = np.ones(n, dtype=np.float32)
    pad = int(0.15 * sr)
    if 2 * pad < n:
        env[:pad] *= np.linspace(0, 1, pad)
        env[-pad:] *= np.linspace(1, 0, pad)
        env[: int(0.08 * sr)] *= 0.05
        env[-int(0.08 * sr) :] *= 0.05
    y = y * env
    y += 0.002 * rng.normal(size=n).astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-12
    return (0.8 * y / peak).astype(np.float32)


SPEAKERS = {
    # Distinct formant sets → separable MFCC voiceprints for the demo
    "speaker_01": {
        "seed_off": 10,
        "f0": 110.0,
        "formants": [(450, 60), (1450, 90), (2450, 120)],
    },
    "speaker_02": {
        "seed_off": 20,
        "f0": 190.0,
        "formants": [(800, 90), (2000, 120), (3100, 150)],
    },
    "speaker_03": {
        "seed_off": 30,
        "f0": 95.0,
        "formants": [(350, 50), (1050, 80), (1900, 100)],
    },
    "impostor": {
        "seed_off": 90,
        "f0": 155.0,
        "formants": [(1100, 110), (2600, 160), (3800, 180)],
    },
}


def main() -> None:
    enroll_n, test_n = 6, 4
    seed_base = 42

    for sid, cfg in SPEAKERS.items():
        out_dir = DATA_RAW / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        # remove old demo wavs
        for old in out_dir.glob("*.wav"):
            old.unlink()

        off = int(cfg["seed_off"])
        if sid == "impostor":
            for i in range(1, test_n + 1):
                audio = synthesize_utterance(cfg, duration=3.5 + 0.2 * i, seed=seed_base + off + i)
                save_wav(out_dir / f"test_{i:02d}.wav", audio)
            print(f"{sid}: {test_n} impostor tests")
            continue

        for i in range(1, enroll_n + 1):
            audio = synthesize_utterance(cfg, duration=3.5 + 0.15 * i, seed=seed_base + off + i)
            save_wav(out_dir / f"enroll_{i:02d}.wav", audio)
        for i in range(1, test_n + 1):
            audio = synthesize_utterance(
                cfg, duration=3.2 + 0.25 * i, seed=seed_base + 500 + off + i
            )
            save_wav(out_dir / f"test_{i:02d}.wav", audio)
        print(f"{sid}: {enroll_n} enroll + {test_n} test")

    print(f"Demo dataset written under {DATA_RAW}")
    print("Replace with real brigade recordings using the same filenames when ready.")


if __name__ == "__main__":
    main()
