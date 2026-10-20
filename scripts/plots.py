#!/usr/bin/env python3
"""Generate spectrogram / MFCC figures for the explanatory note."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_io import load_wav  # noqa: E402
from src.config import DATA_RAW, HOP_MS  # noqa: E402
from src.features import mfcc_from_audio  # noqa: E402
from src.preprocess import frame_signal, prepare_audio  # noqa: E402


OUT_DIR = ROOT / "docs" / "figures"


def plot_waveform(path: Path, out: Path) -> None:
    audio, sr = load_wav(path)
    t = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t, audio, color="#1f4e79", linewidth=0.6)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда")
    ax.set_title(f"Волновая форма: {path.parent.name}/{path.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spectrogram(path: Path, out: Path) -> None:
    audio, sr = load_wav(path)
    audio = prepare_audio(audio, sr=sr)
    frames = frame_signal(audio, sr=sr)
    window = np.hamming(frames.shape[1])
    n_fft = 1 << int(np.ceil(np.log2(frames.shape[1])))
    spec = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2
    spec_db = 10 * np.log10(np.maximum(spec.T, 1e-12))
    hop = int(round(sr * HOP_MS / 1000.0))
    extent = [0, frames.shape[0] * hop / sr, 0, sr / 2]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(spec_db, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    ax.set_title(f"Спектрограмма (FFT): {path.parent.name}")
    fig.colorbar(im, ax=ax, label="дБ")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mfcc(path: Path, out: Path) -> None:
    audio, sr = load_wav(path)
    mfcc = mfcc_from_audio(audio, sr=sr)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mfcc.T, origin="lower", aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Кадр")
    ax.set_ylabel("Коэффициент MFCC")
    ax.set_title(f"MFCC: {path.parent.name}/{path.name}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_voiceprints_compare(out: Path) -> None:
    from src.features import extract_voiceprint

    ids = ["speaker_01", "speaker_02", "speaker_03"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(13)
    width = 0.25
    for i, sid in enumerate(ids):
        wav = next((DATA_RAW / sid).glob("enroll_*.wav"))
        audio, sr = load_wav(wav)
        vp = extract_voiceprint(audio, sr=sr)
        ax.bar(x + i * width, vp, width=width, label=sid)
    ax.set_xlabel("Индекс MFCC")
    ax.set_ylabel("Значение (норм.)")
    ax.set_title("Сравнение голосовых слепков (mean MFCC)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = DATA_RAW / "speaker_01" / "enroll_01.wav"
    if not sample.exists():
        raise SystemExit("Dataset missing. Run: python scripts/generate_demo_dataset.py")

    plot_waveform(sample, OUT_DIR / "waveform_speaker01.png")
    plot_spectrogram(sample, OUT_DIR / "spectrogram_speaker01.png")
    plot_mfcc(sample, OUT_DIR / "mfcc_speaker01.png")
    for sid in ("speaker_02", "speaker_03"):
        p = DATA_RAW / sid / "enroll_01.wav"
        plot_spectrogram(p, OUT_DIR / f"spectrogram_{sid}.png")
    plot_voiceprints_compare(OUT_DIR / "voiceprints_compare.png")
    print(f"Figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
