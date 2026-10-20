"""Enrollment: build and store voiceprints from WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .audio_io import load_wav
from .config import DATA_PROFILES, DATA_RAW, SPEAKER_NAMES
from .features import extract_voiceprint


def collect_wavs(speaker_dir: Path) -> list[Path]:
    files = sorted(speaker_dir.glob("*.wav"))
    return [p for p in files if not p.name.startswith(".")]


def enroll_speaker(
    speaker_id: str,
    wav_paths: list[Path] | None = None,
    raw_root: Path = DATA_RAW,
    profiles_dir: Path = DATA_PROFILES,
) -> Path:
    """
    Average voiceprints from multiple enrollment WAVs and save to .npy.
    """
    if wav_paths is None:
        speaker_dir = raw_root / speaker_id
        wav_paths = collect_wavs(speaker_dir)
        # Prefer files tagged enroll_; else use all except test_
        enroll_files = [p for p in wav_paths if "enroll" in p.stem.lower()]
        if enroll_files:
            wav_paths = enroll_files
        else:
            wav_paths = [p for p in wav_paths if "test" not in p.stem.lower()]

    if not wav_paths:
        raise FileNotFoundError(f"No enrollment WAV files for {speaker_id}")

    vectors = []
    for path in wav_paths:
        audio, sr = load_wav(path)
        vectors.append(extract_voiceprint(audio, sr=sr))

    stacked = np.stack(vectors, axis=0)
    profile = stacked.mean(axis=0)
    profile = profile / (np.linalg.norm(profile) + 1e-12)
    profile = profile.astype(np.float32)

    profiles_dir.mkdir(parents=True, exist_ok=True)
    out = profiles_dir / f"{speaker_id}.npy"
    np.save(out, profile)
    return out


def enroll_all(raw_root: Path = DATA_RAW, profiles_dir: Path = DATA_PROFILES) -> list[Path]:
    saved: list[Path] = []
    for speaker_dir in sorted(raw_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        if speaker_dir.name.lower() == "impostor":
            continue
        path = enroll_speaker(speaker_dir.name, raw_root=raw_root, profiles_dir=profiles_dir)
        saved.append(path)
        name = SPEAKER_NAMES.get(speaker_dir.name, speaker_dir.name)
        print(f"Enrolled {name} ({speaker_dir.name}) -> {path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll speakers into voiceprint database")
    parser.add_argument("--speaker", help="Speaker folder name under data/raw (optional)")
    parser.add_argument("--all", action="store_true", help="Enroll all speakers in data/raw")
    args = parser.parse_args()

    if args.all or not args.speaker:
        enroll_all()
    else:
        out = enroll_speaker(args.speaker)
        print(f"Saved profile: {out}")


if __name__ == "__main__":
    main()
