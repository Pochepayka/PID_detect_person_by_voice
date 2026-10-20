"""Identification: compare a WAV against stored voiceprints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import load_wav
from .config import DATA_PROFILES, DEFAULT_THRESHOLD, SPEAKER_NAMES
from .features import cosine_similarity, extract_voiceprint


@dataclass
class IdentifyResult:
    speaker_id: str | None
    display_name: str
    score: float
    accepted: bool
    all_scores: dict[str, float]


def load_profiles(profiles_dir: Path = DATA_PROFILES) -> dict[str, np.ndarray]:
    profiles: dict[str, np.ndarray] = {}
    if not profiles_dir.exists():
        return profiles
    for path in sorted(profiles_dir.glob("*.npy")):
        profiles[path.stem] = np.load(path)
    return profiles


def identify_audio(
    audio: np.ndarray,
    sr: int,
    profiles: dict[str, np.ndarray] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> IdentifyResult:
    if profiles is None:
        profiles = load_profiles()
    if not profiles:
        raise RuntimeError("No enrolled profiles found. Run enrollment first.")

    query = extract_voiceprint(audio, sr=sr)
    scores = {sid: cosine_similarity(query, vec) for sid, vec in profiles.items()}
    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]
    accepted = best_score >= threshold

    if accepted:
        display = SPEAKER_NAMES.get(best_id, best_id)
        return IdentifyResult(best_id, display, best_score, True, scores)

    return IdentifyResult(None, "Не распознан", best_score, False, scores)


def identify_file(
    wav_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    profiles_dir: Path = DATA_PROFILES,
) -> IdentifyResult:
    audio, sr = load_wav(wav_path)
    profiles = load_profiles(profiles_dir)
    return identify_audio(audio, sr, profiles=profiles, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify speaker from WAV file")
    parser.add_argument("wav", type=Path, help="Path to WAV file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold (default {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    result = identify_file(args.wav, threshold=args.threshold)
    print(f"Result: {result.display_name}")
    print(f"Best score: {result.score:.4f} (threshold={args.threshold})")
    print("Scores:")
    for sid, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        name = SPEAKER_NAMES.get(sid, sid)
        mark = "*" if sid == result.speaker_id else " "
        print(f"  {mark} {name:16s} {sid:12s} {score:.4f}")


if __name__ == "__main__":
    main()
