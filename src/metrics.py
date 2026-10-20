"""Evaluation metrics: Accuracy, FAR, FRR."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_PROFILES, DATA_RAW, DEFAULT_THRESHOLD
from .identify import identify_file, load_profiles


@dataclass
class EvalReport:
    threshold: float
    n_target: int
    n_impostor: int
    accuracy: float
    far: float
    frr: float
    correct: int
    false_accept: int
    false_reject: int


def collect_test_files(raw_root: Path = DATA_RAW) -> list[tuple[Path, str | None]]:
    """
    Returns list of (path, true_speaker_id or None for impostor).
    Prefers *test*.wav; if none, uses files not used as enroll_*.
    """
    items: list[tuple[Path, str | None]] = []
    for speaker_dir in sorted(raw_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        wavs = sorted(speaker_dir.glob("*.wav"))
        tests = [p for p in wavs if "test" in p.stem.lower()]
        if not tests and speaker_dir.name.lower() != "impostor":
            tests = [p for p in wavs if "enroll" not in p.stem.lower()]
        if speaker_dir.name.lower() == "impostor":
            for p in wavs:
                items.append((p, None))
        else:
            for p in tests:
                items.append((p, speaker_dir.name))
    return items


def evaluate(
    threshold: float = DEFAULT_THRESHOLD,
    raw_root: Path = DATA_RAW,
    profiles_dir: Path = DATA_PROFILES,
) -> EvalReport:
    profiles = load_profiles(profiles_dir)
    if not profiles:
        raise RuntimeError("No profiles. Run: python -m src.enroll --all")

    items = collect_test_files(raw_root)
    if not items:
        raise RuntimeError("No test WAV files found under data/raw")

    correct = 0
    false_accept = 0
    false_reject = 0
    n_target = 0
    n_impostor = 0

    for path, true_id in items:
        result = identify_file(path, threshold=threshold, profiles_dir=profiles_dir)
        if true_id is None:
            n_impostor += 1
            if result.accepted:
                false_accept += 1
            else:
                correct += 1  # correctly rejected
        else:
            n_target += 1
            if not result.accepted:
                false_reject += 1
            elif result.speaker_id == true_id:
                correct += 1
            else:
                # accepted as wrong speaker — count as false accept-like error
                false_accept += 1

    total = len(items)
    accuracy = correct / total if total else 0.0
    far = false_accept / n_impostor if n_impostor else (
        false_accept / total if total else 0.0
    )
    # If no dedicated impostors, FAR uses wrong-accept among all; document this.
    if n_impostor == 0:
        far = false_accept / total if total else 0.0
    frr = false_reject / n_target if n_target else 0.0

    return EvalReport(
        threshold=threshold,
        n_target=n_target,
        n_impostor=n_impostor,
        accuracy=accuracy,
        far=far,
        frr=frr,
        correct=correct,
        false_accept=false_accept,
        false_reject=false_reject,
    )


def sweep_thresholds(
    thresholds: list[float] | None = None,
    raw_root: Path = DATA_RAW,
    profiles_dir: Path = DATA_PROFILES,
) -> list[EvalReport]:
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    return [evaluate(t, raw_root=raw_root, profiles_dir=profiles_dir) for t in thresholds]


def write_metrics_csv(reports: list[EvalReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["threshold", "accuracy", "FAR", "FRR", "n_target", "n_impostor", "correct", "FA", "FR"]
        )
        for r in reports:
            writer.writerow(
                [
                    f"{r.threshold:.2f}",
                    f"{r.accuracy:.4f}",
                    f"{r.far:.4f}",
                    f"{r.frr:.4f}",
                    r.n_target,
                    r.n_impostor,
                    r.correct,
                    r.false_accept,
                    r.false_reject,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate identification metrics")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds and write CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/metrics_table.csv"),
        help="CSV output for --sweep",
    )
    args = parser.parse_args()

    if args.sweep:
        reports = sweep_thresholds()
        write_metrics_csv(reports, args.out)
        print(f"Wrote {args.out}")
        for r in reports:
            print(
                f"thr={r.threshold:.2f}  Acc={r.accuracy:.3f}  "
                f"FAR={r.far:.3f}  FRR={r.frr:.3f}"
            )
    else:
        r = evaluate(args.threshold)
        print(f"Threshold: {r.threshold:.2f}")
        print(f"Target trials: {r.n_target}, Impostor trials: {r.n_impostor}")
        print(f"Accuracy: {r.accuracy:.3f}")
        print(f"FAR:      {r.far:.3f}")
        print(f"FRR:      {r.frr:.3f}")


if __name__ == "__main__":
    main()
