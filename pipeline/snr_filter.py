"""
snr_filter.py — Audio Signal-to-Noise Ratio Grader
====================================================
Estimates the broadband SNR of each raw audio file in DATA/birds/ using
a noise-floor percentile method and assigns a letter grade.  Low-grade
files are quarantined rather than deleted, so you can inspect them.

SNR Grade Table
---------------
  A  ≥ 20 dB  — studio / clean field recording, near-silent background
  B  ≥ 12 dB  — slight wind/hum, still clearly usable
  C  ≥  6 dB  — moderate noise; usable but reduces model clarity
  D  <  6 dB  — noise-dominated; quarantined by default

Method
------
  1. Load audio at 22 050 Hz mono.
  2. Compute short-time RMS energy per 50 ms frame.
  3. Noise floor = median of the lowest 20 % of frames (quietest regions).
  4. Signal level = mean of the top 30 % of frames (most active regions).
  5. SNR_dB = 20 * log10(signal_rms / noise_rms).

Usage
-----
  python pipeline/snr_filter.py [--data-dir DATA/birds]
                                [--grade-min B]
                                [--quarantine]
                                [--report]
                                [--class Crocodylia]
"""

import os
import sys
import argparse
import shutil
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
SR         = 22_050
FRAME_LEN  = int(SR * 0.050)  # 50 ms frames

GRADE_THRESHOLDS = [
    ("A", 20.0),
    ("B", 12.0),
    ("C",  6.0),
    ("D", -999),   # catch-all
]

ANSI = {
    "A": "\033[92m",   # green
    "B": "\033[96m",   # cyan
    "C": "\033[93m",   # yellow
    "D": "\033[91m",   # red
    "reset": "\033[0m",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    """Load any audio file to a 22 050 Hz mono float32 array."""
    try:
        y, sr = librosa.load(path, sr=SR, mono=True, res_type="kaiser_fast")
        return y
    except Exception as e:
        raise RuntimeError(f"librosa failed: {e}") from e


def estimate_snr(y: np.ndarray) -> float:
    """
    Estimate broadband SNR in dB using a percentile noise-floor method.
    Returns float('nan') for silent / too-short files.
    """
    if len(y) < FRAME_LEN * 4:
        return float("nan")

    # Compute per-frame RMS
    n_frames = len(y) // FRAME_LEN
    frames   = y[: n_frames * FRAME_LEN].reshape(n_frames, FRAME_LEN)
    rms      = np.sqrt(np.mean(frames ** 2, axis=1))

    # Guard: all-silent file
    if rms.max() < 1e-9:
        return float("nan")

    # Noise floor: median of bottom 20 % quietest frames
    n_noise     = max(1, int(n_frames * 0.20))
    noise_floor = np.median(np.sort(rms)[:n_noise])

    # Signal level: mean of top 30 % loudest frames
    n_signal     = max(1, int(n_frames * 0.30))
    signal_level = np.mean(np.sort(rms)[-n_signal:])

    # Guard division by zero
    if noise_floor < 1e-9:
        return 40.0   # essentially no noise floor → excellent

    snr_db = 20.0 * np.log10(signal_level / noise_floor)
    return float(snr_db)


def grade(snr_db: float) -> str:
    if np.isnan(snr_db):
        return "D"
    for letter, threshold in GRADE_THRESHOLDS:
        if snr_db >= threshold:
            return letter
    return "D"


# ─────────────────────────────────────────────────────────────────────────────
def audit_class(class_dir: str) -> list[dict]:
    """Analyse every audio file in a class directory. Returns list of result dicts."""
    results = []
    files   = sorted([
        p for p in Path(class_dir).iterdir()
        if p.suffix.lower() in AUDIO_EXTS
    ])
    for p in files:
        try:
            y      = load_audio(str(p))
            snr_db = estimate_snr(y)
            g      = grade(snr_db)
        except RuntimeError as e:
            snr_db = float("nan")
            g      = "D"
            print(f"    [warn] {p.name}: {e}")
        results.append({"path": str(p), "name": p.name, "snr_db": snr_db, "grade": g})
    return results


def print_class_report(class_name: str, results: list[dict]):
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in results:
        grade_counts[r["grade"]] += 1
    total = len(results)
    usable = grade_counts["A"] + grade_counts["B"] + grade_counts["C"]
    print(f"\n  {'─'*56}")
    print(f"  Class: {class_name}  ({total} files)")
    for g, cnt in grade_counts.items():
        pct  = 100 * cnt / total if total else 0
        bar  = "█" * int(pct / 4)
        col  = ANSI[g]
        rst  = ANSI["reset"]
        print(f"  {col}Grade {g}{rst}  {cnt:4d} ({pct:5.1f}%) {bar}")
    print(f"  Usable (A+B+C): {usable}/{total}  |  "
          f"Quarantine (D): {grade_counts['D']}/{total}")
    print(f"  {'─'*56}")


# ─────────────────────────────────────────────────────────────────────────────
def quarantine_files(results: list[dict], quarantine_dir: str,
                     grade_min: str = "C") -> int:
    """Move files graded below grade_min into quarantine_dir. Returns count moved."""
    grade_order = ["A", "B", "C", "D"]
    min_idx     = grade_order.index(grade_min)
    os.makedirs(quarantine_dir, exist_ok=True)
    moved = 0
    for r in results:
        if grade_order.index(r["grade"]) > min_idx:
            dst = os.path.join(quarantine_dir, os.path.basename(r["path"]))
            if not os.path.exists(dst):
                shutil.move(r["path"], dst)
                moved += 1
    return moved


# ─────────────────────────────────────────────────────────────────────────────
def run_audit(data_dir: str, target_classes: list[str] | None = None,
              grade_min: str = "C", do_quarantine: bool = False,
              save_json: str | None = None) -> dict:
    """
    Main entry point for the SNR audit.

    Returns dict mapping class_name → list of result dicts.
    """
    data_path = Path(data_dir)
    class_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (target_classes is None or d.name in target_classes)
    ])

    all_results = {}
    grand_keep = grand_quarantine = 0

    for cdir in class_dirs:
        print(f"\nAnalysing [{cdir.name}] …", flush=True)
        results = audit_class(str(cdir))
        all_results[cdir.name] = results
        print_class_report(cdir.name, results)

        if do_quarantine:
            qdir  = str(data_path.parent / "DATA_quarantine" / cdir.name)
            moved = quarantine_files(results, qdir, grade_min=grade_min)
            if moved:
                print(f"  ⮞ Quarantined {moved} Grade-D files → {qdir}")
            keep = len([r for r in results
                        if ["A","B","C","D"].index(r["grade"])
                           <= ["A","B","C","D"].index(grade_min)])
            grand_keep       += keep
            grand_quarantine += moved
        else:
            grand_keep += len(results)

    print(f"\n{'═'*60}")
    print(f"  SNR AUDIT SUMMARY")
    print(f"  Files analysed : {sum(len(v) for v in all_results.values())}")
    if do_quarantine:
        print(f"  Kept           : {grand_keep}")
        print(f"  Quarantined    : {grand_quarantine}")
    print(f"{'═'*60}\n")

    if save_json:
        serialisable = {
            cls: [
                {**r, "snr_db": None if np.isnan(r["snr_db"]) else round(r["snr_db"], 2)}
                for r in recs
            ]
            for cls, recs in all_results.items()
        }
        with open(save_json, "w") as fh:
            json.dump(serialisable, fh, indent=2)
        print(f"  JSON report saved → {save_json}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SNR-grade every audio file in DATA/birds/")
    parser.add_argument("--data-dir",  default="DATA/birds",
                        help="Root directory of raw audio classes")
    parser.add_argument("--class",     dest="only_class", default=None,
                        help="Only analyse this one class (e.g. Crocodylia)")
    parser.add_argument("--grade-min", default="C", choices=["A", "B", "C", "D"],
                        help="Minimum passing grade (default: C)")
    parser.add_argument("--quarantine", action="store_true",
                        help="Move Grade-D files to DATA_quarantine/")
    parser.add_argument("--report",    default=None,
                        help="Save full per-file SNR report to this JSON path")
    args = parser.parse_args()

    target = [args.only_class] if args.only_class else None

    run_audit(
        data_dir      = args.data_dir,
        target_classes = target,
        grade_min      = args.grade_min,
        do_quarantine  = args.quarantine,
        save_json      = args.report,
    )
