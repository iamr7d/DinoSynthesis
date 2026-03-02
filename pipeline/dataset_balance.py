"""
dataset_balance.py — Strategic Dataset Equalization for DinoSynthesis
======================================================================
Orchestrates the full quality-over-quantity pipeline in four phases:

  Phase 1  DOWNLOAD   — top-up Crocodylia from iNaturalist (multi-taxon)
                         and optionally Freesound (if FREESOUND_API_KEY set).
  Phase 2  SNR GRADE  — estimate signal-to-noise for every audio file,
                         quarantine Grade-D files.
  Phase 3  EQUALIZE   — cap over-represented groups so all three synthesis
                         anchors (Bird / Reptile / Mass) match the smallest.
  Phase 4  PREPROCESS — re-run DinoDataPipeline on the balanced set.

Synthesis group mapping
-----------------------
  Bird    : Tinamou_Tinamus, Tinamou_Crypturellus, Kiwi, Cassowary,
            Rhea, Emu, Ostrich
  Reptile : Crocodylia
  Mass    : Whippomorpha, Elephantidae, Phocoidea

Equalization rule
-----------------
  target = min(500, N_reptile_after_grading)

  If Reptile < 500:
    Cap each other group to the same count by moving the lowest-SNR
    files from over-represented classes into DATA/held_out/<Group>/.
    This guarantees a perfectly balanced training set WITHOUT deleting
    any recordings (held-out files can be restored later).

Usage
-----
  # Full pipeline (downloads + grades + balances + preprocesses):
  python pipeline/dataset_balance.py

  # Just audit — no changes:
  python pipeline/dataset_balance.py --audit-only

  # Skip download (use what you already have):
  python pipeline/dataset_balance.py --skip-download

  # Skip preprocessing (balance only, re-preprocess manually later):
  python pipeline/dataset_balance.py --skip-preprocess

  # Preview what would move without touching anything:
  python pipeline/dataset_balance.py --dry-run

  # Set explicit target (overrides the min-reptile logic):
  python pipeline/dataset_balance.py --target 150
"""

import os
import sys
import json
import shutil
import argparse
import time
from pathlib import Path

import numpy as np

# ── Make project root importable ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pipeline.snr_filter import load_audio, estimate_snr, grade as snr_grade

# ── Configuration ─────────────────────────────────────────────────────────────
AUDIO_DIR    = os.path.join(ROOT, "DATA", "birds")
TENSOR_DIR   = os.path.join(ROOT, "DATA", "tensors")
HELD_OUT_DIR = os.path.join(ROOT, "DATA", "held_out")
SNR_CACHE    = os.path.join(ROOT, "DATA", "snr_cache.json")

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

# Synthesis group definitions (order = priority for keeping when capping)
GROUPS: dict[str, list[str]] = {
    "Bird": [
        "Tinamou_Tinamus",         # primary bird anchor — highest phylo priority
        "Tinamou_Crypturellus",    # secondary tinamou anchor
        "Kiwi",
        "Cassowary",
        "Rhea",
        "Emu",
        "Ostrich",
    ],
    "Reptile": [
        "Crocodylia",              # sole reptile anchor
    ],
    "Mass": [
        "Whippomorpha",            # primary mass anchor
        "Phocoidea",
        "Elephantidae",
    ],
}

# Grade ordering for sorting (A = best)
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}

# ANSI colours
C = {
    "green":  "\033[92m",
    "cyan":   "\033[96m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def audio_files(class_dir: str) -> list[str]:
    """Return sorted list of audio file paths in a class directory."""
    return sorted([
        str(p) for p in Path(class_dir).iterdir()
        if p.suffix.lower() in AUDIO_EXTS
    ]) if Path(class_dir).is_dir() else []


def group_count(group_name: str) -> int:
    """Count total audio files in all classes of a group."""
    total = 0
    for cls in GROUPS[group_name]:
        total += len(audio_files(os.path.join(AUDIO_DIR, cls)))
    return total


def print_header(title: str):
    bar = "═" * 64
    print(f"\n{C['bold']}{bar}")
    print(f"  {title}")
    print(f"{bar}{C['reset']}")


def print_counts(label: str = "Current dataset"):
    print_header(label)
    for group, classes in GROUPS.items():
        total = 0
        print(f"\n  {C['bold']}{group}{C['reset']}")
        for cls in classes:
            n = len(audio_files(os.path.join(AUDIO_DIR, cls)))
            bar = "█" * min(40, n // 5)
            total += n
            col = C["green"] if n >= 100 else (C["yellow"] if n >= 40 else C["red"])
            print(f"    {cls:<28} {col}{n:>4}{C['reset']}  {bar}")
        col = C["green"] if total >= 300 else C["yellow"]
        print(f"    {'TOTAL':<28} {col}{total:>4}{C['reset']}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Download
# ─────────────────────────────────────────────────────────────────────────────

def phase_download():
    print_header("Phase 1 — Download More Crocodylia")

    # ── iNaturalist (no API key needed) ───────────────────────────────────────
    from pipeline.ia_downloader import run as ia_run
    ia_run(output_dir=AUDIO_DIR, target_per_class=500)

    # ── Freesound (optional — only if API key present) ────────────────────────
    import dotenv
    dotenv.load_dotenv(os.path.join(ROOT, ".env"))
    fs_key = os.getenv("FREESOUND_API_KEY", "").strip()
    if fs_key:
        print(f"\n{C['cyan']}  Freesound API key found — downloading additional crocodilians …{C['reset']}")
        from pipeline.freesound_downloader import FreesoundDownloader
        dl = FreesoundDownloader(api_key=fs_key, output_dir=AUDIO_DIR)
        dl.download_data(target_per_folder=500)
    else:
        print(f"\n  {C['yellow']}No FREESOUND_API_KEY in .env — skipping Freesound download.")
        print(f"  Add one at freesound.org/api/apply/ to access ~200 extra crocodilian recordings.{C['reset']}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — SNR Grading
# ─────────────────────────────────────────────────────────────────────────────

def load_snr_cache() -> dict[str, dict]:
    """
    Returns {filepath: {"snr_db": float, "grade": str}} from disk cache.
    """
    if os.path.exists(SNR_CACHE):
        with open(SNR_CACHE) as f:
            return json.load(f)
    return {}


def save_snr_cache(cache: dict[str, dict]):
    os.makedirs(os.path.dirname(SNR_CACHE), exist_ok=True)
    with open(SNR_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def phase_snr_grade(quarantine_d: bool = True) -> dict[str, dict]:
    """
    Grade every audio file in AUDIO_DIR.  Results cached to SNR_CACHE.
    Returns full cache dict.
    """
    print_header("Phase 2 — SNR Grading")
    cache = load_snr_cache()
    new_count = 0

    all_classes = [cls for group in GROUPS.values() for cls in group]
    for cls in all_classes:
        cls_dir = os.path.join(AUDIO_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = audio_files(cls_dir)
        if not files:
            continue

        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        print(f"\n  Grading [{cls}]  ({len(files)} files) …", flush=True)

        for path in files:
            if path in cache:
                g = cache[path]["grade"]
            else:
                try:
                    y      = load_audio(path)
                    snr_db = estimate_snr(y)
                    g      = snr_grade(snr_db)
                    cache[path] = {
                        "snr_db": None if np.isnan(snr_db) else round(float(snr_db), 2),
                        "grade":  g,
                    }
                    new_count += 1
                except Exception as e:
                    cache[path] = {"snr_db": None, "grade": "D"}
                    g = "D"
                    print(f"    [warn] {os.path.basename(path)}: {e}")
            grade_counts[g] += 1

        total = len(files)
        for g, cnt in grade_counts.items():
            col = {"A": C["green"], "B": C["cyan"],
                   "C": C["yellow"], "D": C["red"]}[g]
            pct = 100 * cnt / total if total else 0
            print(f"    Grade {col}{g}{C['reset']}:  {cnt:4d} ({pct:5.1f}%)")

        if quarantine_d:
            moved = _quarantine_grade_d(cls, files, cache)
            if moved:
                print(f"    {C['yellow']}⮞ Quarantined {moved} Grade-D files{C['reset']}")

    if new_count > 0:
        save_snr_cache(cache)
        print(f"\n  {C['green']}SNR cache updated (+{new_count} new entries) → {SNR_CACHE}{C['reset']}")
    else:
        print(f"\n  All grades loaded from cache.")

    return cache


def _quarantine_grade_d(cls: str, files: list[str], cache: dict) -> int:
    """Move Grade-D files for a class to DATA_quarantine/<cls>/."""
    qdir  = os.path.join(ROOT, "DATA_quarantine", cls)
    moved = 0
    for path in files:
        if cache.get(path, {}).get("grade") == "D":
            os.makedirs(qdir, exist_ok=True)
            dst = os.path.join(qdir, os.path.basename(path))
            if not os.path.exists(dst) and os.path.exists(path):
                shutil.move(path, dst)
                moved += 1
    return moved


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Equalization
# ─────────────────────────────────────────────────────────────────────────────

def _ranked_files_for_group(group: str, cache: dict) -> list[tuple[str, str, float]]:
    """
    Return list of (path, grade, snr_db) for every kept audio file in a group,
    sorted best-SNR first.
    """
    rows: list[tuple[str, str, float]] = []
    for cls in GROUPS[group]:
        cls_dir = os.path.join(AUDIO_DIR, cls)
        for path in audio_files(cls_dir):
            info   = cache.get(path, {"grade": "C", "snr_db": 0.0})
            g      = info.get("grade", "C")
            snr_db = info.get("snr_db") or 0.0
            rows.append((path, g, float(snr_db)))
    # Sort: grade ascending (A < B < C), then SNR descending within grade
    rows.sort(key=lambda r: (GRADE_ORDER.get(r[1], 3), -r[2]))
    return rows


def phase_equalize(target: int | None = None,
                   dry_run: bool = False,
                   cache: dict | None = None) -> int:
    """
    Cap each group to `target` files (lowest-SNR overflow → held_out/).
    Returns the effective target used.
    """
    print_header("Phase 3 — Equalization")
    if cache is None:
        cache = load_snr_cache()

    # ── Determine target ──────────────────────────────────────────────────────
    reptile_count = group_count("Reptile")
    if target is None:
        target = min(500, reptile_count)

    print(f"\n  Reptile files on disk : {reptile_count}")
    print(f"  {C['bold']}Equalization target   : {target}{C['reset']}")
    if reptile_count < 500:
        print(f"  {C['yellow']}  (capped by Reptile count — {500 - reptile_count} short of 500){C['reset']}")

    if target == 0:
        print(f"  {C['red']}No Reptile files found — run Phase 1 download first.{C['reset']}")
        return 0

    # ── Process each group ────────────────────────────────────────────────────
    for group in ("Bird", "Mass"):
        ranked  = _ranked_files_for_group(group, cache)
        current = len(ranked)
        excess  = current - target

        print(f"\n  {C['bold']}{group}{C['reset']}  ({current} files → target {target})")
        if excess <= 0:
            col = C["green"] if current >= target * 0.8 else C["yellow"]
            print(f"    {col}No capping needed ({current} ≤ {target}){C['reset']}")
            continue

        # Files to move = the LAST `excess` in ranked list (lowest SNR)
        to_move = [path for path, _, _ in ranked[target:]]
        held_dir = os.path.join(HELD_OUT_DIR, group)

        if dry_run:
            print(f"    {C['cyan']}[DRY-RUN] Would move {excess} lowest-SNR files → {held_dir}{C['reset']}")
            for path, g, snr_db in ranked[target:target + 5]:
                print(f"      {g}  {snr_db:6.1f} dB  {os.path.basename(path)}")
            if excess > 5:
                print(f"      … and {excess - 5} more")
        else:
            os.makedirs(held_dir, exist_ok=True)
            moved = 0
            for path in to_move:
                dst = os.path.join(held_dir, os.path.basename(path))
                if os.path.exists(path):
                    shutil.move(path, dst)
                    moved += 1
            print(f"    {C['yellow']}↳ Moved {moved} files → {held_dir}{C['reset']}")

    # ── Reptile check ─────────────────────────────────────────────────────────
    print(f"\n  {C['bold']}Reptile{C['reset']}  ({reptile_count} files — no capping needed, this is the floor)")

    # ── Final report ──────────────────────────────────────────────────────────
    print_header("Balance After Equalization")
    for group, classes in GROUPS.items():
        total = group_count(group)
        col   = C["green"] if abs(total - target) <= 5 else C["yellow"]
        print(f"  {group:<10}  {col}{total:>4} files{C['reset']}  "
              f"(target {target})")

    return target


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.5 — Tensor equalization
# ─────────────────────────────────────────────────────────────────────────────

def phase_cap_tensors(audio_target: int, dry_run: bool = False):
    """
    Cap each synthesis GROUP's tensor count to match the Reptile group.

    The number of tensors per audio file varies (recording length / 3 s chunk).
    We derive a per-group tensor target by scaling the audio_target proportionally
    to each group's current tensors-per-audio-file ratio, then cap the Bird and
    Mass groups by moving the excess (highest-index, i.e. last-recorded) tensors
    to DATA/held_out_tensors/<group>/.

    This keeps the training set balanced even before the WeightedRandomSampler
    runs, and gives honest class-balance numbers in the thesis.
    """
    print_header("Phase 3.5 — Tensor Equalization")

    HELD_TENSORS = os.path.join(ROOT, "DATA", "held_out_tensors")

    def tensor_files(cls: str) -> list[str]:
        cdir = Path(TENSOR_DIR) / cls
        return sorted(str(p) for p in cdir.glob("*.pt")) if cdir.is_dir() else []

    def group_tensor_count(group: str) -> int:
        return sum(len(tensor_files(c)) for c in GROUPS[group])

    reptile_tensors = group_tensor_count("Reptile")

    # Derive tensor target: same ratio as audio target × tensors-per-file for
    # Reptile. If Reptile has 149 tensors from 95 files = 1.57 t/f, then
    # target tensors = audio_target × 1.57 — capped at current count.
    audio_reptile = len(audio_files(os.path.join(AUDIO_DIR, "Crocodylia")))
    tpf_reptile   = reptile_tensors / max(1, audio_reptile)   # tensors-per-file
    tensor_target = max(reptile_tensors, int(audio_target * tpf_reptile))

    print(f"\n  Reptile tensors      : {reptile_tensors}")
    print(f"  Tensors-per-file     : {tpf_reptile:.2f}")
    print(f"  {C['bold']}Tensor target/group  : {tensor_target}{C['reset']}")

    for group in ("Bird", "Mass"):
        all_tensors = []
        for cls in GROUPS[group]:
            all_tensors.extend(tensor_files(cls))
        # Sort by filename so highest-index chunks (last files) come last
        all_tensors.sort()
        current = len(all_tensors)
        excess  = current - tensor_target

        print(f"\n  {C['bold']}{group}{C['reset']}  ({current} tensors → target {tensor_target})")
        if excess <= 0:
            col = C["green"] if current >= tensor_target * 0.8 else C["yellow"]
            print(f"    {col}No capping needed ({current} ≤ {tensor_target}){C['reset']}")
            continue

        to_move = all_tensors[tensor_target:]   # the tail = highest-index = last files
        held_dir = os.path.join(HELD_TENSORS, group)

        if dry_run:
            print(f"    {C['cyan']}[DRY-RUN] Would move {excess} tensors → {held_dir}{C['reset']}")
            for p in to_move[:4]:
                print(f"      {os.path.basename(p)}")
            if excess > 4:
                print(f"      … and {excess - 4} more")
        else:
            os.makedirs(held_dir, exist_ok=True)
            moved = 0
            for path in to_move:
                dst = os.path.join(held_dir, os.path.basename(path))
                if os.path.exists(path):
                    shutil.move(path, dst)
                    moved += 1
            print(f"    {C['yellow']}↳ Moved {moved} tensors → {held_dir}{C['reset']}")

    # ── Reptile ────────────────────────────────────────────────────────────────
    print(f"\n  {C['bold']}Reptile{C['reset']}  ({reptile_tensors} tensors — floor, no capping)")

    # ── Summary table ──────────────────────────────────────────────────────────
    print_header("Tensor Balance After Phase 3.5")
    for group in ("Bird", "Reptile", "Mass"):
        n   = group_tensor_count(group)
        col = C["green"] if abs(n - tensor_target) <= tensor_target * 0.1 else C["yellow"]
        print(f"  {group:<10}  {col}{n:>6} tensors{C['reset']}  (target ~{tensor_target})")




def phase_preprocess():
    print_header("Phase 4 — Preprocessing (Audio → Tensors)")
    from pipeline.dino_data_pipeline import DinoDataPipeline

    # Only preprocess classes that have changed (new audio or rebalanced)
    changed_classes = set()
    for group_classes in GROUPS.values():
        for cls in group_classes:
            audio_count  = len(audio_files(os.path.join(AUDIO_DIR, cls)))
            tensor_count = len(list(Path(TENSOR_DIR, cls).glob("*.pt"))
                                if Path(TENSOR_DIR, cls).exists() else [])
            # Heuristic: if audio files × ~8 chunks ≠ tensor count, reprocess
            expected_tensors = audio_count * 8
            if abs(tensor_count - expected_tensors) > audio_count:
                changed_classes.add(cls)

    if not changed_classes:
        print("  All tensor counts look current. Skipping preprocessing.")
        print("  (Run dino_data_pipeline.py directly to force re-preprocess.)")
        return

    print(f"  Classes needing preprocessing: {sorted(changed_classes)}")
    pipeline = DinoDataPipeline(input_dir=AUDIO_DIR, output_dir=TENSOR_DIR)

    for cls in sorted(changed_classes):
        import glob
        mp3s = sorted(glob.glob(os.path.join(AUDIO_DIR, cls, "*.mp3")) +
                      glob.glob(os.path.join(AUDIO_DIR, cls, "*.wav")) +
                      glob.glob(os.path.join(AUDIO_DIR, cls, "*.ogg")))
        print(f"\n  [{cls}]  {len(mp3s)} audio files …")
        total = 0
        for idx, p in enumerate(mp3s):
            total += pipeline.process_file(p, cls, idx)
        print(f"    → {total} tensors saved")


# ─────────────────────────────────────────────────────────────────────────────
# Restore held-out files (utility)
# ─────────────────────────────────────────────────────────────────────────────

def restore_held_out(group: str | None = None):
    """
    Move all held-out files back to their original class directories.
    Useful when you get more Reptile recordings and want to re-balance.
    """
    print_header("Restoring Held-Out Files")
    groups_to_restore = [group] if group else list(GROUPS.keys())

    for grp in groups_to_restore:
        held_dir = os.path.join(HELD_OUT_DIR, grp)
        if not os.path.isdir(held_dir):
            print(f"  {grp}: nothing held out.")
            continue
        files    = [f for f in os.listdir(held_dir)
                    if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
        restored = 0
        for fname in files:
            # Infer class from filename prefix (e.g. Tinamou_Tinamus_0012_snd.mp3)
            cls = _infer_class(fname)
            if cls:
                dst_dir = os.path.join(AUDIO_DIR, cls)
                os.makedirs(dst_dir, exist_ok=True)
                src = os.path.join(held_dir, fname)
                dst = os.path.join(dst_dir, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    restored += 1
        print(f"  {grp}: restored {restored} files")


def _infer_class(filename: str) -> str | None:
    """Guess the class directory from a filename by matching known class names."""
    all_classes = [cls for grp in GROUPS.values() for cls in grp]
    for cls in all_classes:
        if filename.startswith(cls):
            return cls
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DinoSynthesis dataset balancing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else ""
    )
    parser.add_argument("--audit-only",      action="store_true",
                        help="Just print current counts, no changes")
    parser.add_argument("--skip-download",   action="store_true",
                        help="Skip Phase 1 (iNat / Freesound download)")
    parser.add_argument("--skip-snr",        action="store_true",
                        help="Skip Phase 2 (use cached SNR grades if available)")
    parser.add_argument("--skip-equalize",   action="store_true",
                        help="Skip Phase 3 (audio equalization/capping)")
    parser.add_argument("--skip-cap-tensors", action="store_true",
                        help="Skip Phase 3.5 (tensor equalization)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip Phase 4 (tensor preprocessing)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Phase 3 preview only — no files moved")
    parser.add_argument("--target",          type=int, default=None,
                        help="Override equalization target (default: min(500, reptile_count))")
    parser.add_argument("--no-quarantine",   action="store_true",
                        help="Grade files but do NOT quarantine Grade-D files")
    parser.add_argument("--restore",         action="store_true",
                        help="Restore held-out files and exit")
    parser.add_argument("--restore-group",   default=None,
                        choices=list(GROUPS.keys()),
                        help="Restore only this group from held_out/")
    args = parser.parse_args()

    # ── Show current state ────────────────────────────────────────────────────
    print_counts("DinoSynthesis Dataset — Current State")

    if args.restore:
        restore_held_out(group=args.restore_group)
        return

    if args.audit_only:
        return

    t0 = time.time()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not args.skip_download:
        phase_download()
        print_counts("After Download")
    else:
        print(f"\n{C['yellow']}  [skip] Phase 1 — Download{C['reset']}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if not args.skip_snr:
        cache = phase_snr_grade(quarantine_d=not args.no_quarantine)
    else:
        print(f"\n{C['yellow']}  [skip] Phase 2 — SNR Grading (loading cache){C['reset']}")
        cache = load_snr_cache()

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if not args.skip_equalize:
        effective_target = phase_equalize(
            target=args.target, dry_run=args.dry_run, cache=cache)
    else:
        print(f"\n{C['yellow']}  [skip] Phase 3 — Equalization{C['reset']}")
        effective_target = args.target or group_count("Reptile")

    # ── Phase 3.5 ─────────────────────────────────────────────────
    if not args.skip_cap_tensors and effective_target:
        phase_cap_tensors(audio_target=effective_target, dry_run=args.dry_run)
    else:
        reason = "--skip-cap-tensors" if args.skip_cap_tensors else "no target"
        print(f"\n{C['yellow']}  [skip] Phase 3.5 — Tensor Equalization ({reason}){C['reset']}")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    if not args.skip_preprocess and not args.dry_run:
        phase_preprocess()
    else:
        reason = "dry-run" if args.dry_run else "--skip-preprocess"
        print(f"\n{C['yellow']}  [skip] Phase 4 — Preprocessing ({reason}){C['reset']}")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print_header(f"Pipeline Complete  ({elapsed:.0f}s)")
    print_counts("Final Dataset State")
    if effective_target:
        print(f"\n  {C['green']}Target per group : {effective_target}{C['reset']}")
    print(f"\n  Held-out audio   : {HELD_OUT_DIR}")
    print(f"  Held-out tensors : {os.path.join(ROOT, 'DATA', 'held_out_tensors')}")
    print(f"  SNR cache        : {SNR_CACHE}")
    print(f"  Tensors          : {TENSOR_DIR}")
    print(f"\n  To restore held-out files when you add more Reptile recordings:")
    print(f"  {C['cyan']}python pipeline/dataset_balance.py --restore{C['reset']}\n")


if __name__ == "__main__":
    main()
