#!/usr/bin/env python3
"""
cap_tensors.py — Phase 3.5 in isolation.
Randomly samples exactly `target` tensors per group (Bird / Mass / Reptile)
and moves the rest to DATA/held_out_tensors/<group>/.

Usage:
    python pipeline/cap_tensors.py            # live run  (target = Reptile count)
    python pipeline/cap_tensors.py --dry-run  # preview only
    python pipeline/cap_tensors.py --target 149
    python pipeline/cap_tensors.py --restore  # move everything back
"""

import argparse
import os
import random
import shutil
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TENSOR_DIR  = os.path.join(ROOT, "DATA", "tensors")
HELD_DIR    = os.path.join(ROOT, "DATA", "held_out_tensors")

# ── Group → class prefixes ────────────────────────────────────────────────────
GROUPS = {
    "Bird":    ["Tinamou_Tinamus", "Tinamou_Crypturellus", "Kiwi",
                "Cassowary", "Rhea", "Emu", "Ostrich"],
    "Reptile": ["Crocodylia"],
    "Mass":    ["Whippomorpha", "Elephantidae", "Phocoidea"],
}

# ANSI colours
C = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def classify_tensor(fname: str) -> str | None:
    for group, prefixes in GROUPS.items():
        for p in prefixes:
            if fname.startswith(p):
                return group
    return None


def collect_tensors() -> dict[str, list[str]]:
    """Return {group: [absolute_path, …]} for every .pt under TENSOR_DIR/<class>/."""
    buckets: dict[str, list[str]] = {g: [] for g in GROUPS}
    if not os.path.isdir(TENSOR_DIR):
        sys.exit(f"ERROR: tensor dir not found: {TENSOR_DIR}")
    for cls_dir in sorted(os.listdir(TENSOR_DIR)):
        cls_path = os.path.join(TENSOR_DIR, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        group = next(
            (g for g, prefixes in GROUPS.items()
             if any(cls_dir.startswith(p) for p in prefixes)),
            None,
        )
        if group is None:
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.endswith(".pt"):
                buckets[group].append(os.path.join(cls_path, fname))
    return buckets


def collect_held() -> dict[str, list[str]]:
    """Return {group: [absolute_path, …]} for every .pt under HELD_DIR/<group>/."""
    buckets: dict[str, list[str]] = {g: [] for g in GROUPS}
    for group in GROUPS:
        held_group = os.path.join(HELD_DIR, group)
        if not os.path.isdir(held_group):
            continue
        # Held tensors may be stored flat or in class subdirs
        for item in sorted(os.listdir(held_group)):
            item_path = os.path.join(held_group, item)
            if os.path.isdir(item_path):
                for fname in sorted(os.listdir(item_path)):
                    if fname.endswith(".pt"):
                        buckets[group].append(os.path.join(item_path, fname))
            elif item.endswith(".pt"):
                buckets[group].append(item_path)
    return buckets


def print_counts(label: str, buckets: dict[str, list[str]]) -> None:
    bar_max = 40
    total   = sum(len(v) for v in buckets.values())
    peak    = max((len(v) for v in buckets.values()), default=1)
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    for group, paths in buckets.items():
        n   = len(paths)
        bar = "█" * int(bar_max * n / max(peak, 1))
        print(f"  {group:<10} {n:>6}  {bar}")
    print(f"  {'TOTAL':<10} {total:>6}")


# ── cap ────────────────────────────────────────────────────────────────────────

def cap(target: int | None = None, dry_run: bool = False, seed: int = 42) -> None:
    random.seed(seed)

    active = collect_tensors()
    print_counts("Tensors — Before Capping", active)

    # Determine target from Reptile floor
    reptile_count = len(active["Reptile"])
    if reptile_count == 0:
        sys.exit("ERROR: No Reptile tensors found. Check DATA/tensors/.")
    if target is None:
        target = reptile_count
    print(f"\n  Target per group : {target}  (Reptile floor = {reptile_count})")

    total_moved = 0
    for group, paths in active.items():
        n = len(paths)
        if n <= target:
            print(f"  {group:<10} {n:>5} tensors — at or below target, no capping")
            continue

        # Random sample: KEEP target, MOVE the rest
        random.shuffle(paths)
        keep = set(paths[:target])
        move = [p for p in paths if p not in keep]
        held_group_dir = os.path.join(HELD_DIR, group)

        print(f"\n  {group:<10} {n:>5} tensors → keep {target}, move {len(move)}")
        if dry_run:
            examples = [os.path.basename(p) for p in move[:3]]
            print(f"    [DRY-RUN] would move {len(move)} tensors → {held_group_dir}")
            for ex in examples:
                print(f"      {ex}")
            if len(move) > 3:
                print(f"      … and {len(move)-3} more")
        else:
            os.makedirs(held_group_dir, exist_ok=True)
            for src in move:
                # Preserve class subdirectory structure inside held_out_tensors/<group>/<class>/
                cls_name = os.path.basename(os.path.dirname(src))
                cls_held = os.path.join(held_group_dir, cls_name)
                os.makedirs(cls_held, exist_ok=True)
                dst = os.path.join(cls_held, os.path.basename(src))
                shutil.move(src, dst)
            total_moved += len(move)
            print(f"    {C['green']}Moved {len(move)} tensors → {held_group_dir}{C['reset']}")

    if not dry_run:
        active_after = collect_tensors()
        print_counts("Tensors — After Capping", active_after)
        print(f"\n  {C['green']}{C['bold']}Done. Moved {total_moved} tensors total.{C['reset']}")
        print(f"  Held-out tensors : {HELD_DIR}")
        print(f"  Restore with     : python pipeline/cap_tensors.py --restore\n")
    else:
        print(f"\n  [DRY-RUN] No files moved.")


# ── restore ───────────────────────────────────────────────────────────────────

def restore(group_filter: str | None = None) -> None:
    held = collect_held()
    total = 0
    groups = [group_filter] if group_filter else list(GROUPS.keys())
    for group in groups:
        paths = held.get(group, [])
        if not paths:
            print(f"  {group}: nothing held out.")
            continue
        for src in paths:
            # Infer class from parent directory name
            cls_name = os.path.basename(os.path.dirname(src))
            cls_dir  = os.path.join(TENSOR_DIR, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            dst = os.path.join(cls_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.move(src, dst)
                total += 1
        print(f"  {group}: restored {len(paths)} tensors")
    print(f"\n  Total restored: {total}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cap/restore tensors for group balance")
    parser.add_argument("--target",    type=int, default=None,
                        help="Tensors to KEEP per group (default: Reptile count)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Preview without moving files")
    parser.add_argument("--restore",   action="store_true",
                        help="Move all held-out tensors back to DATA/tensors/")
    parser.add_argument("--restore-group", default=None,
                        help="Restore only one group (Bird|Mass|Reptile)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for sampling (default 42)")
    args = parser.parse_args()

    if args.restore:
        restore(args.restore_group)
    else:
        cap(target=args.target, dry_run=args.dry_run, seed=args.seed)


if __name__ == "__main__":
    main()
