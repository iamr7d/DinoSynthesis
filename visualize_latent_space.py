#!/usr/bin/env python3
"""
visualize_latent_space.py  —  Latent Universe Map for the DinoSynthesis project.

Encodes every tensor in DATA/tensors into latent μ vectors, runs PCA to 2D,
and plots the three anchor-group clusters plus the synthesised Dinosaur midpoint.

Usage:
    python visualize_latent_space.py
    python visualize_latent_space.py --checkpoint checkpoints/best.pt
    python visualize_latent_space.py --blend 0.5 0.3 0.2   # Bird Reptile Mass weights
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                      # no display required
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA

# ── Resolve imports from this repo's structure ───────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.dino_vae import DinoVAE          # noqa: E402
from train import CLASS_GROUPS              # noqa: E402

# ── Constants (must match train.py) ──────────────────────────────────────────
TIME_FRAMES  = 256
TENSOR_DIR   = os.path.join(ROOT, "DATA", "tensors")
CHECKPOINT   = os.path.join(ROOT, "checkpoints", "best.pt")
OUTPUT_DIR   = os.path.join(ROOT, "research_output")

GROUP_COLOURS = {
    "Bird":    "#00f2ff",   # cyan
    "Reptile": "#ff2255",   # red
    "Mass":    "#aa44ff",   # violet
}
DINO_COLOUR  = "#ffffff"
DINO_MARKER  = "*"
BG_COLOUR    = "#05070a"
GRID_COLOUR  = "#1a1f2e"


# ── Data Loading ──────────────────────────────────────────────────────────────

def _group_from_path(path: str) -> str | None:
    """Return 'Bird', 'Reptile', or 'Mass' for a tensor file path."""
    cls_dir = os.path.basename(os.path.dirname(path))
    return CLASS_GROUPS.get(cls_dir)


def encode_dataset(
    checkpoint: str,
    tensor_dir: str,
    latent_dim: int = 128,
    device: torch.device | None = None,
    max_per_group: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (latents, group_labels) where latents is (N, latent_dim) float32.
    Caps samples per group to `max_per_group` so PCA is not dominated by one group
    even if DATA/tensors is not yet perfectly balanced.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DinoVAE(latent_dim=latent_dim).to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    # Checkpoint may be a raw state_dict or wrapped in {"model": state_dict, ...}
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"  Model loaded from {checkpoint}")

    # Collect paths and stratify by group
    rng     = random.Random(seed)
    by_group: dict[str, list[str]] = {g: [] for g in GROUP_COLOURS}

    for cls_dir in sorted(os.listdir(tensor_dir)):
        cls_path = os.path.join(tensor_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        group = CLASS_GROUPS.get(cls_dir)
        if group is None:
            continue
        files = sorted(f for f in os.listdir(cls_path) if f.endswith(".pt"))
        by_group[group].extend(os.path.join(cls_path, f) for f in files)

    # Random sample up to cap
    sampled_paths = []
    sampled_groups = []
    for group, paths in by_group.items():
        subset = rng.sample(paths, min(len(paths), max_per_group))
        sampled_paths.extend(subset)
        sampled_groups.extend([group] * len(subset))
        print(f"  {group:<10}: {len(paths):>4} tensors total, encoding {len(subset)}")

    # Encode
    latents = []
    print(f"  Encoding {len(sampled_paths)} tensors …")
    with torch.no_grad():
        for path in sampled_paths:
            t = torch.load(path, weights_only=True)   # (1, 128, T)
            # Deterministic centre crop / pad
            T = t.shape[-1]
            if T >= TIME_FRAMES:
                start = (T - TIME_FRAMES) // 2
                t = t[:, :, start : start + TIME_FRAMES]
            else:
                t = F.pad(t, (0, TIME_FRAMES - T))
            t = t.unsqueeze(0).to(device)             # (1, 1, 128, 256)
            _, mu, _ = model(t)
            latents.append(mu.squeeze(0).cpu().numpy())

    return np.array(latents, dtype=np.float32), sampled_groups


# ── PCA + Plot ────────────────────────────────────────────────────────────────

def plot(
    latents: np.ndarray,
    group_labels: list[str],
    blend: tuple[float, float, float] = (0.5, 0.5, 0.0),
    out_path: str | None = None,
) -> str:
    """
    Project latents to 2D via PCA, compute the synthesis midpoint, and save.

    blend = (bird_w, reptile_w, mass_w) — controls where the star lands.
    """
    print("  Running PCA (128D → 2D) …")
    pca    = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(latents)

    labels_arr = np.array(group_labels)
    centroids  = {}
    for group in GROUP_COLOURS:
        idx = labels_arr == group
        if idx.any():
            centroids[group] = coords[idx].mean(axis=0)

    # Blend weights (Bird, Reptile, Mass)
    bird_w, reptile_w, mass_w = blend
    total = bird_w + reptile_w + mass_w
    dino_pt = (
        (bird_w    / total) * centroids.get("Bird",    np.zeros(2)) +
        (reptile_w / total) * centroids.get("Reptile", np.zeros(2)) +
        (mass_w    / total) * centroids.get("Mass",    np.zeros(2))
    )

    var_explained = pca.explained_variance_ratio_ * 100

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG_COLOUR)
    ax.set_facecolor(BG_COLOUR)

    # Subtle grid
    ax.grid(True, color=GRID_COLOUR, linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Scatter per group
    for group, colour in GROUP_COLOURS.items():
        idx = labels_arr == group
        n   = idx.sum()
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=colour, label=f"{group} (n={n})",
            alpha=0.45, s=18, linewidths=0,
        )
        # Centroid marker
        if group in centroids:
            cx, cy = centroids[group]
            ax.scatter(cx, cy, c=colour, marker="D", s=90, zorder=4, linewidths=0)
            ax.annotate(
                group,
                (cx, cy), (cx + 0.25, cy + 0.25),
                color=colour, fontsize=9, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    # Triangle between centroids
    if len(centroids) == 3:
        tri_pts = np.array([centroids["Bird"], centroids["Reptile"], centroids["Mass"], centroids["Bird"]])
        ax.plot(tri_pts[:, 0], tri_pts[:, 1],
                color="#ffffff", linewidth=0.7, linestyle="--", alpha=0.3, zorder=3)

    # Dino synthesis star
    ax.scatter(
        dino_pt[0], dino_pt[1],
        c=DINO_COLOUR, marker=DINO_MARKER, s=420,
        edgecolors="#ffff00", linewidths=0.8,
        label=f"DINO SYNTHESIS\n(B={bird_w:.0%} R={reptile_w:.0%} M={mass_w:.0%})",
        zorder=10,
    )
    ax.annotate(
        "z_dino",
        (dino_pt[0], dino_pt[1]),
        (dino_pt[0] + 0.35, dino_pt[1] - 0.35),
        color="#ffff00", fontsize=9,
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    # Axes labels & title
    ax.set_xlabel(f"PC-1  ({var_explained[0]:.1f}% variance)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel(f"PC-2  ({var_explained[1]:.1f}% variance)", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOUR)

    ax.set_title(
        "Phylogenetic Latent Universe — The Synthesis Triangle",
        color="white", fontsize=15, pad=18, fontweight="bold",
    )

    legend = ax.legend(
        facecolor="#111520", edgecolor="#333355",
        labelcolor="white", fontsize=9,
        loc="lower right", framealpha=0.85,
    )

    # Variance caption
    total_var = sum(var_explained)
    fig.text(
        0.01, 0.01,
        f"PCA explains {total_var:.1f}% of total latent variance",
        color="#555555", fontsize=7, va="bottom",
    )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "latent_map.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOUR)
    plt.close(fig)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VAE latent space")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--tensor-dir", default=TENSOR_DIR)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--blend", type=float, nargs=3, default=[0.5, 0.5, 0.0],
                        metavar=("BIRD", "REPTILE", "MASS"),
                        help="Blend weights for the synthesis midpoint (default: 0.5 0.5 0.0)")
    parser.add_argument("--max-per-group", type=int, default=500,
                        help="Maximum tensors per group to encode (speed vs. density)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    latents, group_labels = encode_dataset(
        checkpoint=args.checkpoint,
        tensor_dir=args.tensor_dir,
        latent_dim=args.latent_dim,
        device=device,
        max_per_group=args.max_per_group,
        seed=args.seed,
    )

    out_path = plot(
        latents=latents,
        group_labels=group_labels,
        blend=tuple(args.blend),
        out_path=args.out,
    )

    print(f"\n  Latent map saved → {out_path}\n")


if __name__ == "__main__":
    main()
