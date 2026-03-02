"""
train.py  —  β-VAE training loop for the Dino-VAE project  (v2 — anti-overfitting)

Anti-overfitting measures:
  • SpecAugment  : FrequencyMasking(15) + TimeMasking(30) applied on GPU
  • Gaussian noise: σ=0.005 on every training input
  • Random gain  : Uniform(0.9, 1.1) amplitude jitter
  • Random crop  : stochastic time-axis crop (in dataset)
  • KL annealing : β ramps from 0 → target over --kl_warmup epochs
  • Free-bits    : 0.0 nats/dim default (no KL floor — avoids posterior collapse)
  • GroupNorm    : in all conv blocks (no batch-size dependency)
  • Residual+SE  : residual + squeeze-excitation in every encoder/decoder block
  • Weight decay : 1e-4 via AdamW
  • OneCycleLR   : warm-up + cosine anneal in one shot
  • Grad clip    : max_norm=1.0
  • Early stopping: patience on val_recon (not total loss — immune to β annealing)
  • Patience begins only after KL warmup completes
  • Mixed precision: torch.amp.autocast (2× GPU speed)
  • Grad accumulation: effective_batch = batch_size × grad_accum

v3 improvements:
  • Multi-scale spectral loss: log1p-domain L1 + spectral convergence @ 3 scales
  • free_bits default = 0.0 (floor was causing KL collapse at 0.5 nats/dim)
  • beta default = 0.5  (was 2.0 — old default over-regularised the posterior)
  • Bug fix: free_bits argument is now actually passed to vae_loss_function

Usage:
    python train.py
    python train.py --epochs 150 --beta 4 --kl_warmup 25
    python train.py --resume checkpoints/best.pt

Outputs (./checkpoints/):
    best.pt          — best validation-loss checkpoint
    epoch_NNN.pt     — periodic checkpoint every --save_every epochs
    loss_log.csv     — full loss breakdown per epoch
"""

import os
import glob
import argparse
import csv
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import torchaudio.transforms as AT
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

from tqdm import tqdm
from model.dino_vae import DinoVAE, vae_loss_function


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis group mapping
# ─────────────────────────────────────────────────────────────────────────────
# Used by make_sampler() to give EQUAL WEIGHT to each anchor group
# (Bird / Reptile / Mass) regardless of how many subclasses or tensor files
# each group contains.  Without this, Bird's 7 subclasses would receive 63 %
# of every batch even with per-class weighting, starving the Reptile anchor.
CLASS_GROUPS: dict[str, str] = {
    "Tinamou_Tinamus":      "Bird",
    "Tinamou_Crypturellus": "Bird",
    "Kiwi":                 "Bird",
    "Cassowary":            "Bird",
    "Rhea":                 "Bird",
    "Emu":                  "Bird",
    "Ostrich":              "Bird",
    "Crocodylia":           "Reptile",
    "Whippomorpha":         "Mass",
    "Elephantidae":         "Mass",
    "Phocoidea":            "Mass",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Loads .pt tensors from DATA/tensors/<class>/*.pt
    Shape: (1, 128, T) — crops / pads to TIME_FRAMES on the fly.

    Training augmentations in __getitem__:
      1. Random crop along time axis
      2. Gaussian noise (σ=0.01)
      3. Random gain (±15%)
    SpecAugment is applied as a batch transform in the training loop
    on the GPU for efficiency.
    """
    TIME_FRAMES = 256

    def __init__(
        self,
        tensor_dir: str,
        split: str = "train",
        val_frac: float = 0.12,
        augment: bool = True,
        noise_std: float = 0.005,
        gain_range: tuple = (0.90, 1.10),
    ):
        self.augment   = augment
        self.noise_std = noise_std
        self.gain_lo, self.gain_hi = gain_range
        self.paths  = []
        self.labels = []
        self.classes = sorted([
            d for d in os.listdir(tensor_dir)
            if os.path.isdir(os.path.join(tensor_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            pts = sorted(glob.glob(os.path.join(tensor_dir, cls, "*.pt")))
            split_at = max(1, int(len(pts) * (1 - val_frac)))
            pts = pts[:split_at] if split == "train" else pts[split_at:]
            for p in pts:
                self.paths.append(p)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        t = torch.load(self.paths[idx], weights_only=True)   # (1, 128, T)

        # 1. Crop / pad time axis
        T = t.shape[-1]
        if T >= self.TIME_FRAMES:
            start = random.randint(0, T - self.TIME_FRAMES) if self.augment else 0
            t = t[:, :, start : start + self.TIME_FRAMES]
        else:
            t = F.pad(t, (0, self.TIME_FRAMES - T))

        if self.augment:
            # 2. Gaussian noise
            t = (t + torch.randn_like(t) * self.noise_std).clamp(0.0, 1.0)
            # 3. Random gain
            t = (t * random.uniform(self.gain_lo, self.gain_hi)).clamp(0.0, 1.0)

        return t, self.labels[idx]

    def make_sampler(self, num_samples: int | None = None) -> WeightedRandomSampler:
        """
        Group-weighted sampler: Bird / Reptile / Mass each contribute
        exactly 1/3 of sampled items per batch, regardless of how many
        subclasses or tensor files each group contains.

        Old per-class weighting gave Bird 7/11 = 63 % of every batch.
        Group weighting gives every anchor exactly 33 %.
        """
        from collections import Counter
        sample_groups = [
            CLASS_GROUPS.get(self.classes[lbl], "Bird") for lbl in self.labels
        ]
        group_counts = Counter(sample_groups)
        n_groups     = len(group_counts)
        # weight_i = 1 / (n_groups * |group(i)|)
        # → each group contributes the same total weight regardless of size
        weights = torch.tensor(
            [1.0 / (n_groups * group_counts[g]) for g in sample_groups],
            dtype=torch.double,
        )
        n = num_samples if num_samples is not None else len(self.labels)
        return WeightedRandomSampler(weights, num_samples=n, replacement=True)

    def group_distribution(self) -> dict[str, dict[str, int]]:
        """Return {group: {class: count}} for the dataset diagnostic print."""
        from collections import defaultdict
        dist: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for lbl in self.labels:
            cls = self.classes[lbl]
            grp = CLASS_GROUPS.get(cls, "Bird")
            dist[grp][cls] += 1
        return {g: dict(d) for g, d in dist.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SpecAugment (applied in-batch on GPU)
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """Frequency + time masking on a (B, 1, F, T) batch."""
    def __init__(self, freq_param: int = 20, time_param: int = 40, num_masks: int = 2):
        super().__init__()
        self.enabled   = _HAS_TORCHAUDIO
        self.num_masks = num_masks
        if _HAS_TORCHAUDIO:
            self.freq_mask = AT.FrequencyMasking(freq_mask_param=freq_param)
            self.time_mask = AT.TimeMasking(time_mask_param=time_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        B, C, F, T = x.shape
        out = x.squeeze(1)              # (B, F, T)
        for _ in range(self.num_masks):
            out = self.freq_mask(out)
            out = self.time_mask(out)
        return out.unsqueeze(1)         # (B, 1, F, T)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta(epoch: int, warmup_epochs: int, target_beta: float) -> float:
    """Linear KL warm-up: β = 0 at epoch 1, = target_beta at epoch warmup_epochs."""
    if warmup_epochs <= 1:
        return target_beta
    return min(target_beta, target_beta * (epoch / warmup_epochs))


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    beta: float,
    scaler,
    augment_fn,
    training: bool,
    grad_accum: int = 1,
    free_bits: float = 0.5,
):
    model.train(training)
    total_loss = total_recon = total_kl = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, (x, _) in enumerate(
            tqdm(loader, desc="  train" if training else "  val  ", leave=False, unit="b")
        ):
            x = x.to(device, non_blocking=True)

            if training and augment_fn is not None:
                x = augment_fn(x)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                recon, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon, x, mu, logvar, beta, free_bits=free_bits
                )

            if training:
                scaler.scale(loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()        # OneCycleLR steps per optimizer step

            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl_loss.item()
            n_batches   += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val, args):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_val":  best_val,
        "args":      vars(args),
    }, path)
    print(f"    ✓ Saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass   # scheduler config may differ on resume
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best_val", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the Dino β-VAE (v2)")
    parser.add_argument("--tensor_dir",     default="./DATA/tensors")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--latent_dim",     type=int,   default=128)
    parser.add_argument("--beta",           type=float, default=0.5,
                        help="Final KL weight after annealing. Lower β = better recon.")
    parser.add_argument("--kl_warmup",      type=int,   default=20,
                        help="Epochs to linearly anneal β from 0 → --beta.")
    parser.add_argument("--val_frac",       type=float, default=0.12)
    parser.add_argument("--save_every",     type=int,   default=10)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--grad_accum",     type=int,   default=2,
                        help="Gradient accumulation steps.")
    parser.add_argument("--patience",       type=int,   default=30,
                        help="Early stopping patience in epochs (counted after kl_warmup).")
    parser.add_argument("--free_bits",      type=float, default=0.0,
                        help="Free-bits per latent dim (nats). 0 = no floor (recommended with low beta).")
    parser.add_argument("--freq_mask",      type=int,   default=15)
    parser.add_argument("--time_mask",      type=int,   default=30)
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--reset_best",     action="store_true",
                        help="Reset best_val to ∞ when resuming. Use when loss function changes.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"AMP        : {device.type == 'cuda'}")
    print(f"SpecAugment: {_HAS_TORCHAUDIO}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SpectrogramDataset(
        args.tensor_dir, split="train", val_frac=args.val_frac, augment=True
    )
    val_ds = SpectrogramDataset(
        args.tensor_dir, split="val", val_frac=args.val_frac, augment=False
    )
    print(f"Classes ({len(train_ds.classes)}): {', '.join(train_ds.classes)}")
    print(f"Train  : {len(train_ds):,} samples")
    print(f"Val    : {len(val_ds):,}   samples")

    # ── Group distribution diagnostic ─────────────────────────────────────────
    print("\nGroup distribution (train tensors):")
    dist = train_ds.group_distribution()
    GROUP_ORDER = ["Bird", "Reptile", "Mass"]
    for grp in GROUP_ORDER:
        if grp not in dist:
            continue
        total = sum(dist[grp].values())
        bar   = "█" * min(40, total // 200)
        print(f"  {grp:<8} {total:>6}  {bar}")
        for cls, cnt in sorted(dist[grp].items(), key=lambda x: -x[1]):
            print(f"    {cls:<30} {cnt:>5}")
    print()

    eff_bs = args.batch_size * args.grad_accum
    print(f"Effective batch size: {eff_bs}")
    print("Sampler       : group-weighted (Bird=1/3, Reptile=1/3, Mass=1/3)\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_ds.make_sampler(),
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        # Group-weighted sampler on val too: prevents val_loss being
        # 90 % Bird and ensures best_val reflects all three anchor groups.
        sampler=val_ds.make_sampler(num_samples=len(val_ds)),
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DinoVAE(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}  |  β_target={args.beta}  |  kl_warmup={args.kl_warmup}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99)
    )

    # ── OneCycleLR (steps per optimizer step, i.e., per grad_accum block) ────
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=10,
        final_div_factor=500,
        anneal_strategy="cos",
    )

    # ── Mixed precision + SpecAugment ─────────────────────────────────────────
    scaler     = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    augment_fn = SpecAugment(args.freq_mask, args.time_mask).to(device)

    start_epoch      = 1
    best_val         = float("inf")
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming from {args.resume}")
        start_epoch, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        if args.reset_best:
            best_val = float("inf")
            print(f"  best_val RESET to ∞  (loss function changed — old metric incomparable)")
        print(f"Starting at epoch {start_epoch},  best_val={best_val:.6f}\n")
    else:
        print()

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    if not os.path.exists(log_path) or start_epoch == 1:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "beta", "lr",
                "train_loss", "val_loss",
                "train_recon", "train_kl",
                "val_recon",   "val_kl",
            ])

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        beta_now = compute_beta(epoch, args.kl_warmup, args.beta)
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs}  β={beta_now:.3f}  lr={lr_now:.2e}")

        train_loss, train_recon, train_kl = run_epoch(
            model, train_loader, optimizer, scheduler, device,
            beta_now, scaler, augment_fn, training=True,
            grad_accum=args.grad_accum, free_bits=args.free_bits,
        )
        val_loss, val_recon, val_kl = run_epoch(
            model, val_loader, optimizer, scheduler, device,
            beta_now, scaler, None, training=False,
            grad_accum=1, free_bits=args.free_bits,
        )

        gap = val_recon - train_recon
        print(
            f"  train={train_loss:.5f} (r={train_recon:.5f} kl={train_kl:.5f})  "
            f"val={val_loss:.5f} (r={val_recon:.5f} kl={val_kl:.5f})  "
            f"recon_gap={gap:+.5f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{beta_now:.3f}", f"{lr_now:.6f}",
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{train_recon:.6f}", f"{train_kl:.6f}",
                f"{val_recon:.6f}", f"{val_kl:.6f}",
            ])

        # ── Best model (track val_recon — immune to β annealing) ──────────
        if val_recon < best_val - 1e-5:
            best_val = val_recon
            patience_counter = 0
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pt"),
                epoch, model, optimizer, scheduler, scaler, best_val, args,
            )
        else:
            # Don't penalise during KL warmup — val_recon isn't stable yet
            if epoch > args.kl_warmup:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
                    break
                elif patience_counter > 5:
                    print(f"  No recon improvement ({patience_counter}/{args.patience})")

        if epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt"),
                epoch, model, optimizer, scheduler, scaler, best_val, args,
            )

    print(f"\nDone.  Best val loss : {best_val:.6f}")
    print(f"Best checkpoint     : {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
