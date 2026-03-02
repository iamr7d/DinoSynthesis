"""
train_vae_gan.py  —  VAE-GAN training for DinoSynthesis
========================================================
Strategy
─────────
  Phase 1  (epochs 1 → --vae_warmup):
    • Load best.pt from standard VAE training
    • Freeze encoder  →  decoder fine-tunes via recon loss only
    • λ_adv = 0 (no adversarial yet — lets decoder adapt before seeing D)

  Phase 2  (epochs --vae_warmup+1 → --adv_warmup):
    • Adversarial loss ramps from 0 → λ_adv_max
    • Encoder stays frozen  →  latent geometry intact
    • Feature-matching loss stabilises training

  Phase 3  (epochs --adv_warmup+1 → --epochs):
    • Encoder unfrozen, full end-to-end GAN fine-tuning
    • Lower learning rate to avoid destroying the latent space

The three-phase strategy gives you:
  ✓ Sharp harmonics  (GAN discriminator)
  ✓ Perceptual quality  (feature-matching)
  ✓ Preserved latent geometry  (frozen encoder in early phases)
  ✓ Spectral convergence  (missing-frequency penalty)

Outputs
───────
  checkpoints_gan/best_gen.pt      — best generator (tracks val_recon)
  checkpoints_gan/best_gen_full.pt — full checkpoint with D state
  checkpoints_gan/loss_log_gan.csv — per-epoch loss breakdown

Usage
─────
  python train_vae_gan.py --vae_checkpoint checkpoints/best.pt
  python train_vae_gan.py --vae_checkpoint checkpoints/best.pt \\
      --epochs 100 --vae_warmup 10 --adv_warmup 30 \\
      --lambda_adv 0.15 --lambda_fm 10.0
"""

import os
import sys
import glob
import argparse
import csv
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import torchaudio.transforms as AT
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

from tqdm import tqdm

# Reuse dataset + augment from train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import SpectrogramDataset, SpecAugment, CLASS_GROUPS, compute_beta

from model.dino_vae_gan import (
    DinoVAE_GAN,
    MultiScaleDiscriminator,
    hinge_d_loss,
    vae_gan_g_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (one epoch)
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_gan(
    G, D,
    loader,
    opt_g, opt_d,
    sched_g, sched_d,
    scaler_g, scaler_d,
    device,
    beta:       float,
    lambda_adv: float,
    lambda_fm:  float,
    lambda_sc:  float,
    free_bits:  float,
    augment_fn,
    training:   bool,
    grad_accum: int = 1,
):
    G.train(training); D.train(training)
    sums = dict(d=0., g_total=0., recon=0., kl=0., sc=0., adv=0., fm=0.)
    n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, (x, _) in enumerate(
            tqdm(loader, desc="  train" if training else "  val  ", leave=False, unit="b")
        ):
            x = x.to(device, non_blocking=True)
            if training and augment_fn is not None:
                x = augment_fn(x)

            # ── 1. Discriminator step ─────────────────────────────────────
            if training and lambda_adv > 0:
                with torch.amp.autocast("cuda"):
                    recon, mu, logvar = G(x)
                    real_res = D(x)
                    fake_res_d = D(recon.detach())   # detach: don't update G here
                    d_loss = hinge_d_loss(real_res, fake_res_d)

                scaler_d.scale(d_loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    opt_d.zero_grad(set_to_none=True)
                    sched_d.step()
                sums["d"] += d_loss.item()

            # ── 2. Generator step ─────────────────────────────────────────
            with torch.amp.autocast("cuda"):
                recon, mu, logvar = G(x)
                if lambda_adv > 0 and training:
                    real_res = D(x)
                    fake_res_g = D(recon)
                else:
                    real_res = []; fake_res_g = []

                g_loss, bd = vae_gan_g_loss(
                    recon=recon, x=x, mu=mu, logvar=logvar,
                    real_results=real_res if real_res else [],
                    fake_results=fake_res_g if fake_res_g else [],
                    beta=beta, free_bits=free_bits,
                    lambda_adv=lambda_adv,
                    lambda_fm=lambda_fm if lambda_adv > 0 else 0.0,
                    lambda_sc=lambda_sc,
                )

            if training:
                scaler_g.scale(g_loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                    scaler_g.step(opt_g)
                    scaler_g.update()
                    opt_g.zero_grad(set_to_none=True)
                    sched_g.step()

            for k, v in bd.items():
                if k in sums: sums[k] += v
            n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_full(path, epoch, G, D, opt_g, opt_d, sched_g, sched_d,
              scaler_g, scaler_d, best_val, args):
    torch.save({
        "epoch": epoch, "best_val": best_val, "args": vars(args),
        "G": G.state_dict(), "D": D.state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
        "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(),
    }, path)
    print(f"    ✓ Saved full GAN checkpoint → {path}")


def save_gen_only(path, epoch, G, best_val, args):
    """Save a checkpoint compatible with app.py (same format as train.py)."""
    # app.py loads `ckpt["model"]` and expects DinoVAE keys
    sd = {k.removeprefix("vae."): v for k, v in G.state_dict().items()}
    torch.save({"epoch": epoch, "model": sd, "best_val": best_val, "args": vars(args)}, path)
    print(f"    ✓ Saved generator-only checkpoint → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    P = argparse.ArgumentParser(description="VAE-GAN fine-tuning for DinoSynthesis")
    P.add_argument("--vae_checkpoint",  default="checkpoints/best.pt",
                   help="Path to pre-trained DinoVAE checkpoint (best.pt)")
    P.add_argument("--tensor_dir",      default="./DATA/tensors")
    P.add_argument("--checkpoint_dir",  default="./checkpoints_gan")
    P.add_argument("--epochs",          type=int,   default=80)

    # Phase boundaries
    P.add_argument("--vae_warmup",      type=int,   default=10,
                   help="Epochs of decoder-only warm-up before adversarial starts")
    P.add_argument("--adv_warmup",      type=int,   default=30,
                   help="Epoch at which encoder is unfrozen for end-to-end tuning")

    # Loss weights
    P.add_argument("--beta",            type=float, default=0.5,
                   help="KL weight (lower than VAE-only since adversarial adds regularisation)")
    P.add_argument("--lambda_adv",      type=float, default=0.15)
    P.add_argument("--lambda_fm",       type=float, default=10.0)
    P.add_argument("--lambda_sc",       type=float, default=1.0)
    P.add_argument("--free_bits",       type=float, default=0.5)

    # Optimiser
    P.add_argument("--lr_g",            type=float, default=5e-5,
                   help="Generator LR (lower than from-scratch to preserve latent space)")
    P.add_argument("--lr_d",            type=float, default=2e-4)
    P.add_argument("--batch_size",      type=int,   default=48)
    P.add_argument("--grad_accum",      type=int,   default=2)
    P.add_argument("--latent_dim",      type=int,   default=128)
    P.add_argument("--workers",         type=int,   default=4)
    P.add_argument("--val_frac",        type=float, default=0.12)
    P.add_argument("--save_every",      type=int,   default=5)
    P.add_argument("--patience",        type=int,   default=20)
    P.add_argument("--freq_mask",       type=int,   default=15)
    P.add_argument("--time_mask",       type=int,   default=30)
    P.add_argument("--resume",          default=None,
                   help="Resume from checkpoints_gan/best_gen_full.pt")
    args = P.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"VAE ckpt   : {args.vae_checkpoint}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SpectrogramDataset(args.tensor_dir, split="train",
                                  val_frac=args.val_frac, augment=True)
    val_ds   = SpectrogramDataset(args.tensor_dir, split="val",
                                  val_frac=args.val_frac, augment=False)
    print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_ds.make_sampler(),
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              sampler=val_ds.make_sampler(len(val_ds)),
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True)

    # ── Models ────────────────────────────────────────────────────────────────
    G = DinoVAE_GAN(latent_dim=args.latent_dim).to(device)
    D = MultiScaleDiscriminator(n_scales=3).to(device)

    # Load pre-trained VAE weights
    if os.path.isfile(args.vae_checkpoint):
        G.load_vae_checkpoint(args.vae_checkpoint, device=device)

    G.freeze_encoder()   # Phase 1: decoder-only warm-up

    ng = sum(p.numel() for p in G.parameters() if p.requires_grad)
    nd = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"G trainable : {ng:,}  |  D params : {nd:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_g = torch.optim.AdamW(filter(lambda p: p.requires_grad, G.parameters()),
                               lr=args.lr_g, weight_decay=1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(D.parameters(),
                               lr=args.lr_d, weight_decay=1e-4, betas=(0.5, 0.9))

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))

    def make_scheduler(opt, lr):
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1, div_factor=5, final_div_factor=200,
            anneal_strategy="cos",
        )

    sched_g   = make_scheduler(opt_g, args.lr_g)
    sched_d   = make_scheduler(opt_d, args.lr_d)
    scaler_g  = torch.amp.GradScaler("cuda")
    scaler_d  = torch.amp.GradScaler("cuda")
    augment   = SpecAugment(args.freq_mask, args.time_mask).to(device)

    start_epoch      = 1
    best_val         = float("inf")
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"]); D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        try:
            sched_g.load_state_dict(ckpt["sched_g"])
            sched_d.load_state_dict(ckpt["sched_d"])
        except Exception: pass
        scaler_g.load_state_dict(ckpt["scaler_g"])
        scaler_d.load_state_dict(ckpt["scaler_d"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"  Resumed at epoch {start_epoch},  best_val={best_val:.6f}")

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.checkpoint_dir, "loss_log_gan.csv")
    if not os.path.exists(log_path) or start_epoch == 1:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch","phase","beta","lambda_adv",
                "train_d","train_g","train_recon","train_kl","train_sc","train_adv","train_fm",
                "val_recon",
            ])

    phase_names = {1: "decoder_warmup", 2: "adv_ramp", 3: "full_finetune"}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # Determine phase
        if epoch <= args.vae_warmup:
            phase     = 1
            lam_adv   = 0.0
        elif epoch <= args.adv_warmup:
            phase     = 2
            # Ramp λ_adv linearly from 0 → target
            prog      = (epoch - args.vae_warmup) / max(1, args.adv_warmup - args.vae_warmup)
            lam_adv   = args.lambda_adv * prog
        else:
            phase     = 3
            lam_adv   = args.lambda_adv

        # Phase transitions
        if epoch == args.vae_warmup + 1:
            print(f"\n[Phase 2] Adversarial loss ramping on (lambda_adv → {args.lambda_adv})")

        if epoch == args.adv_warmup + 1:
            print(f"\n[Phase 3] Encoder unfrozen — end-to-end fine-tuning")
            G.unfreeze_encoder()
            # Rebuild opt_g with all parameters at a lower LR
            opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g * 0.3,
                                       weight_decay=1e-4, betas=(0.5, 0.9))
            sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_g, T_max=(args.epochs - args.adv_warmup), eta_min=1e-6
            )

        beta_now = compute_beta(epoch, max(1, args.vae_warmup), args.beta)
        print(f"Epoch {epoch:03d}/{args.epochs}  phase={phase_names[phase]}  "
              f"β={beta_now:.3f}  λ_adv={lam_adv:.4f}  "
              f"lr_g={opt_g.param_groups[0]['lr']:.2e}")

        train_m = run_epoch_gan(
            G, D, train_loader, opt_g, opt_d, sched_g, sched_d,
            scaler_g, scaler_d, device,
            beta=beta_now, lambda_adv=lam_adv,
            lambda_fm=args.lambda_fm, lambda_sc=args.lambda_sc,
            free_bits=args.free_bits,
            augment_fn=augment, training=True,
            grad_accum=args.grad_accum,
        )

        val_m = run_epoch_gan(
            G, D, val_loader, opt_g, opt_d, sched_g, sched_d,
            scaler_g, scaler_d, device,
            beta=beta_now, lambda_adv=0.0,  # no adversarial on val
            lambda_fm=0.0, lambda_sc=args.lambda_sc,
            free_bits=args.free_bits,
            augment_fn=None, training=False,
            grad_accum=1,
        )

        print(
            f"  D={train_m['d']:.4f}  "
            f"G={train_m['g_total']:.4f}  "
            f"recon={train_m['recon']:.5f}  "
            f"kl={train_m['kl']:.4f}  "
            f"sc={train_m['sc']:.4f}  "
            f"adv={train_m['adv']:.4f}  "
            f"fm={train_m['fm']:.4f}  "
            f"| val_recon={val_m['recon']:.5f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, phase_names[phase], f"{beta_now:.3f}", f"{lam_adv:.4f}",
                f"{train_m['d']:.5f}", f"{train_m['g_total']:.5f}",
                f"{train_m['recon']:.5f}", f"{train_m['kl']:.5f}",
                f"{train_m['sc']:.5f}", f"{train_m['adv']:.5f}", f"{train_m['fm']:.5f}",
                f"{val_m['recon']:.5f}",
            ])

        # Best model (track val_recon — compatible with app.py)
        val_recon = val_m["recon"]
        if val_recon < best_val - 1e-5:
            best_val = val_recon
            patience_counter = 0
            save_gen_only(
                os.path.join(args.checkpoint_dir, "best_gen.pt"),
                epoch, G, best_val, args,
            )
            save_full(
                os.path.join(args.checkpoint_dir, "best_gen_full.pt"),
                epoch, G, D, opt_g, opt_d, sched_g, sched_d,
                scaler_g, scaler_d, best_val, args,
            )
        else:
            if epoch > args.vae_warmup:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break
                elif patience_counter > 5:
                    print(f"  No recon improvement ({patience_counter}/{args.patience})")

        if epoch % args.save_every == 0:
            save_full(
                os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}_full.pt"),
                epoch, G, D, opt_g, opt_d, sched_g, sched_d,
                scaler_g, scaler_d, best_val, args,
            )

    print(f"\nDone.  Best val_recon : {best_val:.6f}")
    print(f"Load into app.py with: checkpoints_gan/best_gen.pt")


if __name__ == "__main__":
    main()
