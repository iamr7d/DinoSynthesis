"""
synthesize_ensemble.py — Brownian Bridge Ensemble Synthesis (Phase 2.0)
========================================================================
Implements Stochastic Ancestral State Estimation (SASE):

    z(t) = (1 - t)·z_start + t·z_end + σ(T)·√(t(1-t))·ε

where:
  z_start = bird centroid
  z_end   = croc centroid
  t       = 0.5 (evolutionary midpoint)
  σ(T)    = branch-length-scaled noise:  σ = base_sigma·√(T_dino / T_max)
  ε       ~ N(0, I)  — stochastic path samples

Rather than one deterministic "Dino.wav", this script generates an ensemble
of N samples representing the *probability cloud* of potential dinosaur calls
at the t=0.5 phylogenetic midpoint.

Outputs (saved to output_synthesis/ensemble/):
  - dino_ensemble_000.wav  …  dino_ensemble_019.wav    (N=20 samples)
  - dino_ensemble_mean.wav                              (posterior mean)
  - dino_ensemble_spectrograms.png                     (5×4 grid overview)
  - dino_ensemble_stats.json                           (spectral statistics)

Usage:
    python synthesize_ensemble.py [--n_samples 20] [--sigma 0.6]
                                  [--branch_mya 166] [--gl_iter 128]
                                  [--sharpness 1.5] [--no_raup]

"The fossil record tells us what was; the latent space tells us what could
 have been. The Raup Constraint ensures we don't imagine the impossible."
  — DinoSynthesis Strategic Evolution Plan (Phase 2.0)
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import math

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import save_wav, SR, N_FFT, HOP_LENGTH, N_MELS, DB_RANGE
from model.dino_vae import DinoVAE

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
TENSOR_DIR  = os.path.join(ROOT, "DATA", "tensors")
CKPT_PATH   = os.path.join(ROOT, "checkpoints", "best.pt")
OUT_DIR     = os.path.join(ROOT, "output_synthesis", "ensemble")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ensemble] Using device: {device}")

# ── Phylogenetic branch constants ─────────────────────────────────────────────
_T_MAX_MYA   = 320.0   # Max patristic distance used for σ normalisation
_DINO_MYA    = 166.0   # Estimated branch length to non-avian dinosaur ancestor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> DinoVAE:
    model = DinoVAE().to(device)
    model.eval()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    ep = ckpt.get("epoch", "?")
    bv = ckpt.get("best_val", float("nan"))
    print(f"[ensemble] Loaded checkpoint — epoch {ep}, best_val={bv:.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class centroid computation
# ─────────────────────────────────────────────────────────────────────────────

def class_centroid(
    model:      DinoVAE,
    class_dir:  str,
    max_samples: int = 200,
) -> torch.Tensor:
    """Encode up to max_samples tensors and return the mean latent vector."""
    files = sorted(glob.glob(os.path.join(class_dir, "*.pt")))[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .pt tensors found in {class_dir}")

    mus = []
    with torch.no_grad():
        for f in files:
            x   = torch.load(f, map_location=device, weights_only=True)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            mu, _ = model.encode(x.to(device))
            mus.append(mu.squeeze(0))
    centroid = torch.stack(mus).mean(0)
    print(f"  [{os.path.basename(class_dir)}] {len(files)} samples | μ norm={centroid.norm():.4f}")
    return centroid


# ─────────────────────────────────────────────────────────────────────────────
# 3. Brownian Bridge latent sampling
# ─────────────────────────────────────────────────────────────────────────────

def brownian_bridge_samples(
    z_start:     torch.Tensor,   # (D,) — e.g. bird centroid
    z_end:       torch.Tensor,   # (D,) — e.g. croc centroid
    t:           float  = 0.5,   # Evolutionary time [0, 1]
    sigma:       float  = 0.5,   # Base noise amplitude
    branch_mya:  float  = _DINO_MYA,
    n_samples:   int    = 20,
    generator:   torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample N latent vectors from the Brownian Bridge distribution at time t.

    The Brownian Bridge from z_start to z_end:

        z(t) = (1-t)·z_start + t·z_end + σ_eff·√(t(1-t))·ε

    where σ_eff is scaled by the branch length in millions of years:

        σ_eff = sigma · √(branch_mya / T_max)

    Longer branches (deeper evolutionary time) → more uncertainty → more spread.

    Args:
        z_start    : Start of evolutionary path (e.g. bird latent centroid).
        z_end      : End of evolutionary path (e.g. croc latent centroid).
        t          : Position along the bridge; 0.5 = midpoint (ancestor).
        sigma      : Base noise amplitude.
        branch_mya : Branch length to the reconstructed ancestor (Mya).
        n_samples  : Number of stochastic samples to draw.
        generator  : Optional torch.Generator for reproducibility.

    Returns:
        Tensor of shape (N, D) — N latent vectors sampled at time t.
    """
    # Deterministic midpoint (linear interpolation component)
    z_mid    = (1.0 - t) * z_start + t * z_end  # (D,)

    # Brownian Bridge variance at time t:  Var = σ_eff² · t(1-t)
    sigma_eff = sigma * math.sqrt(branch_mya / _T_MAX_MYA)
    std_t     = sigma_eff * math.sqrt(t * (1.0 - t))

    D        = z_mid.shape[0]
    noise    = torch.randn(n_samples, D, device=z_mid.device, generator=generator)
    samples  = z_mid.unsqueeze(0) + std_t * noise  # (N, D)

    print(f"[ensemble] σ_eff={sigma_eff:.4f}  std(t=0.5)={std_t:.4f}  "
          f"||z_mid||={z_mid.norm():.4f}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spectrogram → Audio (matches synthesize.py pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def latent_to_wav(
    model:       DinoVAE,
    z:           torch.Tensor,   # (D,) — single latent vector
    gl_iter:     int   = 128,
    sharpness:   float = 1.5,
    preemph:     float = 0.97,
    gate:        float = 0.0,
    dino_mass_kg: float = 5000.0,  # 5-tonne non-avian dinosaur
    apply_raup:  bool  = True,
) -> np.ndarray:
    """Decode a latent vector to a numpy audio waveform."""
    import librosa

    with torch.no_grad():
        recon = model.decode(z.unsqueeze(0))  # (1, 1, 128, 256)

    spec = recon.squeeze().cpu().numpy()  # (128, 256) ∈ [0, 1]

    # Spectral gate (noise floor removal)
    if gate > 0.0:
        spec[spec < gate] = 0.0

    # Power-law sharpening
    if sharpness != 1.0:
        spec = np.clip(spec ** sharpness, 0, 1)

    # ── Raup Constraint: allometric low-pass filter ──────────────────────
    if apply_raup:
        # F_c ∝ M^(-1/3); normalise so that 500 kg → full band, 5000 kg → ~46%
        m_ref   = 500.0   # kg reference mass
        fc_frac = (m_ref / dino_mass_kg) ** (1.0 / 3.0)
        fc_frac = float(np.clip(fc_frac, 0.05, 1.0))
        cutoff_bin = int(fc_frac * spec.shape[0])
        # Smooth sigmoid ramp (avoids hard cutoff artefacts)
        bins     = np.linspace(0, 1, spec.shape[0])
        ramp     = 1.0 / (1.0 + np.exp(12.0 * (bins - fc_frac)))
        spec     = spec * ramp[:, None]

    # Denormalise: [0,1] → dB → power
    spec_db  = spec * DB_RANGE - DB_RANGE          # [-80, 0] dB
    spec_pow = librosa.db_to_power(spec_db)

    # Griffin-Lim inversion
    audio = librosa.griffinlim(
        spec_pow,
        n_iter=gl_iter,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )

    # Pre-emphasis
    if preemph > 0.0:
        audio = librosa.effects.preemphasis(audio, coef=preemph)

    # Normalise peak
    peak = np.abs(audio).max()
    if peak > 1e-7:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spectrogram statistics
# ─────────────────────────────────────────────────────────────────────────────

def spectral_centroid_hz(spec: np.ndarray) -> float:
    """Estimate spectral centroid from a normalised mel spectrogram."""
    import librosa
    mel_freqs = librosa.mel_frequencies(n_mels=spec.shape[0], fmin=0, fmax=SR // 2)
    power     = (spec ** 2).mean(axis=1)  # energy per mel bin
    total     = power.sum() + 1e-9
    return float((mel_freqs * power).sum() / total)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation: 5×4 spectrogram grid
# ─────────────────────────────────────────────────────────────────────────────

def save_ensemble_figure(
    specs:     list[np.ndarray],   # list of (128, 256) arrays
    out_path:  str,
    n_cols:    int = 5,
) -> None:
    n_samples = len(specs)
    n_rows    = math.ceil(n_samples / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 2.5),
                             squeeze=False)
    fig.patch.set_facecolor("#0a0a14")

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_facecolor("#0a0a14")
        if idx < n_samples:
            ax.imshow(specs[idx], origin="lower", aspect="auto",
                      cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"Sample {idx:02d}", color="#aabbcc", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "Brownian Bridge Ensemble — Dinosaur Vocalization Probability Cloud\n"
        "t = 0.5 (Phylogenetic Midpoint)",
        color="#e0e8ff", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ensemble] Saved grid PNG → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    model = load_model(CKPT_PATH)

    # ── Compute class centroids ───────────────────────────────────────────
    print("[ensemble] Computing centroids …")
    class_dirs = {
        "bird": os.path.join(TENSOR_DIR, "Tinamou_Tinamus"),
        "croc": os.path.join(TENSOR_DIR, "Crocodylia"),
    }
    z_bird = class_centroid(model, class_dirs["bird"], args.max_samples)
    z_croc = class_centroid(model, class_dirs["croc"], args.max_samples)

    # ── Brownian Bridge sampling ──────────────────────────────────────────
    print(f"\n[ensemble] Sampling Brownian Bridge (N={args.n_samples}, "
          f"sigma={args.sigma}, branch={args.branch_mya} Mya) …")
    gen     = torch.Generator(device=device).manual_seed(args.seed)
    samples = brownian_bridge_samples(
        z_start    = z_bird,
        z_end      = z_croc,
        t          = 0.5,
        sigma      = args.sigma,
        branch_mya = args.branch_mya,
        n_samples  = args.n_samples,
        generator  = gen,
    )  # (N, D)

    # ── Synthesise each sample ────────────────────────────────────────────
    print(f"\n[ensemble] Synthesising {args.n_samples} vocalisations …")
    wavs     = []
    specs    = []
    centroids_hz = []

    for i in range(args.n_samples):
        wav = latent_to_wav(
            model,
            samples[i],
            gl_iter=args.gl_iter,
            sharpness=args.sharpness,
            apply_raup=not args.no_raup,
            dino_mass_kg=args.dino_mass_kg,
        )
        with torch.no_grad():
            recon = model.decode(samples[i].unsqueeze(0))
        spec = recon.squeeze().cpu().numpy()

        wavs.append(wav)
        specs.append(spec)
        centroids_hz.append(spectral_centroid_hz(spec))

        out_wav = os.path.join(OUT_DIR, f"dino_ensemble_{i:03d}.wav")
        save_wav(wav, out_wav)
        print(f"  [{i:02d}] centroid={centroids_hz[-1]:.0f} Hz → {out_wav}")

    # ── Posterior mean synthesis ──────────────────────────────────────────
    z_mean   = samples.mean(0)
    wav_mean = latent_to_wav(
        model, z_mean,
        gl_iter=args.gl_iter, sharpness=args.sharpness,
        apply_raup=not args.no_raup, dino_mass_kg=args.dino_mass_kg,
    )
    mean_path = os.path.join(OUT_DIR, "dino_ensemble_mean.wav")
    save_wav(wav_mean, mean_path)
    print(f"\n[ensemble] Posterior mean WAV → {mean_path}")

    # ── Save spectrogram grid ─────────────────────────────────────────────
    grid_path = os.path.join(OUT_DIR, "dino_ensemble_spectrograms.png")
    save_ensemble_figure(specs, grid_path)

    # ── Statistics JSON ───────────────────────────────────────────────────
    stats = {
        "n_samples":          args.n_samples,
        "sigma":              args.sigma,
        "branch_mya":         args.branch_mya,
        "dino_mass_kg":       args.dino_mass_kg,
        "raup_filter":        not args.no_raup,
        "t":                  0.5,
        "centroid_hz_mean":   float(np.mean(centroids_hz)),
        "centroid_hz_std":    float(np.std(centroids_hz)),
        "centroid_hz_min":    float(np.min(centroids_hz)),
        "centroid_hz_max":    float(np.max(centroids_hz)),
        "centroid_hz_all":    centroids_hz,
        "latent_norm_mean":   float(samples.norm(dim=1).mean().item()),
        "latent_norm_std":    float(samples.norm(dim=1).std().item()),
    }

    stats_path = os.path.join(OUT_DIR, "dino_ensemble_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[ensemble] Stats JSON → {stats_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           BROWNIAN BRIDGE ENSEMBLE — COMPLETE                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Samples:          {args.n_samples:>4d}                                         ║
║  Spectral centroid {np.mean(centroids_hz):>8.1f} ± {np.std(centroids_hz):.1f} Hz                      ║
║  Latent norm       {float(samples.norm(dim=1).mean()):>8.4f} ± {float(samples.norm(dim=1).std()):.4f}                   ║
║  Raup LPF applied: {str(not args.no_raup):>5}                                    ║
║  Output dir:  {OUT_DIR:<45} ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brownian Bridge Ensemble Synthesis — DinoSynthesis Phase 2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples",    type=int,   default=20,
                        help="Number of stochastic ensemble samples")
    parser.add_argument("--sigma",        type=float, default=0.6,
                        help="Base Brownian Bridge noise amplitude σ")
    parser.add_argument("--branch_mya",   type=float, default=_DINO_MYA,
                        help="Branch length to reconstructed ancestor (Mya)")
    parser.add_argument("--dino_mass_kg", type=float, default=5000.0,
                        help="Estimated dinosaur mass (kg) for Raup LPF")
    parser.add_argument("--gl_iter",      type=int,   default=128,
                        help="Griffin-Lim iterations")
    parser.add_argument("--sharpness",    type=float, default=1.5,
                        help="Spectrogram sharpening power (>1 boosts formants)")
    parser.add_argument("--max_samples",  type=int,   default=200,
                        help="Max encoder samples per class for centroid")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_raup",      action="store_true",
                        help="Skip allometric low-pass filter (Raup Constraint)")
    args = parser.parse_args()
    main(args)
