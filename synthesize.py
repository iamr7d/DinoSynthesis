"""
synthesize.py — DinoSynthesis: Phylogenetic Interpolation
==========================================================
Reconstructs a hypothetical non-avian dinosaur vocalization by
interpolating between:
  - Tinamou_Tinamus  (Bird anchor — Neornithes, crown Archosaur)
  - Crocodylia        (Crocodile anchor — sister group to birds)
  - Whippomorpha      (Mass/size proxy — hippopotamus + cetacean)

Interpolation formula:
  z_mid  = 0.5 * z_bird + 0.5 * z_croc   (phylogenetic midpoint)
  z_dino = 0.8 * z_mid  + 0.2 * z_mass   (mass/body-size influence)

Outputs (saved to output_synthesis/):
  - dino_100pct_bird.wav
  - dino_100pct_croc.wav
  - dino_synthesis.wav
  - dino_spectrogram_comparison.png
"""

import os
import sys
import glob
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

# ── Shared audio utilities ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import save_wav, save_comparison_figure, SR, DB_RANGE

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
TENSOR_DIR = os.path.join(ROOT, "DATA", "tensors")
CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best.pt")
OUT_DIR    = os.path.join(ROOT, "output_synthesis")
os.makedirs(OUT_DIR, exist_ok=True)

from model.dino_vae import DinoVAE

# ── Audio parameters ──────────────────────────────────────────────────────────
N_FFT      = 1024
HOP_LENGTH = 256
N_MELS     = 128

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[synthesize] Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    model = DinoVAE().to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)          # handle both formats
    model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("best_val", ckpt.get("val_recon", "?"))
    print(f"[synthesize] Loaded checkpoint — epoch {epoch}, best_val={val:.6f}" if isinstance(val, float) else f"[synthesize] Loaded checkpoint — epoch {epoch}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute class centroid in latent space
# ─────────────────────────────────────────────────────────────────────────────
def class_centroid(model, class_name, max_samples=200):
    """
    Load up to `max_samples` tensors for `class_name`, encode each with the
    deterministic mean (mu), and return the average latent vector.
    """
    tensor_glob = os.path.join(TENSOR_DIR, class_name, "*.pt")
    paths = sorted(glob.glob(tensor_glob))
    if not paths:
        raise FileNotFoundError(f"No tensors found for class '{class_name}' at {tensor_glob}")
    paths = paths[:max_samples]

    mus = []
    with torch.no_grad():
        for p in paths:
            x = torch.load(p, map_location=device, weights_only=True)  # (1,128,259) or (1,128,256)
            # Ensure shape (1,1,128,256)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x[:, :, :, :256]       # crop to model input width
            mu, _ = model.encode(x)    # encode returns (mu, logvar)
            mus.append(mu.squeeze(0))  # (latent_dim,)

    centroid = torch.stack(mus).mean(dim=0)
    print(f"  [{class_name:25s}] {len(mus):4d} samples  |  μ norm={centroid.norm().item():.4f}")
    return centroid


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decode latent vector → normalised spectrogram numpy array
# ─────────────────────────────────────────────────────────────────────────────
def decode_to_spec(model, z):
    """z: (latent_dim,) tensor → (128, 256) numpy array in [0,1]"""
    with torch.no_grad():
        recon = model.decode(z.unsqueeze(0))   # (1,1,128,256)
    spec = recon.squeeze().cpu().numpy()       # (128,256), values in [0,1] via Sigmoid
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Spectrogram → WAV and comparison figure are provided by audio_utils
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharpness", type=float, default=1.5,
                        help="Spectrogram sharpening power (0=off, 1.5=mild, 2.5=strong)")
    parser.add_argument("--preemphasis", type=float, default=0.97,
                        help="Pre-emphasis coefficient (0=off, 0.97=standard)")
    parser.add_argument("--gl_iter", type=int, default=128,
                        help="Griffin-Lim iterations")
    parser.add_argument("--gate", type=float, default=0.0,
                        help="Spectral gate threshold (0=off, 0.10–0.20 typical)")
    parser.add_argument("--sobel", type=float, default=0.0,
                        help="Sobel vertical sharpening strength (0=off, 0.3–0.8 typical)")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Latent jitter std-dev for stochastic sampling (0=off, 0.05 typical)")
    parser.add_argument("--n_jitter", type=int, default=6,
                        help="Number of jittered samples to average (only used when --jitter>0)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  DinoSynthesis — Phylogenetic Interpolation")
    print(f"  sharpness={args.sharpness}  preemph={args.preemphasis}  gl_iter={args.gl_iter}")
    print(f"  gate={args.gate}  sobel={args.sobel}  jitter={args.jitter}  n_jitter={args.n_jitter}")
    print("="*65 + "\n")

    # 1. Load model ─────────────────────────────────────────────────
    model = load_model(CKPT_PATH)

    # 2. Compute centroids ─────────────────────────────────────────
    print("[synthesize] Computing class centroids …")
    z_bird = class_centroid(model, "Tinamou_Tinamus")
    z_croc = class_centroid(model, "Crocodylia")
    z_mass = class_centroid(model, "Whippomorpha")

    # 3. Interpolate ───────────────────────────────────────────────
    print("\n[synthesize] Building interpolated latent vector …")
    z_mid  = 0.5 * z_bird + 0.5 * z_croc
    z_dino = 0.8 * z_mid  + 0.2 * z_mass

    print(f"  z_bird  norm = {z_bird.norm().item():.4f}")
    print(f"  z_croc  norm = {z_croc.norm().item():.4f}")
    print(f"  z_mass  norm = {z_mass.norm().item():.4f}")
    print(f"  z_dino  norm = {z_dino.norm().item():.4f}")

    # 4. Decode all three vectors (─── with optional jitter) ─────────────────
    print("\n[synthesize] Decoding latent vectors …")

    def jitter_decode(z):
        """Decode z, averaging --n_jitter stochastic samples if --jitter>0."""
        if args.jitter > 0:
            specs = []
            for _ in range(args.n_jitter):
                z_j = z + args.jitter * torch.randn_like(z)
                specs.append(decode_to_spec(model, z_j))
            return np.mean(specs, axis=0)
        return decode_to_spec(model, z)

    spec_bird = jitter_decode(z_bird)
    spec_croc = jitter_decode(z_croc)
    spec_dino = jitter_decode(z_dino)

    # 5. Save WAVs (hifi Griffin-Lim with sharpening + pre-emphasis) ──────────
    print(f"\n[synthesize] Synthesising audio "
          f"(GL×{args.gl_iter}, sharp={args.sharpness}, preemph={args.preemphasis}"
          f", gate={args.gate}, sobel={args.sobel}) …")
    kw = dict(n_iter=args.gl_iter, sharpness=args.sharpness,
              preemphasis_coef=args.preemphasis, gate=args.gate,
              sobel_strength=args.sobel)
    save_wav(spec_bird, os.path.join(OUT_DIR, "dino_100pct_bird.wav"), **kw)
    save_wav(spec_croc, os.path.join(OUT_DIR, "dino_100pct_croc.wav"), **kw)
    save_wav(spec_dino, os.path.join(OUT_DIR, "dino_synthesis.wav"),   **kw)

    # 6. Save comparison figure ────────────────────────────────────
    print("\n[synthesize] Rendering spectrogram comparison …")
    from collections import OrderedDict
    specs = OrderedDict([
        ("Bird Anchor\n(Tinamou tinamus)",                           spec_bird),
        ("Dinosaur Synthesis\n(0.5 Bird + 0.5 Croc + 0.2 Mass)",    spec_dino),
        ("Croc Anchor\n(Crocodylia)",                                spec_croc),
    ])
    save_comparison_figure(
        specs,
        os.path.join(OUT_DIR, "dino_spectrogram_comparison.png"),
        sharpness=args.sharpness, gate=args.gate, sobel_strength=args.sobel,
    )

    print("\n" + "="*65)
    print("  SYNTHESIS COMPLETE")
    print(f"  Output directory: {OUT_DIR}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
