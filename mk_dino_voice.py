"""
mk_dino_voice.py  —  DinoSynthesis High-Fidelity Voice Pipeline
================================================================
Three-stage pipeline targeting a biologically plausible dinosaur vocalization:

  Stage A — Spectrogram Shaping
      • Stochastic ensemble decode (N jittered samples → mean spec)
      • Low-end amplification:  boost mel bands below ~300 Hz
      • High-end suppression:   roll off mel bands above ~2 kHz
      • Temporal coherence smoothing  (removes frame-discontinuity beating)
      • Spectral gate             (kills the diffuse VAE noise floor)
      • Sobel-Y sharpening        (restores onset/harmonic vertical structure)

  Stage B — Audio Inversion
      • hifi_griffinlim()  with  preemphasis=0.0  (avoids reintroducing bird sibilance)
      • 256 Griffin-Lim iterations for convergence

  Stage C — Physiological Polish (large-body physics)
      • LPF at 700 Hz  — simulates a metre-long soft-tissue vocal tract
      • Sub-harmonic blend (−12 st, 40%)  — adds infrasonic body mass
      • Slow ADSR  (attack 250 ms, release 800 ms)  — biological breath dynamics
      • Slapback delay  (60 ms, 15%)  — open-environment spatial depth

Usage:
    python mk_dino_voice.py
    python mk_dino_voice.py --mass 0.3 --jitter 0.05 --n_jitter 8
    python mk_dino_voice.py --checkpoint checkpoints/best.pt --out_dir output_synthesis

Outputs to ./output_synthesis/:
    dino_voice_raw.wav         —  after Stage B only  (for comparison)
    dino_voice.wav             —  full pipeline output  ← the main result
    dino_voice_spectrogram.png —  three-panel figure (raw / shaped / polished spectra)
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.dino_vae import DinoVAE
from audio_utils import hifi_griffinlim, spectral_gate, SR, N_FFT, HOP_LENGTH, N_MELS, DB_RANGE
from audio_polish import organic_polish

TENSOR_DIR = os.path.join(ROOT, "DATA", "tensors")
OUT_DIR    = os.path.join(ROOT, "output_synthesis")
os.makedirs(OUT_DIR, exist_ok=True)

CENTROID_CLASSES = {
    "bird":  "Tinamou_Tinamus",
    "croc":  "Crocodylia",
    "mass":  "Whippomorpha",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utilities
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> DinoVAE:
    model = DinoVAE().to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("best_val", "?")
    print(f"[voice] Model loaded — epoch={epoch}  best_val={val}")
    return model


def class_centroid(model: DinoVAE, class_name: str, device: torch.device,
                   max_samples: int = 200) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(TENSOR_DIR, class_name, "*.pt")))[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No tensors for '{class_name}'")
    mus = []
    with torch.no_grad():
        for p in paths:
            x = torch.load(p, map_location=device, weights_only=True)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            T = x.shape[-1]
            if T >= 256:
                x = x[:, :, :, (T-256)//2 : (T-256)//2+256]
            else:
                x = torch.nn.functional.pad(x, (0, 256-T))
            mu, _ = model.encode(x)
            mus.append(mu.squeeze(0))
    c = torch.stack(mus).mean(0)
    print(f"[voice]   {class_name:<28} {len(mus):>4} samples  norm={c.norm():.4f}")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stage A — Spectrogram Shaping
# ─────────────────────────────────────────────────────────────────────────────
def _mel_freq_bins(n_mels: int = N_MELS, sr: int = SR) -> np.ndarray:
    """Return the centre frequency (Hz) for each of the n_mels mel bins."""
    return librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr/2)


def frequency_reshape(spec: np.ndarray,
                      boost_below_hz: float = 400.0,
                      boost_db: float = 8.0,
                      cut_above_hz: float  = 2000.0,
                      cut_db: float  = 12.0) -> np.ndarray:
    """
    Mel-band frequency emphasis curve.

    Boosts sub-400 Hz bins (infrasonic body resonance) and rolls off
    above 2 kHz (suppresses avian sibilance / decoder HF bias).
    Operates directly on the [0,1] normalised log-mel spectrogram
    by applying a gain curve in the dB domain and renormalising.
    """
    freqs   = _mel_freq_bins(spec.shape[0])   # (n_mels,)
    gain_db = np.zeros(spec.shape[0], dtype=np.float32)

    for i, f in enumerate(freqs):
        if f <= boost_below_hz:
            gain_db[i] = boost_db
        elif f <= cut_above_hz:
            # Linear crossfade from +boost_db → −cut_db
            t = (f - boost_below_hz) / (cut_above_hz - boost_below_hz)
            gain_db[i] = boost_db + t * (-cut_db - boost_db)
        else:
            gain_db[i] = -cut_db

    # Apply gain: spec is normalised log-mel; shift each row in dB space
    gain_lin = 10 ** (gain_db / 20.0)          # (n_mels,)
    out = spec * gain_lin[:, np.newaxis]         # broadcast over time axis
    # Clip and renormalise to [0, 1]
    out = np.clip(out, 0.0, None)
    vmax = out.max()
    if vmax > 1e-6:
        out /= vmax
    return out.astype(np.float32)


def ensemble_decode(model: DinoVAE, z: torch.Tensor, device: torch.device,
                    n_samples: int = 6, jitter_std: float = 0.04) -> np.ndarray:
    """
    Decode z N times with small Gaussian jitter, return the mean spectrogram.
    Averaging in spec space reduces random noise while preserving structure.
    """
    specs = []
    with torch.no_grad():
        for _ in range(n_samples):
            z_j  = z + jitter_std * torch.randn_like(z)
            recon = model.decode(z_j.unsqueeze(0))
            s     = recon.squeeze().cpu().numpy()
            s     = (s + 1.0) / 2.0   # [-1,1] → [0,1]
            specs.append(np.clip(s, 0.0, 1.0))
    return np.mean(specs, axis=0).astype(np.float32)


def shape_spectrogram(spec: np.ndarray,
                      boost_below_hz: float = 400.0,
                      boost_db: float = 8.0,
                      cut_above_hz: float = 2000.0,
                      cut_db: float = 12.0,
                      gate: float = 0.10,
                      temporal_smooth: float = 0.7,
                      sobel_strength: float = 0.35) -> np.ndarray:
    """Full Stage A pipeline."""
    import scipy.ndimage as ndi

    s = spec.copy()

    # 1. Temporal coherence smoothing — kills frame-discontinuity AM buzz
    if temporal_smooth > 0:
        s = ndi.gaussian_filter1d(s, sigma=temporal_smooth, axis=1)
        s = np.clip(s, 0.0, 1.0)

    # 2. Spectral gate — kill diffuse VAE noise floor
    s = spectral_gate(s, threshold=gate)

    # 3. Frequency reshape — amplify low end, suppress avian HF
    s = frequency_reshape(s,
                          boost_below_hz=boost_below_hz, boost_db=boost_db,
                          cut_above_hz=cut_above_hz,    cut_db=cut_db)

    # 4. Sobel-Y sharpening — restore onset/harmonic vertical definition
    if sobel_strength > 0:
        sobel_k  = np.array([-1, 0, 1], dtype=np.float32).reshape(3, 1)
        import scipy.ndimage as ndi2
        edge_map = np.abs(ndi2.convolve(s, sobel_k, mode='reflect'))
        e_max    = edge_map.max()
        if e_max > 1e-6:
            edge_map /= e_max
        s = np.clip(s + sobel_strength * edge_map, 0.0, 1.0)

    return s


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stage B — Audio Inversion
# ─────────────────────────────────────────────────────────────────────────────
def invert_to_audio(spec: np.ndarray, gl_iters: int = 256) -> np.ndarray:
    """
    hifi_griffinlim with pre-emphasis DISABLED.
    Pre-emphasis (coef=0.97) boosts ~6 kHz by +6 dB — exactly what we don't want
    for a large-animal vocalization.  Setting coef=0.0 skips the filter entirely.
    """
    return hifi_griffinlim(spec, n_iter=gl_iters, preemphasis_coef=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage C — Physiological Polish
# ─────────────────────────────────────────────────────────────────────────────
def dino_polish(audio: np.ndarray,
                lpf_hz: float = 700.0,
                sub_blend: float = 0.40,
                attack_ms: float = 250.0,
                release_ms: float = 800.0,
                delay_ms: float = 60.0,
                delay_blend: float = 0.15) -> np.ndarray:
    """
    Physiological Polish tuned for a large non-avian archosaur.

    Key differences from the generic organic_polish defaults:
      lpf_hz    700  (was 3500) — metre-long soft-tissue vocal tract
      sub_blend 0.40  (was 0.30) — more infrasonic body weight
      attack_ms 250   (was 150)  — slower breath onset
      release_ms 800  (was 300)  — long resonant tail-off
    """
    return organic_polish(
        audio, sr=SR,
        attack_ms=attack_ms,
        release_ms=release_ms,
        lpf_hz=lpf_hz,
        sub_blend=sub_blend,
        delay_ms=delay_ms,
        delay_blend=delay_blend,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def save_spectrogram_figure(raw_spec: np.ndarray,
                            shaped_spec: np.ndarray,
                            audio_raw: np.ndarray,
                            audio_polished: np.ndarray,
                            out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("DinoSynthesis Voice Pipeline", fontsize=14, fontweight="bold")

    kw = dict(aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")

    axes[0, 0].imshow(raw_spec,    vmin=0, vmax=1, **kw)
    axes[0, 0].set_title("A: Raw VAE Decode (pre-shaping)")
    axes[0, 0].set_ylabel("Mel bin"); axes[0, 0].set_xlabel("Frame")

    axes[0, 1].imshow(shaped_spec, vmin=0, vmax=1, **kw)
    axes[0, 1].set_title("B: After Frequency Reshape + Gate + Sobel")
    axes[0, 1].set_ylabel("Mel bin"); axes[0, 1].set_xlabel("Frame")

    # Waveforms
    t_raw = np.linspace(0, len(audio_raw)    / SR, len(audio_raw))
    t_pol = np.linspace(0, len(audio_polished)/ SR, len(audio_polished))
    axes[1, 0].plot(t_raw, audio_raw,       color="#e06c75", linewidth=0.4)
    axes[1, 0].set_title("C: Raw Waveform (no polish)");  axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].plot(t_pol, audio_polished,  color="#56b6c2", linewidth=0.4)
    axes[1, 1].set_title("D: Final Waveform (full polish)"); axes[1, 1].set_xlabel("Time (s)")

    for ax in axes[1]:
        ax.set_ylim(-1.05, 1.05); ax.set_ylabel("Amplitude")
        ax.axhline(0, color="#444", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[voice] Figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DinoSynthesis High-Fidelity Voice Pipeline")
    parser.add_argument("--checkpoint",  default="auto")
    parser.add_argument("--out_dir",     default=OUT_DIR)
    parser.add_argument("--mass",        type=float, default=0.20,
                        help="Mass anchor blend weight (0=pure Bird/Croc, 1=pure Mass). Default=0.20.")
    parser.add_argument("--croc_bias",   type=float, default=0.55,
                        help="Croc vs Bird split of the midpoint. 0.5=equal, 0.6=croc-lean. Default=0.55.")
    parser.add_argument("--jitter",      type=float, default=0.04,
                        help="Latent jitter std-dev for ensemble decode. Default=0.04.")
    parser.add_argument("--n_jitter",    type=int,   default=8,
                        help="Number of jitter samples to average. Default=8.")
    parser.add_argument("--gl_iters",   type=int,   default=256,
                        help="Griffin-Lim iterations. Default=256.")
    parser.add_argument("--boost_db",   type=float, default=8.0,
                        help="Low-end boost in dB (below boost_hz). Default=8.")
    parser.add_argument("--cut_db",     type=float, default=14.0,
                        help="High-end cut in dB (above cut_hz). Default=14.")
    parser.add_argument("--boost_hz",   type=float, default=400.0,
                        help="Frequency below which to boost. Default=400 Hz.")
    parser.add_argument("--cut_hz",     type=float, default=1800.0,
                        help="Frequency above which to cut. Default=1800 Hz.")
    parser.add_argument("--lpf",        type=float, default=700.0,
                        help="Throat LPF cutoff in Hz. Default=700 Hz.")
    parser.add_argument("--sub",        type=float, default=0.40,
                        help="Sub-harmonic blend (0-1). Default=0.40.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if args.checkpoint == "auto":
        gan = os.path.join(ROOT, "checkpoints_gan", "best_gen.pt")
        vae = os.path.join(ROOT, "checkpoints", "best.pt")
        args.checkpoint = gan if os.path.exists(gan) else vae

    print(f"\n{'='*64}")
    print("  DinoSynthesis — High-Fidelity Voice Pipeline")
    print(f"  checkpoint : {os.path.relpath(args.checkpoint, ROOT)}")
    print(f"  blend      : croc={args.croc_bias:.2f}  bird={1-args.croc_bias:.2f}  mass={args.mass:.2f}")
    print(f"  Hz shaping : boost<{args.boost_hz:.0f}Hz+{args.boost_db:.0f}dB  cut>{args.cut_hz:.0f}Hz-{args.cut_db:.0f}dB")
    print(f"  LPF        : {args.lpf:.0f} Hz   sub-harmonic: {args.sub*100:.0f}%")
    print(f"{'='*64}\n")

    # ── Load model + centroids ─────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)
    print("[voice] Computing centroids …")
    z_bird = class_centroid(model, CENTROID_CLASSES["bird"], device)
    z_croc = class_centroid(model, CENTROID_CLASSES["croc"], device)
    z_mass = class_centroid(model, CENTROID_CLASSES["mass"], device)

    # ── Interpolate ───────────────────────────────────────────────────────────
    bird_w = 1.0 - args.croc_bias
    croc_w = args.croc_bias
    z_mid  = bird_w * z_bird + croc_w * z_croc
    z_dino = (1.0 - args.mass) * z_mid + args.mass * z_mass
    print(f"\n[voice] z_dino norm={z_dino.norm():.4f}  "
          f"(bird={bird_w:.2f} croc={croc_w:.2f} mass={args.mass:.2f})")

    # ── Stage A: Ensemble decode + spectrogram shaping ─────────────────────────
    print(f"[voice] Stage A — ensemble decode (N={args.n_jitter}, σ={args.jitter}) …")
    raw_spec    = ensemble_decode(model, z_dino, device, args.n_jitter, args.jitter)
    print(f"[voice]   raw spec  : mean={raw_spec.mean():.4f}  max={raw_spec.max():.4f}")

    shaped_spec = shape_spectrogram(
        raw_spec,
        boost_below_hz=args.boost_hz, boost_db=args.boost_db,
        cut_above_hz=args.cut_hz,    cut_db=args.cut_db,
        gate=0.10, temporal_smooth=0.7, sobel_strength=0.35,
    )
    print(f"[voice]   shaped spec: mean={shaped_spec.mean():.4f}  max={shaped_spec.max():.4f}")

    # ── Stage B: Audio inversion (no pre-emphasis) ─────────────────────────────
    print(f"[voice] Stage B — Griffin-Lim inversion ({args.gl_iters} iters, preemphasis=OFF) …")
    audio_raw = invert_to_audio(shaped_spec, gl_iters=args.gl_iters)
    centroid_raw = float(np.mean(librosa.feature.spectral_centroid(y=audio_raw, sr=SR)))
    print(f"[voice]   audio_raw  : {len(audio_raw)/SR:.2f}s  centroid={centroid_raw:.0f} Hz")

    # ── Stage C: Physiological polish ─────────────────────────────────────────
    print(f"[voice] Stage C — Physiological Polish (LPF={args.lpf:.0f}Hz, sub={args.sub*100:.0f}%) …")
    audio_polished = dino_polish(audio_raw, lpf_hz=args.lpf, sub_blend=args.sub)
    centroid_pol   = float(np.mean(librosa.feature.spectral_centroid(y=audio_polished, sr=SR)))
    print(f"[voice]   polished   : {len(audio_polished)/SR:.2f}s  centroid={centroid_pol:.0f} Hz")

    # ── Save outputs ───────────────────────────────────────────────────────────
    raw_path  = os.path.join(args.out_dir, "dino_voice_raw.wav")
    pol_path  = os.path.join(args.out_dir, "dino_voice.wav")
    fig_path  = os.path.join(args.out_dir, "dino_voice_spectrogram.png")

    sf.write(raw_path,  audio_raw.astype(np.float32),  SR)
    sf.write(pol_path,  audio_polished.astype(np.float32), SR)
    print(f"[voice] WAV (raw)     → {raw_path}")
    print(f"[voice] WAV (polished)→ {pol_path}")

    save_spectrogram_figure(raw_spec, shaped_spec, audio_raw, audio_polished, fig_path)

    print(f"\n{'='*64}")
    print(f"  DONE")
    print(f"  Spectral centroid: {centroid_raw:.0f} Hz (raw) → {centroid_pol:.0f} Hz (polished)")
    print(f"  Main output : {pol_path}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
