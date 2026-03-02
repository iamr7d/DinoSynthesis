"""
audio_utils.py — Shared high-fidelity audio utilities for DinoSynthesis
========================================================================
Fixes the "metallic static / latent oversmoothing" problem with four
corrective techniques:

  1. Spectral gate         — hard threshold kills the noise floor
  2. Power-law sharpening  — γ-correction boosts formants over background
  3. Sobel vertical filter — restores transient/chirp vertical definition
  4. Pre-emphasis filter   — restores HF texture lost in mel inversion
  5. Latent jittering      — stochastic sampling removes "perfect average" flatness

All synthesis paths in synthesize.py, app.py, and speculative_hybrid.py
should import exclusively from this module so tuning in one place applies
everywhere.
"""

import io
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Shared constants (match pipeline/dino_data_pipeline.py) ──────────────────
SR         = 22050
N_FFT      = 1024
HOP_LENGTH = 256
N_MELS     = 128
DB_RANGE   = 80.0    # dB dynamic range used during normalisation
DB_CEIL    = 0.0     # dB ceiling (0 dBFS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spectral gate (hard threshold)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_gate(spec: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Hard-threshold the noise floor.

    VAE decoders output a diffuse probability fog below the real signal.
    Zeroing everything under `threshold` converts "cloud" to "structure"
    before Griffin-Lim — the phase estimator stops wasting iterations on
    background noise and converges on real transients instead.

    Args:
        spec      : (n_mels, T) float32 in [0, 1]
        threshold : values below this are set to 0.  0.10–0.20 typical.
    Returns:
        Gated (n_mels, T) float32
    """
    out = spec.copy()
    out[out < threshold] = 0.0
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectrogram sharpening (power-law + Sobel vertical)
# ─────────────────────────────────────────────────────────────────────────────

def sharpen_spectrogram(spec: np.ndarray, power: float = 1.5,
                        clip_percentile: float = 98,
                        gate: float = 0.0,
                        sobel_strength: float = 0.0,
                        temporal_smooth: float = 0.0) -> np.ndarray:
    """
    Boost contrast of a [0,1] normalised log-mel spectrogram.

    Steps applied in order:
      a) Temporal coherence smoothing — Gaussian along time axis (axis=1).
         Removes frame-discontinuity AM beating.  sigma=0.8–1.5 typical.
      b) Optional spectral gate     — zero-out values < gate
      c) Outlier clipping           — maps top clip_percentile% to 1.0
      d) Power-law sharpening       — spec^power (suppresses floor)
      e) Sobel-Y vertical sharpening

    Args:
        spec            : (n_mels, T) float32 in [0, 1]
        power           : gamma exponent. 1.0=off, 1.5=mild, 2.5=strong
        clip_percentile : percentile normalisation before γ
        gate            : hard floor threshold (0 = off)
        sobel_strength  : weight of vertical Sobel response (0 = off)
        temporal_smooth : Gaussian σ along time axis. 0=off, 0.8=mild, 1.5=strong.
                          Reduces frame-discontinuity beating without affecting pitch.
    Returns:
        Processed (n_mels, T) float32 in [0, 1]
    """
    import scipy.ndimage as ndi

    s = spec.copy().astype(np.float32)

    # a: temporal coherence smoothing (axis=1 is time)
    if temporal_smooth > 0:
        s = ndi.gaussian_filter1d(s, sigma=temporal_smooth, axis=1)
        s = np.clip(s, 0.0, 1.0)

    # b: spectral gate
    if gate > 0:
        s = spectral_gate(s, threshold=gate)

    # c+d: outlier clip + power
    if power > 0:
        vmax = np.percentile(s, clip_percentile)
        if vmax > 1e-6:
            s = np.clip(s, 0.0, vmax) / vmax
            s = np.power(s, power)

    # e: Sobel-Y (vertical — along frequency axis) sharpening
    if sobel_strength > 0:
        sobel_k  = np.array([-1, 0, 1], dtype=np.float32).reshape(3, 1)
        edge_map = np.abs(ndi.convolve(s, sobel_k, mode='reflect'))
        e_max = edge_map.max()
        if e_max > 1e-6:
            edge_map /= e_max
        s = np.clip(s + sobel_strength * edge_map, 0.0, 1.0)

    return s.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. High-quality Griffin-Lim
# ─────────────────────────────────────────────────────────────────────────────

def hifi_griffinlim(spec_01: np.ndarray,
                    n_iter: int = 128,
                    preemphasis_coef: float = 0.97,
                    sharpness: float = 0.0,
                    gate: float = 0.0,
                    sobel_strength: float = 0.0,
                    temporal_smooth: float = 0.0) -> np.ndarray:
    """
    [0,1] normalised log-mel spectrogram → audio waveform.

    Pipeline:
      1. Temporal coherence smoothing  (Gaussianσ along time, removes AM beating)
      2. Spectral gate                 (kills noise floor)
      3. Power-law + Sobel sharpening  (restores formants)
      4. De-normalise [0,1] → dB
      5. dB → linear power
      6. Mel⁻¹ → STFT magnitude
      7. Griffin-Lim phase estimation
      8. Pre-emphasis filter
      9. Peak normalise to 0.9 FS
    """
    if sharpness > 0 or gate > 0 or sobel_strength > 0 or temporal_smooth > 0:
        spec_01 = sharpen_spectrogram(spec_01, power=sharpness,
                                       gate=gate,
                                       sobel_strength=sobel_strength,
                                       temporal_smooth=temporal_smooth)

    # 3. De-normalise
    spec_db    = spec_01 * DB_RANGE + (DB_CEIL - DB_RANGE)   # → [−80, 0] dB

    # 4. dB → linear power
    spec_power = librosa.db_to_power(spec_db)

    # 5. Invert mel filterbank
    S_linear = librosa.feature.inverse.mel_to_stft(
        spec_power,
        sr=SR,
        n_fft=N_FFT,
        power=2.0,
    )

    # 6. Griffin-Lim
    audio = librosa.griffinlim(
        S_linear,
        n_iter=n_iter,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
    )

    # 7. Pre-emphasis (restores high-frequency texture)
    if preemphasis_coef > 0:
        audio = librosa.effects.preemphasis(audio, coef=preemphasis_coef)

    # 8. Peak normalise
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak * 0.9

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save WAV to path
# ─────────────────────────────────────────────────────────────────────────────

def save_wav(spec_01: np.ndarray, out_path: str,
             n_iter: int = 128,
             preemphasis_coef: float = 0.97,
             sharpness: float = 0.0,
             gate: float = 0.0,
             sobel_strength: float = 0.0,
             temporal_smooth: float = 0.0) -> np.ndarray:
    """Synthesise audio from a [0,1] mel spectrogram and write to disk."""
    audio = hifi_griffinlim(spec_01, n_iter=n_iter,
                             preemphasis_coef=preemphasis_coef,
                             sharpness=sharpness,
                             gate=gate,
                             sobel_strength=sobel_strength,
                             temporal_smooth=temporal_smooth)
    sf.write(out_path, audio, SR, subtype="PCM_16")
    duration = len(audio) / SR
    print(f"  Saved WAV  → {out_path}  ({duration:.2f}s, "
          f"sharp={sharpness} gate={gate} sobel={sobel_strength} "
          f"smooth={temporal_smooth} preemph={preemphasis_coef})")
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# 4. WAV to in-memory bytes (for Flask)
# ─────────────────────────────────────────────────────────────────────────────

def wav_bytes(spec_01: np.ndarray,
              n_iter: int = 128,
              preemphasis_coef: float = 0.97,
              sharpness: float = 0.0,
              gate: float = 0.0,
              sobel_strength: float = 0.0,
              temporal_smooth: float = 0.0) -> bytes:
    """Return WAV file contents as bytes without touching disk."""
    audio = hifi_griffinlim(spec_01, n_iter=n_iter,
                             preemphasis_coef=preemphasis_coef,
                             sharpness=sharpness,
                             gate=gate,
                             sobel_strength=sobel_strength,
                             temporal_smooth=temporal_smooth)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Spectrogram → PNG base64 string (for Flask)
# ─────────────────────────────────────────────────────────────────────────────

def spec_to_png_b64(spec_01: np.ndarray, title: str = "",
                    sharpness: float = 0.0,
                    gate: float = 0.0,
                    sobel_strength: float = 0.0,
                    temporal_smooth: float = 0.0) -> str:
    """
    Render a mel spectrogram as a dark-theme PNG and return base64 string.
    Applies the same processing stack as the audio path for visual consistency.
    """
    needs_proc = sharpness > 0 or gate > 0 or sobel_strength > 0 or temporal_smooth > 0
    display_spec = sharpen_spectrogram(spec_01, power=sharpness, gate=gate,
                                        sobel_strength=sobel_strength,
                                        temporal_smooth=temporal_smooth) \
                   if needs_proc else spec_01
    spec_db = display_spec * DB_RANGE + (DB_CEIL - DB_RANGE)

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#111")
    librosa.display.specshow(
        spec_db,
        sr=SR, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel",
        ax=ax, cmap="magma", vmin=-80, vmax=0,
    )
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    fig.tight_layout(pad=0.4)

    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-panel comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_figure(specs_dict: dict, out_path: str,
                           suptitle: str = "DinoSynthesis — Latent Space Interpolation",
                           sharpness: float = 0.0,
                           gate: float = 0.0,
                           sobel_strength: float = 0.0) -> None:
    """
    specs_dict : {title_str: spec_np (128,256) in [0,1]}
    Produces a presentation-ready multi-panel figure.
    """
    needs_sharp = sharpness > 0 or gate > 0 or sobel_strength > 0
    n = len(specs_dict)
    fig = plt.figure(figsize=(6 * n, 6), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(1, n, hspace=0.05, wspace=0.08)

    img_handle = None
    for i, (title, spec) in enumerate(specs_dict.items()):
        display = sharpen_spectrogram(spec, power=sharpness, gate=gate,
                                       sobel_strength=sobel_strength) \
                  if needs_sharp else spec
        spec_db = display * DB_RANGE + (DB_CEIL - DB_RANGE)
        ax = fig.add_subplot(gs[i])
        img_handle = librosa.display.specshow(
            spec_db,
            sr=SR, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel",
            ax=ax, cmap="magma", vmin=-80, vmax=0,
        )
        ax.set_title(title, color="white", fontsize=13, pad=8, fontweight="bold")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        if i > 0:
            ax.set_ylabel("")

    if img_handle is not None:
        cbar = fig.colorbar(img_handle, ax=fig.axes, orientation="vertical",
                            fraction=0.015, pad=0.02)
        cbar.set_label("Amplitude (dB)", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(suptitle, color="white", fontsize=15, y=1.02, fontweight="bold")
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved PNG  → {out_path}")
