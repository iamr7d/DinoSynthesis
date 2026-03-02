"""
analyze_vocalization.py — Acoustic Feature Analysis
====================================================
Loads the three output WAVs produced by synthesize.py and computes a
suite of librosa acoustic features.  Results are printed as a formatted
table and saved as a bar-chart comparison PNG.

Usage
-----
    python analyze_vocalization.py
    python analyze_vocalization.py --dir output_synthesis --out analysis_chart.png

Features extracted
------------------
  - Spectral centroid   (Hz)  — perceived brightness
  - Spectral bandwidth  (Hz)  — tonal focus / spread
  - Spectral rolloff    (Hz)  — where 85 % of energy is below
  - Zero-crossing rate  (1/s) — noisiness / transient density
  - RMS energy          (dB)  — perceived loudness
  - Spectral flatness   (0-1) — tonal (0) vs noise-like (1)
  - Tempo               (BPM) — rhythmic pulse (beat-tracker)
"""

import os
import sys
import argparse
import warnings
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)  # librosa PySoundFile

# ── Default paths ──────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
DEF_DIR = os.path.join(ROOT, "output_synthesis")
FILES   = {
    "Bird Anchor\n(Tinamou tinamus)":          "dino_100pct_bird.wav",
    "Dinosaur Synthesis\n(Phylo-interpolated)": "dino_synthesis.wav",
    "Croc Anchor\n(Crocodylia)":               "dino_100pct_croc.wav",
}

# ── Colours for each track ─────────────────────────────────────────────────────
COLOURS = ["#4fc3f7", "#a5d6a7", "#ef9a9a"]   # blue, green, red


# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = 22050):
    """Load a WAV file and return (y, sr).  Resamples if necessary."""
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)          # stereo → mono
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> dict:
    """Return a dict of scalar acoustic features."""
    # Frame-level features → take mean
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
    zcr       = librosa.feature.zero_crossing_rate(y).mean() * sr   # convert to 1/s
    rms_db    = librosa.amplitude_to_db(
                    librosa.feature.rms(y=y)).mean()
    flatness  = librosa.feature.spectral_flatness(y=y).mean()
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):     # librosa ≥ 0.10 may return array
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0

    return {
        "Spectral Centroid (Hz)":   float(centroid),
        "Spectral Bandwidth (Hz)":  float(bandwidth),
        "Spectral Rolloff (Hz)":    float(rolloff),
        "Zero-Crossing Rate (1/s)": float(zcr),
        "RMS Energy (dB)":          float(rms_db),
        "Spectral Flatness":        float(flatness),
        "Tempo (BPM)":              float(tempo),
    }


# ─────────────────────────────────────────────────────────────────────────────
def print_table(results: dict):
    """Pretty-print feature comparison table to stdout."""
    tracks = list(results.keys())
    feats  = list(next(iter(results.values())).keys())

    # Header
    col_w = 26
    track_w = 28
    header = f"{'Feature':<{col_w}}" + "".join(
        f"{t.replace(chr(10), ' '):<{track_w}}" for t in tracks)
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    # Rows
    for feat in feats:
        row = f"{feat:<{col_w}}"
        vals = [results[t][feat] for t in tracks]
        for v in vals:
            if abs(v) >= 1000:
                row += f"{v:>{track_w}.1f}"
            elif abs(v) >= 1:
                row += f"{v:>{track_w}.2f}"
            else:
                row += f"{v:>{track_w}.4f}"
        print(row)
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def save_bar_chart(results: dict, out_path: str):
    """Save a dark-theme grouped bar chart of all features."""
    tracks = list(results.keys())
    feats  = list(next(iter(results.values())).keys())
    track_labels = [t.replace("\n", " ") for t in tracks]

    n_feats  = len(feats)
    n_tracks = len(tracks)
    bar_w    = 0.22

    fig = plt.figure(figsize=(max(16, 3 * n_feats), 8), facecolor="#0d0d0d")
    ax  = fig.add_subplot(111, facecolor="#111111")

    x = np.arange(n_feats)
    offsets = np.linspace(-(n_tracks - 1) / 2, (n_tracks - 1) / 2, n_tracks) * bar_w

    for i, (track, colour) in enumerate(zip(tracks, COLOURS)):
        vals = [results[track][f] for f in feats]
        bars = ax.bar(x + offsets[i], vals, bar_w * 0.9,
                      label=track.replace("\n", " "),
                      color=colour, alpha=0.85, edgecolor="none",
                      zorder=3)
        # Value labels
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=7, color=colour, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(" (", "\n(", 1) for f in feats],
                        fontsize=9, color="#dddddd")
    ax.set_ylabel("Value", color="#aaaaaa", fontsize=11)
    ax.tick_params(axis="y", colors="#888888")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(axis="y", color="#333333", linewidth=0.5, zorder=0)
    ax.set_title("Acoustic Feature Comparison — DinoSynthesis Output",
                  color="#ffffff", fontsize=13, pad=16)
    ax.legend(framealpha=0.2, facecolor="#222222", edgecolor="#444444",
               labelcolor="#cccccc", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Acoustic analysis of DinoSynthesis output WAVs")
    parser.add_argument("--dir", default=DEF_DIR,
                        help="Directory containing the three WAV files")
    parser.add_argument("--out", default="output_synthesis/vocalization_analysis.png",
                        help="Output PNG path for the bar chart")
    parser.add_argument("--sr", type=int, default=22050,
                        help="Target sample rate for analysis (default: 22050)")
    args = parser.parse_args()

    results = {}
    missing = []
    for label, fname in FILES.items():
        fpath = os.path.join(args.dir, fname)
        if not os.path.exists(fpath):
            missing.append(fpath)
            continue
        print(f"  Analysing: {fname} …")
        y, sr = load_audio(fpath, sr=args.sr)
        results[label] = extract_features(y, sr)

    if missing:
        print(f"\n[analyze] WARNING — missing files:")
        for p in missing:
            print(f"  {p}")
        if not results:
            print("[analyze] No files found.  Run synthesize.py first.")
            sys.exit(1)

    print_table(results)
    save_bar_chart(results, args.out)


if __name__ == "__main__":
    main()
