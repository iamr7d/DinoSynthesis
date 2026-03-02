"""
speculative_hybrid.py — Arbitrary Latent-Space Chimera Generator
=================================================================
Synthesize a hybrid vocalization between any two acoustic classes.
Handles the full pipeline from data acquisition through synthesis:

  1. Check for existing tensors
  2. Auto-download from iNaturalist (by taxon ID) if tensors not found
  3. Preprocess audio → normalised log-mel tensors
  4. Encode → compute class centroids
  5. Linear interpolate z vectors
  6. Decode → sharpen → Griffin-Lim + pre-emphasis
  7. Save: WAVs + side-by-side spectrogram comparison PNG

Usage examples
--------------
# Existing classes (no download needed):
python speculative_hybrid.py --anchor_a Tinamou_Tinamus --anchor_b Elephantidae

# New taxon download (Snake + Cat):
python speculative_hybrid.py \
    --anchor_a Snake --taxon_a 85553 \
    --anchor_b Cat   --taxon_b 41939

# Tune audio quality:
python speculative_hybrid.py --anchor_a Crocodylia --anchor_b Whippomorpha \\
    --alpha 0.5 --sharpness 2.0 --preemphasis 0.97 --gl_iter 200
"""

import os
import sys
import glob
import time
import argparse
import requests
import torch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.dino_vae import DinoVAE
from audio_utils import save_wav, save_comparison_figure
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment

AudioSegment.converter = "/home/rahulraj/miniconda3/bin/ffmpeg"

# ── Shared preprocessing constants ───────────────────────────────────────────
_SR           = 22050
_CLIP_LEN     = _SR * 3
_MEL          = T.MelSpectrogram(sample_rate=_SR, n_fft=1024, hop_length=256, n_mels=128)
_AMP_TO_DB    = T.AmplitudeToDB()

CKPT_PATH   = os.path.join(ROOT, "checkpoints", "best.pt")
TENSOR_BASE = os.path.join(ROOT, "DATA", "tensors")
AUDIO_BASE  = os.path.join(ROOT, "DATA", "birds")
OUT_BASE    = os.path.join(ROOT, "output_synthesis", "hybrids")
INAT_API    = "https://api.inaturalist.org/v1"
AUDIO_EXTS  = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ─────────────────────────────────────────────────────────────────────────────

def _inat_collect_urls(taxon_id: int, target: int = 150):
    """Page through iNaturalist observations for taxon_id and collect audio URLs."""
    urls, page = [], 1
    while len(urls) < target:
        try:
            r = requests.get(f"{INAT_API}/observations",
                             headers={"Accept": "application/json"}, timeout=30,
                             params={"taxon_id": taxon_id, "sounds": "true",
                                     "per_page": 200, "page": page})
            r.raise_for_status()
            data    = r.json()
            results = data.get("results", [])
            total   = data.get("total_results", 0)
            if not results:
                break
            for obs in results:
                for snd in obs.get("sounds", []):
                    url = snd.get("file_url") or snd.get("url", "")
                    ext = os.path.splitext(url.split("?")[0])[1].lower() or ".mp3"
                    if url and ext in AUDIO_EXTS:
                        urls.append((url, ext, obs["id"], snd["id"]))
            print(f"    iNat page {page}: +{len(results)} obs → {len(urls)} sounds")
            if page * 200 >= total:
                break
            page += 1
            time.sleep(0.4)
        except Exception as e:
            print(f"    [warn] iNaturalist fetch error: {e}")
            break
    return urls[:target]


def _inat_download(class_name: str, taxon_id: int, target: int = 150) -> str:
    """Download `target` audio files for `class_name` (taxon_id) into DATA/birds/."""
    save_dir = os.path.join(AUDIO_BASE, class_name)
    os.makedirs(save_dir, exist_ok=True)

    existing = [f for f in os.listdir(save_dir)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
    needed   = max(0, target - len(existing))
    if needed == 0:
        print(f"  [{class_name}] Already have {len(existing)} audio files.")
        return save_dir

    print(f"\n  [{class_name}] Downloading from iNaturalist (taxon_id={taxon_id}) …")
    urls = _inat_collect_urls(taxon_id, target=needed + 20)
    print(f"  Found {len(urls)} candidate URLs. Downloading …")

    downloaded = 0
    for url, ext, obs_id, snd_id in urls:
        fname = os.path.join(save_dir, f"inat_{obs_id}_{snd_id}{ext}")
        if os.path.exists(fname):
            continue
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(fname, "wb") as f:
                f.write(resp.content)
            downloaded += 1
        except Exception as e:
            print(f"    [skip] {url}: {e}")
        time.sleep(0.15)

    total_now = len([f for f in os.listdir(save_dir)
                     if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
    print(f"  [{class_name}] Downloaded {downloaded} new files → {total_now} total")
    return save_dir


def _preprocess_class(class_name: str, audio_dir: str, tensor_dir: str) -> int:
    """
    Process all audio files in audio_dir into log-mel tensors in tensor_dir.
    Returns number of tensors saved.
    """
    os.makedirs(tensor_dir, exist_ok=True)
    audio_files = [f for f in sorted(os.listdir(audio_dir))
                   if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
    if not audio_files:
        print(f"  [{class_name}] No audio files found in {audio_dir}")
        return 0

    print(f"  [{class_name}] Preprocessing {len(audio_files)} audio files …")
    saved = 0
    for file_idx, fname in enumerate(audio_files):
        fpath = os.path.join(audio_dir, fname)
        try:
            # Load via pydub (handles mp3/ogg/m4a etc.)
            seg     = AudioSegment.from_file(fpath).set_channels(1).set_frame_rate(_SR)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= float(2 ** (8 * seg.sample_width - 1))
            waveform = torch.from_numpy(samples).unsqueeze(0)   # (1, N)
        except Exception as e:
            print(f"    [skip] {fname}: {e}")
            continue

        n_chunks = max(1, waveform.shape[1] // _CLIP_LEN)
        if waveform.shape[1] < _CLIP_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, _CLIP_LEN - waveform.shape[1]))

        for i in range(n_chunks):
            chunk   = waveform[:, i * _CLIP_LEN:(i + 1) * _CLIP_LEN]
            mel     = _MEL(chunk)
            log_mel = _AMP_TO_DB(mel)
            denom   = log_mel.max() - log_mel.min()
            if denom < 1e-6:
                continue
            spec     = (log_mel - log_mel.min()) / denom
            out_path = os.path.join(tensor_dir,
                                    f"{class_name}_{file_idx:04d}_chunk{i:03d}.pt")
            torch.save(spec, out_path)
            saved += 1

    print(f"  [{class_name}] Saved {saved} tensors → {tensor_dir}")
    return saved


def ensure_tensors(class_name: str, taxon_id: int | None) -> str:
    """
    Return tensor directory for class_name. Downloads + preprocesses if needed.
    """
    tensor_dir = os.path.join(TENSOR_BASE, class_name)
    pts = glob.glob(os.path.join(tensor_dir, "*.pt"))
    if pts:
        print(f"  [{class_name}] {len(pts)} tensors found — skipping download.")
        return tensor_dir

    # Need to build tensors from audio
    audio_dir = os.path.join(AUDIO_BASE, class_name)
    audio_files = glob.glob(os.path.join(audio_dir, "*"))
    audio_files = [f for f in audio_files
                   if os.path.splitext(f)[1].lower() in AUDIO_EXTS]

    if not audio_files:
        if taxon_id is None:
            raise FileNotFoundError(
                f"No tensors or audio for '{class_name}'. "
                f"Provide --taxon_a/--taxon_b to auto-download from iNaturalist, "
                f"or manually place audio files in {audio_dir}/"
            )
        _inat_download(class_name, taxon_id)

    # Preprocess audio → tensors
    print(f"\n  [{class_name}] Preprocessing audio → tensors …")
    _preprocess_class(class_name, os.path.join(AUDIO_BASE, class_name), tensor_dir)
    return tensor_dir


# ─────────────────────────────────────────────────────────────────────────────
# Centroid computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid(model: DinoVAE, tensor_dir: str,
                     max_samples: int = 200) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(tensor_dir, "*.pt")))[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No tensors in {tensor_dir}")

    mus = []
    with torch.no_grad():
        for p in paths:
            x = torch.load(p, map_location=device, weights_only=True)
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = x[:, :, :, :256]
            mu, _ = model.encode(x)
            mus.append(mu.squeeze(0))

    centroid = torch.stack(mus).mean(0)
    print(f"  Centroid: {len(mus)} samples | norm={centroid.norm():.4f}")
    return centroid


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize a hybrid vocalization between two animal classes."
    )
    parser.add_argument("--anchor_a", required=True,
                        help="Class name A (must match DATA/tensors/ subdirectory, "
                             "or will be created via --taxon_a)")
    parser.add_argument("--anchor_b", required=True,
                        help="Class name B")
    parser.add_argument("--taxon_a", type=int, default=None,
                        help="iNaturalist taxon ID for anchor A (if downloading)")
    parser.add_argument("--taxon_b", type=int, default=None,
                        help="iNaturalist taxon ID for anchor B (if downloading)")
    parser.add_argument("--target_sounds", type=int, default=150,
                        help="Target number of audio files to download (default 150)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend weight for anchor_a. alpha=0.5 → 50/50. "
                             "alpha=0.8 → 80%% A + 20%% B.")
    parser.add_argument("--sharpness", type=float, default=1.5,
                        help="Spectrogram sharpening power (0=off, 1.5=mild, 2.5=strong)")
    parser.add_argument("--preemphasis", type=float, default=0.97,
                        help="Pre-emphasis coefficient (0=off, 0.97=standard)")
    parser.add_argument("--gl_iter", type=int, default=150,
                        help="Griffin-Lim iterations (default 150)")
    parser.add_argument("--n_interp", type=int, default=5,
                        help="Number of interpolation steps to render for the sweep (default 5)")
    args = parser.parse_args()

    label = f"{args.anchor_a}_x_{args.anchor_b}"
    out_dir = os.path.join(OUT_BASE, label)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*65)
    print(f"  Speculative Hybrid: {args.anchor_a}  ×  {args.anchor_b}")
    print(f"  alpha={args.alpha}  sharp={args.sharpness}  "
          f"preemph={args.preemphasis}  GL×{args.gl_iter}")
    print("="*65 + "\n")

    # 1. Ensure tensors exist ───────────────────────────────────────
    print("[hybrid] Checking / acquiring data …")
    dir_a = ensure_tensors(args.anchor_a, args.taxon_a)
    dir_b = ensure_tensors(args.anchor_b, args.taxon_b)

    # 2. Load model ────────────────────────────────────────────────
    print("\n[hybrid] Loading model …")
    mod = DinoVAE().to(device)
    mod.eval()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    mod.load_state_dict(ckpt.get("model", ckpt))
    print(f"  best_val = {ckpt.get('best_val', '?')}")

    # 3. Compute centroids ─────────────────────────────────────────
    print(f"\n[hybrid] Centroid: {args.anchor_a}")
    z_a = compute_centroid(mod, dir_a)
    print(f"[hybrid] Centroid: {args.anchor_b}")
    z_b = compute_centroid(mod, dir_b)

    # 4. Single hybrid vector ──────────────────────────────────────
    alpha = args.alpha
    z_hybrid = alpha * z_a + (1 - alpha) * z_b
    print(f"\n[hybrid] z_a norm={z_a.norm():.4f}  "
          f"z_b norm={z_b.norm():.4f}  "
          f"z_hybrid norm={z_hybrid.norm():.4f}")

    kw = dict(n_iter=args.gl_iter, sharpness=args.sharpness,
              preemphasis_coef=args.preemphasis)

    # 5. Decode & save primary hybrid and pure anchors ─────────────
    print("\n[hybrid] Decoding & synthesising audio …")
    specs = {}
    for name, z in [(args.anchor_a, z_a),
                    (f"Hybrid (α={alpha:.1f})", z_hybrid),
                    (args.anchor_b, z_b)]:
        with torch.no_grad():
            recon = mod.decode(z.unsqueeze(0))
        spec_np = recon.squeeze().cpu().numpy()
        specs[name] = spec_np
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        wav_path = os.path.join(out_dir, f"{safe_name}.wav")
        save_wav(spec_np, wav_path, **kw)

    # 6. Interpolation sweep ───────────────────────────────────────
    n_steps = args.n_interp
    if n_steps > 1:
        print(f"\n[hybrid] Rendering {n_steps}-step interpolation sweep …")
        sweep_specs = {}
        for i, a in enumerate(np.linspace(0, 1, n_steps)):
            z_s = float(a) * z_a + (1 - float(a)) * z_b
            with torch.no_grad():
                recon = mod.decode(z_s.unsqueeze(0))
            s = recon.squeeze().cpu().numpy()
            step_label = f"α={a:.2f}\n{args.anchor_a}←→{args.anchor_b}"
            sweep_specs[step_label] = s
            wav_path = os.path.join(out_dir, f"sweep_step{i:02d}_alpha{a:.2f}.wav")
            save_wav(s, wav_path, **kw)
        save_comparison_figure(
            sweep_specs,
            os.path.join(out_dir, f"{label}_sweep.png"),
            suptitle=f"Latent Sweep: {args.anchor_a} → {args.anchor_b}",
            sharpness=args.sharpness,
        )

    # 7. Primary 3-panel comparison ────────────────────────────────
    print("\n[hybrid] Rendering primary comparison figure …")
    save_comparison_figure(
        specs,
        os.path.join(out_dir, f"{label}_comparison.png"),
        suptitle=(f"Speculative Hybrid: {args.anchor_a} × {args.anchor_b}  "
                  f"(α={alpha:.1f})"),
        sharpness=args.sharpness,
    )

    print("\n" + "="*65)
    print(f"  OUTPUT DIRECTORY: {out_dir}")
    wav_count = len(glob.glob(os.path.join(out_dir, "*.wav")))
    png_count = len(glob.glob(os.path.join(out_dir, "*.png")))
    print(f"  {wav_count} WAV files  |  {png_count} PNG figures")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
