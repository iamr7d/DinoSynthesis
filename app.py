"""
app.py — DinoSynthesis Interactive Web App
==========================================
Flask backend:
  GET  /            → serve the UI
  GET  /api/status  → model/centroid status
  POST /api/synthesize → upload audio + blend weights → WAV + spectrogram
"""

import os
import io
import sys
import json
import glob
import base64
import tempfile
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from sklearn.decomposition import PCA

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from model.dino_vae import DinoVAE
from audio_utils import wav_bytes, spec_to_png_b64, SR, N_FFT, HOP_LENGTH, N_MELS
from image_gen_local import generate as img_generate, build_prompt as img_build_prompt
from audio_polish import organic_polish

app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"))

# ── Audio constants (must match preprocessing pipeline) ───────────────────────
DB_RANGE   = 80.0

# ── Global state (loaded once at startup) ─────────────────────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model       = None
centroids   = {}   # class_name → latent tensor (latent_dim,)
STATUS      = {"ready": False, "error": None}
LATENT_DATA = {}   # cached PCA result for /latent page

CENTROID_CLASSES = {
    "bird":  "Tinamou_Tinamus",
    "croc":  "Crocodylia",
    "mass":  "Whippomorpha",
}

GROUPS_MAP = {
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

# ── Mel transform (same spec as pipeline) ─────────────────────────────────────
mel_transform   = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, n_mels=N_MELS)
amplitude_to_db = T.AmplitudeToDB()


# ─────────────────────────────────────────────────────────────────────────────
# Model & centroid loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_model():
    global model
    # Prefer the VAE-GAN checkpoint if available (best_gen.pt stores DinoVAE weights)
    gan_path = os.path.join(ROOT, "checkpoints_gan", "best_gen.pt")
    vae_path = os.path.join(ROOT, "checkpoints", "best.pt")
    ckpt_path = gan_path if os.path.exists(gan_path) else vae_path
    print(f"[app] Loading checkpoint: {ckpt_path}")
    model = DinoVAE().to(device)
    model.eval()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    best  = ckpt.get("best_val", ckpt.get("val_recon", "?"))
    print(f"[app] Model loaded — best_val={best:.6f}" if isinstance(best, float) else "[app] Model loaded")


def _compute_centroid(class_name, max_samples=200):
    tensor_dir = os.path.join(ROOT, "DATA", "tensors", class_name, "*.pt")
    paths = sorted(glob.glob(tensor_dir))[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No tensors for class '{class_name}'")
    mus = []
    with torch.no_grad():
        for p in paths:
            x  = torch.load(p, map_location=device, weights_only=True)
            x  = x.unsqueeze(0) if x.dim() == 3 else x
            x  = x[:, :, :, :256]
            mu, _ = model.encode(x)
            mus.append(mu.squeeze(0))
    c = torch.stack(mus).mean(0)
    print(f"[app] Centroid {class_name:25s} | {len(mus):4d} samples | norm={c.norm():.4f}")
    return c


def _init():
    global STATUS
    try:
        _load_model()
        for key, cls in CENTROID_CLASSES.items():
            centroids[key] = _compute_centroid(cls)
        _compute_latent_pca()
        STATUS["ready"] = True
        print("[app] ✓ Ready")
    except Exception as e:
        STATUS["error"] = str(e)
        print(f"[app] INIT ERROR: {e}")


def _compute_latent_pca(max_per_group: int = 426, seed: int = 42) -> None:
    """Project every active tensor into 3D via PCA and cache the result."""
    import random as _random
    global LATENT_DATA

    cache_path = os.path.join(ROOT, "research_output", "latent_pca_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            # Only accept cache if it has 3D coords and inverse-transform stored
            if cached.get("dims") == 3 and "pca_components" in cached:
                LATENT_DATA.update(cached)
                print(f"[app] 3D PCA cache loaded ({len(LATENT_DATA.get('points', []))} pts)")
                return
        except Exception:
            pass

    rng        = _random.Random(seed)
    tensor_dir = os.path.join(ROOT, "DATA", "tensors")
    by_group: dict[str, list] = {"Bird": [], "Reptile": [], "Mass": []}

    for cls_dir in sorted(os.listdir(tensor_dir)):
        cls_path = os.path.join(tensor_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        group = GROUPS_MAP.get(cls_dir)
        if not group:
            continue
        files = sorted(f for f in os.listdir(cls_path) if f.endswith(".pt"))
        by_group[group].extend((cls_dir, os.path.join(cls_path, f)) for f in files)

    sampled: list[tuple] = []
    for group, items in by_group.items():
        sub = rng.sample(items, min(len(items), max_per_group))
        sampled.extend((group, cls, path) for cls, path in sub)

    latents_list, groups_out, classes_out = [], [], []
    with torch.no_grad():
        for group, cls, path in sampled:
            t = torch.load(path, weights_only=True)
            T = t.shape[-1]
            if T >= 256:
                start = (T - 256) // 2
                t = t[:, :, start:start + 256]
            else:
                t = torch.nn.functional.pad(t, (0, 256 - T))
            mu, _ = model.encode(t.unsqueeze(0).to(device))
            latents_list.append(mu.squeeze(0).cpu().numpy())
            groups_out.append(group)
            classes_out.append(cls)

    latents_np = np.array(latents_list)
    pca        = PCA(n_components=3, random_state=0)
    coords     = pca.fit_transform(latents_np)  # (N, 3)

    centroids_3d: dict[str, list] = {}
    for group in ["Bird", "Reptile", "Mass"]:
        idx = [i for i, g in enumerate(groups_out) if g == group]
        if idx:
            centroids_3d[group] = coords[idx].mean(axis=0).tolist()

    points = [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]),
         "group": groups_out[i], "cls": classes_out[i]}
        for i in range(len(coords))
    ]

    LATENT_DATA.update({
        "dims":           3,
        "points":         points,
        "centroids":      centroids_3d,
        "variance":       pca.explained_variance_ratio_.tolist(),
        # Inverse-transform: z_latent = mean + pca_point @ pca_components
        "pca_components": pca.components_.tolist(),   # (3, 128)
        "pca_mean":       pca.mean_.tolist(),          # (128,)
    })

    os.makedirs(os.path.join(ROOT, "research_output"), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(LATENT_DATA, f)
    var = pca.explained_variance_ratio_
    print(f"[app] 3D PCA computed & cached: {len(points)} pts  "
          f"pc1={var[0]:.1%} pc2={var[1]:.1%} pc3={var[2]:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def audio_to_tensor(file_bytes: bytes, filename: str):
    """Raw audio bytes → model-ready (1,1,128,256) tensor on device."""
    suffix = os.path.splitext(filename)[-1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Resample + mono
    if sr != SR:
        waveform = T.Resample(sr, SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # Use the middle 3-second window if longer
    target_len = SR * 3
    if waveform.shape[1] >= target_len:
        start = (waveform.shape[1] - target_len) // 2
        waveform = waveform[:, start:start + target_len]
    else:
        # Pad
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    mel     = mel_transform(waveform)
    log_mel = amplitude_to_db(mel)
    denom   = log_mel.max() - log_mel.min()
    if denom < 1e-6:
        raise ValueError("Audio is silent or too quiet to process.")
    spec = (log_mel - log_mel.min()) / denom   # [0,1]

    # Crop to 256 time frames
    spec = spec[:, :, :256]
    return spec.unsqueeze(0).to(device)         # (1,1,128,256)


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram → WAV bytes and PNG b64 are provided by audio_utils
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/favicon.ico")
def favicon():
    """Serve a simple favicon to suppress 404."""
    from flask import Response
    # Minimal 1x1 transparent PNG (suppress 404 error)
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    import base64
    return Response(base64.b64decode(png_b64), mimetype='image/png')


@app.route("/synth")
def synth():
    return render_template("index.html")


@app.route("/latent")
def latent():
    resp = make_response(render_template("latent.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/research/<path:filename>")
def research_static(filename):
    return send_from_directory(os.path.join(ROOT, "research_output"), filename)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(ROOT, "static"), filename)


@app.route("/api/generate-image", methods=["POST"])
def api_generate_image():
    """Generate an animal portrait using local SDXL-Turbo."""
    try:
        data   = request.get_json(force=True)
        group  = data.get("group", "Blend")
        x      = float(data.get("x", 0.0))
        y      = float(data.get("y", 0.0))
        z      = float(data.get("z", 0.0))
        steps  = int(data.get("steps", 4))          # 1-4 for turbo
        width  = int(data.get("width", 512))
        height = int(data.get("height", 512))
        seed   = abs(int(x * 1000 + y * 100 + z * 10)) % 9999
        prompt = img_build_prompt(group, x, y, z)
        b64    = img_generate(prompt, seed=seed, steps=steps,
                              width=width, height=height)
        return jsonify({"image": b64, "prompt": prompt, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":   STATUS["ready"],
        "error":   STATUS["error"],
        "device":  str(device),
        "centroids": {k: float(v.norm()) for k, v in centroids.items()},
        "latent_points": len(LATENT_DATA.get("points", [])),
    })


@app.route("/api/latent-data")
def api_latent_data():
    if not LATENT_DATA:
        return jsonify({"error": "PCA not computed yet"}), 503
    # Send points + centroids + variance (omit heavy pca_components from list endpoint)
    return jsonify({
        "dims":      LATENT_DATA.get("dims", 3),
        "points":    LATENT_DATA.get("points", []),
        "centroids": LATENT_DATA.get("centroids", {}),
        "variance":  LATENT_DATA.get("variance", []),
    })


@app.route("/api/synth-from-pca", methods=["POST"])
def api_synth_from_pca():
    """Synthesize audio from a 3D PCA coordinate.
    Body JSON: {x, y, z, sharpness?, gate?, sobel_strength?}
    Inverse-transform: z_latent = pca_mean + [x,y,z] @ pca_components
    """
    if not STATUS["ready"]:
        return jsonify({"error": "Model not ready"}), 503
    if not LATENT_DATA or "pca_components" not in LATENT_DATA:
        return jsonify({"error": "PCA data not available"}), 503

    data = request.get_json(force=True)
    try:
        px = float(data["x"])
        py = float(data["y"])
        pz = float(data.get("z", 0.0))
        sharpness = float(data.get("sharpness", 1.8))
        gate      = float(data.get("gate", 0.15))
        sobel     = float(data.get("sobel_strength", 0.0))
        preemph   = float(data.get("preemphasis", 0.97))
        t_smooth  = float(data.get("temporal_smooth", 1.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad request: {e}"}), 400

    # Reconstruct latent vector via PCA inverse transform
    components = np.array(LATENT_DATA["pca_components"], dtype=np.float32)  # (3,128)
    mean_vec   = np.array(LATENT_DATA["pca_mean"],       dtype=np.float32)  # (128,)
    point_3d   = np.array([px, py, pz], dtype=np.float32)                   # (3,)
    z_latent   = mean_vec + point_3d @ components                           # (128,)

    z_tensor = torch.from_numpy(z_latent).unsqueeze(0).to(device)           # (1,128)

    with torch.no_grad():
        recon = model.decode(z_tensor)                                       # (1,1,128,256)
    spec = recon.squeeze().cpu().numpy()

    wav_b64  = base64.b64encode(
        wav_bytes(spec, n_iter=128, sharpness=sharpness,
                  preemphasis_coef=preemph, gate=gate, sobel_strength=sobel,
                  temporal_smooth=t_smooth)
    ).decode()
    spec_b64 = spec_to_png_b64(spec, f"PCA ({px:.2f}, {py:.2f}, {pz:.2f})",
                               sharpness=sharpness, gate=gate, sobel_strength=sobel,
                               temporal_smooth=t_smooth)

    return jsonify({
        "wav":    wav_b64,
        "spec":   spec_b64,
        "z_norm": float(np.linalg.norm(z_latent)),
    })


@app.route("/api/polish", methods=["POST"])
def api_polish():
    """Apply organic post-processing to raw WAV (base64 input)."""
    try:
        data = request.get_json(force=True)
        wav_b64 = data.get("wav")
        if not wav_b64:
            return jsonify({"error": "Missing 'wav' in request"}), 400
        
        import base64
        import io
        import soundfile as sf
        
        # Decode base64 WAV → numpy array
        wav_bytes_decoded = base64.b64decode(wav_b64)
        y, sr = sf.read(io.BytesIO(wav_bytes_decoded))
        
        # Apply organic polish
        y_polished = organic_polish(
            y, sr=sr,
            attack_ms=float(data.get("attack_ms", 150)),
            release_ms=float(data.get("release_ms", 300)),
            lpf_hz=float(data.get("lpf_hz", 3500)),
            sub_blend=float(data.get("sub_blend", 0.3)),
            delay_ms=float(data.get("delay_ms", 40)),
            delay_blend=float(data.get("delay_blend", 0.15))
        )
        
        # Encode result back to base64
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, y_polished, sr)
        wav_buffer.seek(0)
        polished_b64 = base64.b64encode(wav_buffer.read()).decode()
        
        return jsonify({"wav": polished_b64, "status": "polished"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/synthesize", methods=["POST"])
def api_synthesize():
    if not STATUS["ready"]:
        return jsonify({"error": "Model not ready yet"}), 503

    # ── Parse weights + quality params ─────────────────────────────────
    try:
        w_input   = float(request.form.get("w_input", 1.0))
        w_bird    = float(request.form.get("w_bird",  0.0))
        w_croc    = float(request.form.get("w_croc",  0.0))
        w_mass    = float(request.form.get("w_mass",  0.0))
        sharpness = float(request.form.get("sharpness", 1.5))
        preemph   = float(request.form.get("preemphasis", 0.97))
        gate      = float(request.form.get("gate", 0.0))
        sobel     = float(request.form.get("sobel_strength", 0.0))
        t_smooth  = float(request.form.get("temporal_smooth", 0.0))
        jitter    = float(request.form.get("jitter", 0.0))
        n_jitter  = max(1, int(request.form.get("n_jitter", 6)))
    except ValueError:
        return jsonify({"error": "Invalid weight values"}), 400

    # ── Get audio file ─────────────────────────────────────────────
    audio_file = request.files.get("audio")
    has_audio  = audio_file is not None and audio_file.filename

    if not has_audio and w_input > 0:
        return jsonify({"error": "Upload an audio file, or set 'Your Sound' weight to 0"}), 400

    # ── Encode input if provided ───────────────────────────────────
    z_input    = None
    input_spec = None
    input_wav  = None

    if has_audio:
        try:
            file_bytes = audio_file.read()
            x = audio_to_tensor(file_bytes, audio_file.filename)
        except Exception as e:
            return jsonify({"error": f"Could not process audio: {e}"}), 422

        with torch.no_grad():
            mu, _ = model.encode(x)
        z_input    = mu.squeeze(0)
        input_spec = x.squeeze().cpu().numpy()   # (128,256)
        input_wav  = base64.b64encode(
            wav_bytes(input_spec, n_iter=64,
                      sharpness=sharpness, preemphasis_coef=preemph,
                      gate=gate, sobel_strength=sobel)
        ).decode()

    # ── Build blended latent vector ───────────────────────────────
    total = w_input + w_bird + w_croc + w_mass
    if total < 1e-6:
        return jsonify({"error": "At least one weight must be > 0"}), 400

    z_blend = torch.zeros(128, device=device)
    if w_input > 0 and z_input is not None:
        z_blend = z_blend + (w_input / total) * z_input
    if w_bird > 0:
        z_blend = z_blend + (w_bird  / total) * centroids["bird"]
    if w_croc > 0:
        z_blend = z_blend + (w_croc  / total) * centroids["croc"]
    if w_mass > 0:
        z_blend = z_blend + (w_mass  / total) * centroids["mass"]

    # ── Decode (with optional latent jitter for organic variability) ────
    if jitter > 0:
        specs_j = []
        for _ in range(n_jitter):
            z_j = z_blend + jitter * torch.randn_like(z_blend)
            with torch.no_grad():
                specs_j.append(model.decode(z_j.unsqueeze(0)).squeeze().cpu().numpy())
        synth_spec = np.mean(specs_j, axis=0)   # (128,256)
    else:
        with torch.no_grad():
            recon = model.decode(z_blend.unsqueeze(0))
        synth_spec = recon.squeeze().cpu().numpy()   # (128,256)

    # ── Render outputs ────────────────────────────────────────────
    wav_kw  = dict(sharpness=sharpness, preemphasis_coef=preemph,
                   gate=gate, sobel_strength=sobel, temporal_smooth=t_smooth)
    png_kw  = dict(sharpness=sharpness, gate=gate,
                   sobel_strength=sobel, temporal_smooth=t_smooth)
    synth_wav_b64  = base64.b64encode(wav_bytes(synth_spec, n_iter=128, **wav_kw)).decode()
    synth_spec_b64 = spec_to_png_b64(synth_spec, "Synthesized Vocalization", **png_kw)
    input_spec_b64 = spec_to_png_b64(input_spec, "Your Audio", **png_kw) if input_spec is not None else None

    return jsonify({
        "synth_wav":   synth_wav_b64,
        "synth_spec":  synth_spec_b64,
        "input_wav":   input_wav,
        "input_spec":  input_spec_b64,
        "z_norm":      float(z_blend.norm()),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading
    threading.Thread(target=_init, daemon=True).start()
    app.run(host="0.0.0.0", port=5050, debug=False)
