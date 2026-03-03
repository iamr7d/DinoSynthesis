# DinoSynthesis: Generative Paleo-Acoustics via Phylogenetic VAE-GAN

[![GitHub](https://img.shields.io/badge/GitHub-iamr7d%2FDinoSynthesis-blue?logo=github)](https://github.com/iamr7d/DinoSynthesis)
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **Reconstruct extinct dinosaur vocalizations using phylogenetic interpolation in latent space.**

DinoSynthesis combines **Variational Autoencoders**, **Generative Adversarial Networks**, and **paleo-acoustic constraints** to synthesise hypothetical dinosaur calls by interpolating between modern birds and crocodilians.

---

## 🎯 Key Innovation

Rather than training on dinosaur audio (impossible—they're extinct), DinoSynthesis:

1. **Trains on living archosaurs** — birds (Neornithes) and crocodiles (sister group)
2. **Learns phylogenetic latent geometry** — encodes each species' vocal anatomy in a shared representation space
3. **Interpolates the "missing link"** — synthesises non-avian dinosaur vocalisations at the evolutionary midpoint
4. **Enforces fossil constraints** — applies morphological ground truth (resonance frequency, body size) as post-synthesis correction

---

## 📊 Quantitative Results

### Evaluation Metrics (Epoch 98)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Latent Displacement Error (LDE)** | **0.0000** | Perfect geometric placement at phylogenetic midpoint |
| **Directional Alignment** | **100.0%** | Synthesis vector perfectly aligned with evolutionary trajectory |
| **Bird/Croc Axis Position** | **50% / 50%** | Genuine "missing link" interpolation (not class averaging) |
| **Spectral Centroid** | 6632.81 Hz | High-frequency avian decoder dominance |
| **F1 Morphological Accuracy** | 0.0% | Demonstrates latent–spectral decoupling (see below) |

### Four-Stage Pipeline Progression

| Feature | Stage 1: β-VAE | Stage 2: Balanced VAE | Stage 3: VAE-GAN | Stage 4: Polish |
|---------|---|---|---|---|
| Dataset | Unbalanced (70:1) | Parity (1:1:1) | Parity (1:1:1) | Post-parity |
| Spectral Clarity | Blurred | Defined | Crisp/Textured | Crisp + Resonant |
| Val Loss | ~0.071 | **0.065** | 0.117 | N/A |
| Reconstruction | Standard Griffin-Lim | Hi-Fi Griffin-Lim | GAN Spectral Critique | Temporal Polish |

---

## 🏗️ Architecture

### Phase 1: Latent Space Learning (Stages 1–3)

```
Audio (Bird/Croc)
       ↓
Mel-Spectrogram (128 bins × 256 frames)
       ↓
DinoVAE Encoder (4× ResBlocks + SE modules)
       ↓
z ~ q(z|x)  [128-D latent vector]
       ↓
Phylogenetic Interpolation:
  z_dino = 0.5 * z_bird + 0.5 * z_croc  (+ mass influence)
       ↓
DinoVAE Decoder (4× UpBlocks)
       ↓
Reconstructed Spectrogram
```

### Phase 2: Adversarial Refinement (Stage 3)

```
VAE Decoder Output
       ↓
Multi-Scale PatchGAN Discriminator
  ├─ Full resolution (critic)
  ├─ 2× downsampled (critic)
  └─ 4× downsampled (critic)
       ↓
Hinge Loss + Feature Matching Loss
       ↓
Sharpened, "Crunchy" Biological Texture
```

### Phase 3: Morphological Post-Processing (Stage 4)

```
Decoded Spectrogram
       ↓
[1] Spectral Gate         [hard threshold noise floor]
[2] Log1p Compression     [emphasise quiet harmonics]
[3] Multi-Scale Spectral  [3-resolution convergence]
[4] Griffin-Lim Inversion [spectrogram → audio waveform]
       ↓
Stage 4: Physiological Polish
  ├─ Infrasonic Sub-Harmonic (−12 semitones, 30% blend)
  ├─ ADSR Breathing Envelope (replace hard transients)
  ├─ Throat Resonance LPF  (Butterworth ≤ 800 Hz)
  └─ Slapback Delay (spatial depth)
       ↓
Final Dinosaur Vocalization (2–4s WAV)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/iamr7d/DinoSynthesis.git
cd DinoSynthesis

# Create conda environment
conda create -n dinosynthesis python=3.13 pytorch::pytorch pytorch::pytorch-cuda=11.8 -y
conda activate dinosynthesis

# Install dependencies
pip install -r requirements.txt
```

### Synthesise Dinosaur Vocalisations

```bash
# Standard synthesis (phylogenetic midpoint + mass influence)
python synthesize.py

# Custom interpolation weights
python synthesize.py --bird_weight 0.4 --croc_weight 0.6

# High-quality output (more Griffin-Lim iterations, stronger sharpening)
python synthesize.py --gl_iter 256 --sharpness 2.5 --gate 0.15

# With latent jittering for ensemble averaging
python synthesize.py --jitter 0.05 --n_jitter 12
```

**Outputs:**
- `output_synthesis/dino_100pct_bird.wav` — pure bird anchor (2.96 s, 22.05 kHz)
- `output_synthesis/dino_100pct_croc.wav` — pure croc anchor (2.96 s, 22.05 kHz)
- `output_synthesis/dino_synthesis.wav` — **🦖 interpolated dinosaur** (2.96 s, 22.05 kHz)
- `output_synthesis/dino_spectrogram_comparison.png` — visual spectrogram side-by-side

### Quantitative Evaluation

```bash
# Evaluate against fossil-derived ground truth (800 Hz Parasaurolophus F1)
python eval_phylogenetic_metrics.py --fossil_target 800

# Custom blend (e.g., pure 1.0 midpoint without mass)
python eval_phylogenetic_metrics.py --blend 1.0 --fossil_target 160

# Generates:
# - research_output/phylogenetic_metrics.json
# - research_output/phylogenetic_metrics_report.md
# - research_output/dino_eval_synthesis.wav
```

### Interactive Web App

```bash
# Start Flask server on http://localhost:5050
python app.py

# Features:
# - Real-time Three.js latent space 3D explorer
# - Click-to-synthesize any point in the space
# - AI animal image generation (SDXL-Turbo)
# - Spectrum analyzer and waveform display
```

---

## 🎵 Audio Outputs & Media Gallery

### Generated Synthesis Files

**Location:** `output_synthesis/`

#### Audio Samples (WAV format, 2.96s each @ 22.05 kHz)

| File | Description | Acoustic Character |
|------|-------------|--------------------|
| `dino_100pct_bird.wav` | **Bird Anchor** — Pure Tinamou tinamus | High-frequency whistles, sibilants, articulate chirps |
| `dino_100pct_croc.wav` | **Croc Anchor** — Pure Crocodylia | Low-frequency growls, guttural rumbles, power |
| `dino_synthesis.wav` | **🦖 DINOSAUR** — 50% Bird + 50% Croc + Mass | Hybrid: bird articulation + croc power + body resonance |

#### Spectrogram Comparison

**File:** `dino_spectrogram_comparison.png`

Visualize the three anchor points and their interpolation:
- **Left**: Bird spectrum (energy concentrated in 2–8 kHz, high-frequency detail)
- **Center**: Dinosaur synthesis (blend of bird articulation with extended low-frequency)
- **Right**: Croc spectrum (energy in 0.5–3 kHz, dominant low-frequency guttural)

The dinosaur spectrogram visually represents the evolutionary "missing link" between avian and reptilian vocalisations.

### How to Listen

```bash
# Play dinosaur vocalization
ffplay output_synthesis/dino_synthesis.wav

# Or use any audio player
aplay output_synthesis/dino_synthesis.wav         # Linux
play output_synthesis/dino_synthesis.wav          # macOS (sox)
start output_synthesis/dino_synthesis.wav         # Windows
```

### How to View Spectrograms

```bash
# View the comparison figure
open output_synthesis/dino_spectrogram_comparison.png    # macOS
eog output_synthesis/dino_spectrogram_comparison.png     # Linux (Eye of GNOME)
feh output_synthesis/dino_spectrogram_comparison.png     # Linux (feh)
```

Or view in your browser:
```bash
python -m http.server 8000 &
# Then open: http://localhost:8000/output_synthesis/
```

---

## 📁 Project Structure

```
DinoSynthesis/
├── README.md                                 # This file
├── train.py                                  # VAE training pipeline
├── train_vae_gan.py                          # GAN adversarial training
├── synthesize.py                             # Batch synthesis script
├── eval_phylogenetic_metrics.py              # Quantitative evaluation
├── app.py                                    # Flask web server
│
├── model/
│   ├── dino_vae.py                          # DinoVAE architecture + loss
│   ├── dino_vae_gan.py                      # VAE-GAN with discriminator
│   └── losses.py                            # Custom loss functions
│
├── checkpoints/
│   ├── best.pt                              # Best VAE (val_recon=0.065539)
│   └── epoch_NNN.pt                         # Periodic checkpoints
│
├── research_output/
│   ├── phylogenetic_metrics_report.md       # 261-line comparative analysis
│   ├── phylogenetic_metrics.json            # Metrics export
│   ├── latent_pca_cache.json                # 3D PCA projection cache
│   └── latent_map.png                       # PCA visualisation
│
├── DATA/
│   └── tensors/
│       ├── Tinamou_Tinamus/                 # Bird anchor (180 samples)
│       ├── Crocodylia/                      # Croc anchor (200 samples)
│       └── Whippomorpha/                    # Mass proxy (200 samples)
│
├── templates/
│   ├── latent.html                          # Interactive 3D explorer
│   ├── synth.html                           # Synthesis control panel
│   └── research/                            # Static page assets
│
├── static/
│   ├── js/three.min.js                      # Three.js r128 (local, no CDN)
│   └── js/OrbitControls.js                  # Camera controls
│
├── output_synthesis/
│   ├── dino_100pct_bird.wav                # Pure bird anchor audio (2.96s)
│   ├── dino_100pct_croc.wav                # Pure croc anchor audio (2.96s)
│   ├── dino_synthesis.wav                  # 🦖 DINOSAUR VOCALIZATION (2.96s)
│   └── dino_spectrogram_comparison.png     # Visual comparison (3 spectrograms)
│
├── logs/                                    # Training logs & metrics
└── requirements.txt                         # Python dependencies
```

---

## 🧬 The Latent–Spectral Decoupling Finding

### The Problem

Despite achieving **100% directional alignment** at the phylogenetic midpoint, the synthesised waveform has a **spectral centroid of 6632 Hz** — far above the fossil-constrained target of 160–800 Hz.

### The Insight

> The **brain** (latent space) is perfectly balanced, but the **vocal apparatus** (decoder) still prefers avian spectral patterns.

This reveals a structural decoupling:
- **Latent geometry** learned the phylogenetic relationships correctly
- **Spectral decoder** learned to prioritise high-frequency structure (bird whistles) because:
  - Avian spectrograms contain more energy-dense high-frequency features
  - These provide stronger gradient signals during reconstruction loss training
  - Even under class parity, the decoder regressed to the spectral patterns with highest information density

### The Solution: Stage 4

This decoupling proves that **morphological post-processing (Stage 4) is not cosmetic**—it is a **forensic necessity** to force the neural output to obey the physical constraints of extinct body plans:

- **Throat LPF (800 Hz cutoff)** — simulates acoustic damping through a 5–7 tonne soft-tissue vocal tract
- **Infrasonic sub-harmonic** — 12 semitones below fundamental mimics large body cavity resonance
- **ADSR envelope** — removes digital artefacts and imposes natural breathing dynamics

---

## 🔮 Future Research Roadmap

### Phase 1: High-Dimensional Phylogenetic Triangulation

**Goal:** Expand from 3-point triangle to 8+ node phylogenetic manifold

- Add anatomical anchors: *Varanidae* (monitor lizards), Ostrich (low-freq booming), Cassowary (high-freq hissing)
- Maintain **1:1:1:1 parity**: balanced sampling ensures manifold geometry
- Conditional vectors ($\mathbf{y}$): encode Vocal Tract Length (cm) + Lung Volume (L) directly

### Phase 2: Neural Architecture Evolution

**Goal:** Replace Griffin-Lim with end-to-end learnable synthesis

- **Latent Diffusion Model (LDM)**: use VAE as compressed space, diffusion for sharp transients
- **DDSP Layer**: differentiable digital signal processing with $F_0$ hard-constrained to [160 Hz, 800 Hz]
- **Native morphology learning**: no post-hoc filtering needed

### Phase 3: Fossil-Grounded Validation

**Goal:** Close the latent–spectral decoupling loop with paleontological data

- **SNR hard-filtering**: discard recordings with background noise; eliminates hiss bias
- **FEA cross-validation**: run Finite Element Analysis on CT-scanned dinosaur skulls to generate secondary frequency ground truth
- **Direct loss term**: $\mathcal{L}_{\text{morph}} = \|F_{\text{observed}} - F_{\text{FEA}}\|_2$

---

## 🧭 Strategic Evolution Plan (Phase 2.0)

> **Primary Objective:** Bridge the "Latent–Spectral Decoupling" by transforming the VAE-GAN from a black-box interpolator into a **Physics-Informed Phylogenetic Morphospace**.

---

### 🛠️ Phase 1: Metric Latent Regularization *(Addressing the "Adams Constraint")*

**Problem:** Standard VAE latent spaces are often non-metric — Euclidean distance does not represent evolutionary time.  
**Solution:** Implement **Phylogenetic Distance Matching (PDM)**.

#### 1.1 Temporal Alignment Loss

$$\mathcal{L}_{\text{phylo}} = \| \text{dist}(z_i, z_j) - k \cdot T_{ij} \|^2$$

where $T_{ij}$ is the **patristic distance** (millions of years) between species $i$ and $j$.  
**Goal:** Ensure the latent space geometry is a linear reflection of the Archosaur evolutionary tree.

#### 1.2 Triplet Anchoring

- Use Birds (Neornithes) and Crocodilians as **hard anchors**
- Implement triplet loss to ensure *within-clade* variance < *between-clade* variance in the latent space

---

### 📈 Phase 2: Brownian Bridge Synthesis *(The "Ancestral State" Engine)*

**Problem:** Simple linear interpolation ($0.5z_b + 0.5z_c$) ignores the stochastic nature of evolution.  
**Solution:** Stochastic Ancestral State Estimation (SASE).

#### 2.1 Latent Brownian Motion

Replace deterministic lerp with a **Brownian Bridge simulation**:

$$z(t) = (1-t)\,z_{\text{anc}} + t\,z_{\text{modern}} + \sigma\,W(t)$$

**Output:** Instead of one `dino_synthesis.wav`, generate an ensemble of 20 samples representing the **probability cloud** of potential calls at the $t=0.5$ evolutionary midpoint.

#### 2.2 Branch Length Conditioning

Incorporate branch length (time) as a variance parameter. Deep-time reconstructions (dinosaurs) should have **higher latent "jitter"** than recent ancestors.

---

### 🛑 Phase 3: Morphological Masking *(The "Raup Constraint")*

**Problem:** The decoder generates high-frequency "bird-like" whistles for high-mass organisms (5–7 tonnes).  
**Solution:** **Mass–Frequency Coupling (MFC)**.

#### 3.1 Physics-Informed Decoder

- **Conditional Injection:** Pass $\log_{10}(\text{Mass})$ directly into the decoder's bottleneck
- **The Raup Filter:** Implement a Differentiable Low-Pass Layer inside the network

Cutoff frequency dynamically calculated:

$$F_c \propto M^{-1/3}$$

This forces the model to find **low-frequency solutions** during training — biologically mandatory for a 5–7 tonne organism.

#### 3.2 Adversarial Mass Critique

Update the **PatchGAN Discriminator** to be "Mass-Aware":  
If the model generates a high-frequency transient for a high-mass latent, the Discriminator penalises it as *"Biologically Impossible."*

---

### 📊 Phase 4: Validation & Visualization

**Goal:** Prove synthesis is grounded in both ML metrics and paleobiology.

#### 4.1 The Raup Plot (Theoretical Morphospace)

Visualise the **Occupied vs. Unoccupied** vocal space:
- Plot Spectral Centroid vs. Mass
- Identify the **"Viable Sliver"** of dinosaur vocalizations in acoustic–mass space

#### 4.2 FEA Cross-Validation

Use existing Finite Element Analysis (FEA) data from *Parasaurolophus* or *T. rex* CT scans as **ground-truth markers** in the latent space.

---

### 📅 Implementation Roadmap (Immediate Sprints)

| Sprint | Task | Target File |
|--------|------|-------------|
| **Sprint A** | Implement `PhyloDistanceLoss` | `model/losses.py` |
| **Sprint B** | Refactor Decoder for Mass-Conditioning | `model/dino_vae.py` |
| **Sprint C** | Develop Brownian Ensemble Script | `synthesize_ensemble.py` |
| **Sprint D** | Generate the "Raup Constraint" Report | `research_output/` |

> *"The fossil record tells us what was; the latent space tells us what could have been. The Raup Constraint ensures we don't imagine the impossible."*

---

## 📚 Conceptual Framework

### Phylogenetic Interpolation vs. Class Averaging

**Class averaging** naively averages bird and croc audio → diluted, unnatural hybrid  
**Phylogenetic interpolation** learns shared latent geometry → meaningful evolutionary trajectory

DinoSynthesis works in latent space precisely *because* the VAE discovers that:
- **Bird latents** encode high-frequency articulation (whistles, sibilants)
- **Croc latents** encode low-frequency guttural power (growls, rumbles)
- **The midpoint** represents the adaptive acoustic niche of a 5–7 tonne cursorial predator

### Why VAE-GAN Matters

Pure VAE reconstructions tend toward blurry "spectral averages"—high reconstruction loss tolerance for low-amplitude features.

The discriminator acts as a **spectral critic**, rejecting any output that regresses toward the mean. This forces training toward sharp, structured features—more "biological," less "digital."

---

## 📖 Usage Examples

### Example 1: Compare Three Anchor Points

```bash
python synthesize.py --sharpness 2.0 --gl_iter 256 --gate 0.15
```

Produces:
- Bird-only synthesis (100% Tinamou)
- Dino synthesis (50% Tinamou + 50% Crocodylia + mass influence)
- Croc-only synthesis (100% Crocodylia)

View spectrogram comparison in `output_synthesis/dino_spectrogram_comparison.png`

### Example 2: Evaluate Against Fossil Target

```bash
python eval_phylogenetic_metrics.py \
  --fossil_target 200 \
  --blend 0.8 \
  --max_samples 300
```

Generates report with:
- Live synthesis at specified blend ratio
- Spectral centroid vs. fossil F1 target
- Latent displacement error from midpoint
- Directional alignment score
- JSON export for further analysis

### Example 3: Interactive Exploration

```bash
python app.py
# Visit http://localhost:5050/latent
# Click any point in the 3D space to synthesize that latent vector
# Real-time spectrogram and waveform display
```

---

## 🔬 Technical Details

### Model Specifications

- **Encoder**: 4 × ResBlock (64 → 128 → 256 → 512 channels) + Squeeze-Excitation
- **Latent dimension**: 128
- **Decoder**: 4 × UpBlock (512 → 256 → 128 → 64 channels)
- **Loss function**: 
  - 45% pixel-domain MSE/L1
  - 35% log1p-domain L1 (emphasises soft harmonics)
  - 20% multi-scale spectral convergence (3 resolutions)
  - β·KL with free bits (β defaults to 0.5, free_bits=0.0)

### Training Hyperparameters (Stage 3)

```
Optimizer:       AdamW (lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.99))
Batch size:      64 (effective=128 with grad accum)
KL schedule:     linear warmup over 20 epochs
Patience:        30 epochs (counts after warmup)
Max epochs:      250
Spectral loss:   [64, 128, 256, 512, 1024, 2048] FFT sizes
GAN discriminator: 3-scale PatchGAN, spectral-normalised
```

### Hardware Requirements

- **GPU**: RTX 4090+ or V100+ (34GB VRAM recommended)
- **CPU**: 8+ cores for data loading
- **Storage**: ~50 GB (model checkpoints + dataset)
- **CUDA**: 11.8+

---

## 📝 Citation

If you use DinoSynthesis in research, please cite:

```bibtex
@thesis{DinoSynthesis2026,
  author = {Raj, Rahul},
  title = {DinoSynthesis: Generative Paleo-Acoustics via Phylogenetic 
           Variational Autoencoders},
  school = {University College London},
  year = {2026},
  note = {GitHub: https://github.com/iamr7d/DinoSynthesis}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit with clear messages (`git commit -m "Add XYZ"`)
4. Push and open a Pull Request

### Areas for Contribution

- [ ] Additional living archosaur anchors (tuatara, gharials, caiman subspecies)
- [ ] FEA integration with paleontology collaborators
- [ ] Diffusion model implementation for Phase 2
- [ ] DDSP layer integration
- [ ] Interactive web UI improvements
- [ ] Documentation & tutorials

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

Commercial use welcome; cite the work.

---

## 🔗 Links & Resources

- **GitHub**: https://github.com/iamr7d/DinoSynthesis
- **Paper**: 
- **Audio Samples**: [link to Zenodo/OSF data deposit]
- **Interactive Demo**: http://localhost:5050/latent (local only)

### Related Work

- [Weishampel et al. (1981)](https://doi.org/10.1073/pnas.78.4.2262) — Parasaurolophus vocalisations & fossil resonance
- [Griffin-Lim (1984)](https://ieeexplore.ieee.org/document/1163405) — Phase reconstruction algorithm
- [HiFi-GAN (2020)](https://arxiv.org/abs/2010.05646) — Adversarial audio synthesis
- [DDSP (2020)](https://arxiv.org/abs/2001.04643) — Differentiable audio synthesis

---

## ✉️ Contact

For questions, collaboration, or feedback:

- **Author**: Rahul Raj
- **Email**: [rahulrajpvr7d.here@gmail.com]
- **Institution**: University of Arts London, MRes Creative Computing
---

<div align="center">

**🦖 Bringing extinct voices back to life through phylogenetic machine learning 🦖**

</div>
