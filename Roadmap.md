PROJECT: Generative Paleo-acoustics via VAE

A Phylogenetic Approach to Extinct Sound Synthesis

1. PROJECT OVERVIEW

This project aims to reconstruct the vocalizations of extinct dinosaurs by training a Variational Autoencoder (VAE) on the acoustic features of their closest living relatives (Archosauria: Birds and Crocodilians). By traversing the "Latent Space" between these groups, we can mathematically estimate the "missing link" sounds of the Mesozoic.

2. PHASE I: BIOLOGICAL DATA MINING

Goal: Create a phylogenetically balanced dataset.

A. The "Living Brackets" (Target Species)

The Croc Side: Alligators, Nile Crocodiles, Caimans (Low-frequency, non-vocal, guttural).

The Bird Side (Ratites): Emus, Cassowaries, Ostriches (Deep, booming, infrasonic resonances).

Scale Anchors: African Elephants, Blue Whales (To provide the "Mass" feature for large theropods).

B. Sourcing Data

Xeno-Canto API: Use for mass-scraping bird and reptile files.

Macaulay Library: Research access for high-fidelity scientific recordings.

Earth Species Project (ESP): For pre-processed, ML-ready bioacoustic datasets.

3. PHASE II: SIGNAL PROCESSING & FEATURE PIPELINE

Goal: Prepare audio for a Convolutional Neural Network.

Normalization: All files to 44.1kHz, Mono, 16-bit PCM.

Segmentation: Fixed 2.0 or 3.0-second clips (Zero-padding if shorter).

Transformations:

Log-Mel Spectrograms: 256 frequency bins (high resolution for animal calls).

Constant-Q Transform (CQT): Better for capturing the harmonic structures found in bird calls.

Augmentation: Time-stretching (simulating larger vocal tracts) and Pitch-shifting (simulating increased body mass).

4. PHASE III: THE VAE ARCHITECTURE

Goal: Build a model that "understands" the relationship between sounds.

The Model Type: $\beta$-VAE (Disentangled Variational Autoencoder).

Encoder: 5-Layer 2D Convolutional layers + Batch Normalization + LeakyReLU.

Bottleneck (The Latent Space): * Dimension: 64-D or 128-D.

KL Divergence Weighting ($\beta$): Increased to force the model to cluster features like "roughness," "pitch," and "resonance" into separate sliders.

Decoder: Transpose Convolutional layers (Symmetric to Encoder).

Loss Function: $\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta \cdot \text{KL}(q(z|x) || p(z))$.

5. PHASE IV: SYNTHESIS & INFERENCE

Goal: "Hallucinate" the dinosaur sound.

Latent Mapping: Visualize the clusters using UMAP. Label points by species and mass.

Phylogenetic Interpolation: * Find the centroid of "Crocodile" vectors ($C$).

Find the centroid of "Emu" vectors ($E$).

Generate a "Dino Vector" ($D$) using Slerp: $D = \text{Slerp}(C, E, t)$.

Physical Scaling: Manually shift the $z$-vector along the "Mass" axis (derived from elephant/whale data) to lower the fundamental frequency.

Resynthesis: Pass the generated spectrogram through HiFi-GAN or Vocos for high-quality audio reconstruction.

6. PHASE V: SCIENTIFIC VALIDATION (PUBLISHABILITY)

Goal: Prove the sound is not arbitrary.

Morphometric Validation: Compare the generated wave's resonant peaks (formants) against the physical dimensions of 3D-scanned dinosaur skulls (e.g., the crest of a Parasaurolophus).

Acoustic Similarity Analysis: Use Fréchet Audio Distance (FAD) to measure how much the synthetic sound shares with its evolutionary relatives.

Blind Listener Test: Conduct a study where zoologists identify if the sounds "fit" the physical constraints of the animal's size.

7. TECH STACK

Language: Python 3.10+

Libraries: PyTorch, Torchaudio, Librosa, Weights & Biases.

Visuals: Matplotlib, Plotly (for interactive 3D latent space exploration).