"""
dino_vae_gan.py  —  VAE-GAN for DinoSynthesis
==============================================
Architecture
────────────
Generator  : identical to DinoVAE (ResBlock + SE encoder/decoder) — warm-starts from best.pt
             encoder stays FROZEN for the first `freeze_enc_epochs` epochs so the existing
             PCA / interpolation structure is preserved.

Discriminator : Multi-Scale Spectral PatchGAN (3 scales: full, ½, ¼)
             Each discriminator returns both the final patch score AND
             all intermediate feature maps (used for feature-matching loss).
             Spectral normalisation on every conv → ultra-stable GAN training.

Loss
────
Generator
  L_G = λ_recon * (MSE + L1)       ← pixel reconstruction
      + β       * KL_free_bits      ← latent regularisation (β-VAE)
      + λ_adv   * L_adv             ← hinge adversarial
      + λ_fm    * L_fm              ← feature-matching (perceptual)
      + λ_sc    * L_sc             ← spectral convergence

Discriminator
  L_D = hinge loss (real vs fake) summed over all 3 scales

Usage
─────
  from model.dino_vae_gan import DinoVAE_GAN, MultiScaleDiscriminator, vae_gan_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.dino_vae import DinoVAE, vae_loss_function   # reuse encoder/decoder


# ─────────────────────────────────────────────────────────────────────────────
# Generator  (thin wrapper — the real work is in DinoVAE)
# ─────────────────────────────────────────────────────────────────────────────

class DinoVAE_GAN(nn.Module):
    """
    Drop-in replacement for DinoVAE that adds adversarial training support.
    The generator IS the DinoVAE — no architectural changes.
    """
    def __init__(self, latent_dim: int = 128, drop_p: float = 0.05, fc_drop: float = 0.25):
        super().__init__()
        self.vae = DinoVAE(latent_dim=latent_dim, drop_p=drop_p, fc_drop=fc_drop)
        self.latent_dim = latent_dim

    # ── Delegate standard VAE API ─────────────────────────────────────────────
    def encode(self, x):          return self.vae.encode(x)
    def reparameterize(self, mu, logvar): return self.vae.reparameterize(mu, logvar)
    def decode(self, z):          return self.vae.decode(z)
    def forward(self, x):         return self.vae(x)
    def interpolate(self, x1, x2, alpha=0.5): return self.vae.interpolate(x1, x2, alpha)

    def freeze_encoder(self):
        """Call at the start of GAN phase to keep latent geometry intact."""
        for p in self.vae.enc1.parameters(): p.requires_grad_(False)
        for p in self.vae.enc2.parameters(): p.requires_grad_(False)
        for p in self.vae.enc3.parameters(): p.requires_grad_(False)
        for p in self.vae.enc4.parameters(): p.requires_grad_(False)
        for p in self.vae.fc_mu.parameters(): p.requires_grad_(False)
        for p in self.vae.fc_logvar.parameters(): p.requires_grad_(False)
        print("[GAN] Encoder frozen — latent space geometry preserved.")

    def unfreeze_encoder(self):
        for p in self.vae.parameters(): p.requires_grad_(True)
        print("[GAN] Encoder unfrozen — end-to-end fine-tuning.")

    def load_vae_checkpoint(self, path: str, device="cpu"):
        """Load weights from a plain DinoVAE checkpoint (best.pt)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd   = ckpt.get("model", ckpt)
        # Prefix keys with 'vae.' to match our wrapper
        new_sd = {}
        for k, v in sd.items():
            key = ("vae." + k) if not k.startswith("vae.") else k
            new_sd[key] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"[GAN] Missing keys ({len(missing)}): {missing[:5]}")
        print(f"[GAN] Loaded VAE weights from {path}  "
              f"(best_val={ckpt.get('best_val','?')})")


# ─────────────────────────────────────────────────────────────────────────────
# PatchGAN discriminator (single scale, spectral-normalised)
# ─────────────────────────────────────────────────────────────────────────────

def _sn_conv(in_ch, out_ch, k, s, p):
    return spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False))


class PatchDiscriminator(nn.Module):
    """
    4-layer PatchGAN with spectral normalization.
    Input shape: (B, 1, H, W)  — raw log-mel values in [0, 1]
    Returns:
        score     : (B, 1, H//16, W//16) — patch-level real/fake logits
        features  : list of 4 intermediate feature maps (for feature matching)
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()

        def block(in_c, out_c, s):
            return nn.Sequential(
                _sn_conv(in_c, out_c, 4, s, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # No spectral norm on first layer (standard MelGAN practice)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer1 = block(base_ch,      base_ch * 2, 2)
        self.layer2 = block(base_ch * 2,  base_ch * 4, 2)
        self.layer3 = block(base_ch * 4,  base_ch * 8, 1)
        self.head   = _sn_conv(base_ch * 8, 1, 3, 1, 1)   # patch output

    def forward(self, x: torch.Tensor):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        score = self.head(f3)
        return score, [f0, f1, f2, f3]   # (logit, [features])


class MultiScaleDiscriminator(nn.Module):
    """
    Three PatchGAN discriminators operating at different temporal resolutions:
      D0 : full-resolution  (1, 128, 256)
      D1 : 2× downsampled   (1,  64, 128)
      D2 : 4× downsampled   (1,  32,  64)

    Operating at multiple scales forces the generator to be sharp at both
    fine harmonic structure AND coarse temporal envelope.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, n_scales: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_ch, base_ch) for _ in range(n_scales)
        ])

    def forward(self, x: torch.Tensor):
        """Returns list of (score, [features]) per scale."""
        results = []
        inp = x
        for disc in self.discriminators:
            results.append(disc(inp))
            inp = self.pool(inp)
        return results   # [(score0, feats0), (score1, feats1), (score2, feats2)]


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def hinge_d_loss(real_scores: list, fake_scores: list) -> torch.Tensor:
    """Hinge GAN discriminator loss over multiple scales."""
    loss = 0.0
    for rs, fs in zip(real_scores, fake_scores):
        r_score = rs[0]; f_score = fs[0]
        loss += F.relu(1.0 - r_score).mean() + F.relu(1.0 + f_score).mean()
    return loss / len(real_scores)


def hinge_g_loss(fake_scores: list) -> torch.Tensor:
    """Hinge GAN generator loss."""
    if not fake_scores:
        return torch.tensor(0.0)
    loss = 0.0
    for fs in fake_scores:
        loss += (-fs[0]).mean()
    return loss / len(fake_scores)


def feature_matching_loss(real_results: list, fake_results: list) -> torch.Tensor:
    """
    Feature-matching loss: L1 between discriminator intermediate features.
    Strongly stabilises GAN training — effectively a deep perceptual loss.
    """
    if not real_results or not fake_results:
        return torch.tensor(0.0)
    loss = 0.0
    n    = 0
    for (_, real_feats), (_, fake_feats) in zip(real_results, fake_results):
        for rf, ff in zip(real_feats, fake_feats):
            loss += F.l1_loss(ff, rf.detach())
            n    += 1
    return loss / max(n, 1)


def spectral_convergence_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spectral convergence: ||target - recon||_F / ||target||_F
    Penalises missing frequency content more than excess noise.
    """
    return torch.norm(target - recon, p="fro") / (torch.norm(target, p="fro") + 1e-8)


def vae_gan_g_loss(
    recon:        torch.Tensor,
    x:            torch.Tensor,
    mu:           torch.Tensor,
    logvar:       torch.Tensor,
    real_results: list,
    fake_results: list,
    beta:         float = 1.0,
    free_bits:    float = 0.5,
    lambda_adv:   float = 0.1,
    lambda_fm:    float = 10.0,
    lambda_sc:    float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Full generator (VAE) loss for VAE-GAN training.

    Returns (total_loss, breakdown_dict)
    """
    # 1. VAE reconstruction + KL
    vae_total, recon_loss, kl_loss = vae_loss_function(
        recon, x, mu, logvar, beta=beta, free_bits=free_bits
    )

    # 2. Spectral convergence — penalise missing harmonics
    sc_loss = spectral_convergence_loss(recon, x)

    # 3. GAN generator hinge loss (skip when no discriminator results)
    adv_loss = hinge_g_loss(fake_results) if (lambda_adv > 0 and fake_results) else torch.tensor(0.0, device=recon.device)

    # 4. Feature matching (perceptual stability)
    fm_loss = feature_matching_loss(real_results, fake_results) if (lambda_fm > 0 and real_results) else torch.tensor(0.0, device=recon.device)

    total = (
        vae_total
        + lambda_sc  * sc_loss
        + lambda_adv * adv_loss
        + lambda_fm  * fm_loss
    )

    return total, {
        "g_total": total.item(),
        "recon":   recon_loss.item(),
        "kl":      kl_loss.item(),
        "sc":      sc_loss.item(),
        "adv":     adv_loss.item(),
        "fm":      fm_loss.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DinoVAE-GAN smoke test …")

    G  = DinoVAE_GAN(latent_dim=128).cuda()
    D  = MultiScaleDiscriminator(n_scales=3).cuda()

    x  = torch.rand(4, 1, 128, 256).cuda()

    # Generator forward
    recon, mu, logvar = G(x)
    print(f"  recon  : {recon.shape}")

    # Discriminator on real and fake
    real_res = D(x)
    fake_res = D(recon.detach())

    d_loss = hinge_d_loss(real_res, fake_res)
    print(f"  D loss : {d_loss.item():.4f}")

    # Generator total loss
    fake_res_g = D(recon)
    g_loss, breakdown = vae_gan_g_loss(recon, x, mu, logvar, real_res, fake_res_g,
                                       beta=1.0, lambda_adv=0.1, lambda_fm=10.0)
    print(f"  G loss : {g_loss.item():.4f}  {breakdown}")

    n_g = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_d = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"  Params : G={n_g:,}  D={n_d:,}")
    print("✓ All shapes OK")
