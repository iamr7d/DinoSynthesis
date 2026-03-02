import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with fallback when channels < groups."""
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    """
    Pre-activation Residual Block: GN → LeakyReLU → Conv → GN → LeakyReLU → Conv
    No spatial subsampling; use a separate strided conv for downsampling.
    """
    def __init__(self, channels: int, drop_p: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            _gn(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _gn(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_p),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class DownBlock(nn.Module):
    """Strided conv (2×) + optional extra channels + ResBlock + SE."""
    def __init__(self, in_ch: int, out_ch: int, drop_p: float = 0.05):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            _gn(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res = ResBlock(out_ch, drop_p)
        self.se  = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.res(self.down(x)))


class UpBlock(nn.Module):
    """ConvTranspose (2×) + ResBlock + SE."""
    def __init__(self, in_ch: int, out_ch: int, drop_p: float = 0.05, final: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=False)
        if final:
            self.post = nn.Sequential(_gn(out_ch), nn.Sigmoid())
        else:
            self.post = nn.Sequential(
                _gn(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(out_ch, drop_p),
                SEBlock(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# DinoVAE
# ─────────────────────────────────────────────────────────────────────────────

class DinoVAE(nn.Module):
    """
    β-VAE for Phylogenetic Audio Interpolation.
    Input  : (B, 1, 128, 256)  — normalized Log-Mel Spectrogram
    Output : same shape

    Architecture:
      Encoder : 4× DownBlock (strided conv + ResBlock + SE)  →  FC bottleneck
      Decoder : FC expand  →  4× UpBlock (ConvTranspose + ResBlock + SE)

    Regularisation:
      • GroupNorm  (no batch-size dependency, great for spectrograms)
      • Dropout2d(0.05) inside every ResBlock
      • Dropout(0.25) on FC bottleneck
      • logvar clamped to [-6, 2]
      • Free-bits: 0.5 nats per latent dimension
    """

    ENC_CHANNELS = (1, 32, 64, 128, 256)   # (in, l1, l2, l3, l4)
    SPATIAL_H    = 8
    SPATIAL_W    = 16
    FLAT_DIM     = 256 * 8 * 16             # 32 768

    def __init__(self, latent_dim: int = 128, drop_p: float = 0.05, fc_drop: float = 0.25):
        super().__init__()
        self.latent_dim = latent_dim
        ch = self.ENC_CHANNELS

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = DownBlock(ch[0], ch[1], drop_p)   # → (32, 64, 128)
        self.enc2 = DownBlock(ch[1], ch[2], drop_p)   # → (64, 32,  64)
        self.enc3 = DownBlock(ch[2], ch[3], drop_p)   # → (128,16,  32)
        self.enc4 = DownBlock(ch[3], ch[4], drop_p)   # → (256, 8,  16)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.fc_drop   = nn.Dropout(fc_drop)
        self.fc_mu     = nn.Linear(self.FLAT_DIM, latent_dim)
        self.fc_logvar = nn.Linear(self.FLAT_DIM, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.FLAT_DIM)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec1 = UpBlock(ch[4], ch[3], drop_p)           # → (128,16,  32)
        self.dec2 = UpBlock(ch[3], ch[2], drop_p)           # → (64, 32,  64)
        self.dec3 = UpBlock(ch[2], ch[1], drop_p)           # → (32, 64, 128)
        self.dec4 = UpBlock(ch[1], ch[0], drop_p, final=True)  # → (1,128, 256)

        self._init_weights()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        h = self.enc4(self.enc3(self.enc2(self.enc1(x))))  # (B,256,8,16)
        h = self.fc_drop(h.flatten(1))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fc_decode(z), 0.2)
        h = h.view(-1, 256, self.SPATIAL_H, self.SPATIAL_W)
        return self.dec4(self.dec3(self.dec2(self.dec1(h))))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, alpha: float = 0.5):
        """Spherical linear interpolation between two spectrograms."""
        with torch.no_grad():
            mu1, lv1 = self.encode(x1)
            mu2, lv2 = self.encode(x2)
            # slerp on means
            z = _slerp(mu1, mu2, alpha)
            return self.decode(z)


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation in latent space."""
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    omega = torch.acos((a_n * b_n).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7))
    sin_o = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_o) * a + (torch.sin(t * omega) / sin_o) * b


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_convergence(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Frobenius-norm spectral convergence: ||x - recon||_F / ||x||_F."""
    num = torch.norm(x - recon, p="fro", dim=(-2, -1))           # (B, C)
    den = torch.norm(x,         p="fro", dim=(-2, -1)).clamp(1e-7)
    return (num / den).mean()


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,     # nats per dimension; 0 = no floor (recommended)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Improved ELBO loss with multi-scale spectral reconstruction.

    recon_loss components:
      • pixel_loss    : 0.5*(MSE + L1) — per-pixel accuracy
      • log_mel_loss  : L1 in log1p-compressed space — up-weights quiet harmonics
      • ms_sc_loss    : spectral convergence at 3 spatial scales (1×, 2×, 4× avg-pool)
                        — penalises coarse frequency structure without needing raw audio

    kl_loss:
      • free-bits per dim (floor); set free_bits=0.0 to disable the floor
        (recommended when using low β so every KL dim always contributes gradient)

    Returns (total, recon, kl) — all batch-mean-reduced.
    """
    # ─── 1. Pixel-space reconstruction (MSE + L1) ─────────────────────────
    pixel_loss = 0.5 * F.mse_loss(recon_x, x) + 0.5 * F.l1_loss(recon_x, x)

    # ─── 2. Log-compressed reconstruction ────────────────────────────────
    # Inputs are normalised log-mel spectrograms; log1p compresses small values
    # further so harmonic overtones (low magnitude) receive stronger gradient.
    x_log = torch.log1p(torch.clamp(x,      min=0.0))
    r_log = torch.log1p(torch.clamp(recon_x, min=0.0))
    log_loss = F.l1_loss(r_log, x_log)

    # ─── 3. Multi-scale spectral convergence (3 spatial resolutions) ─────
    sc_full  = _spectral_convergence(x, recon_x)
    x_2x     = F.avg_pool2d(x,      2, 2)
    r_2x     = F.avg_pool2d(recon_x, 2, 2)
    sc_2x    = _spectral_convergence(x_2x, r_2x)
    x_4x     = F.avg_pool2d(x,      4, 4)
    r_4x     = F.avg_pool2d(recon_x, 4, 4)
    sc_4x    = _spectral_convergence(x_4x, r_4x)
    ms_sc    = (sc_full + sc_2x + sc_4x) / 3.0

    # ─── Combined reconstruction loss ────────────────────────────────────
    recon_loss = 0.45 * pixel_loss + 0.35 * log_loss + 0.20 * ms_sc

    # ─── 4. KL divergence ─────────────────────────────────────────────────
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())   # (B, D)
    if free_bits > 0.0:
        kl_loss = kl_per_dim.clamp(min=free_bits).mean()
    else:
        kl_loss = kl_per_dim.mean().clamp(min=0.0)   # always ≥ 0; no floor

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing Dino-VAE v3 …")
    model = DinoVAE(latent_dim=128)
    x     = torch.randn(4, 1, 128, 256)
    model.train()
    recon, mu, logvar = model(x)
    loss, r, kl = vae_loss_function(recon, x, mu, logvar, beta=1.0)
    print(f"Input    : {x.shape}")
    print(f"Output   : {recon.shape}")
    print(f"Latent   : mu={mu.shape}  logvar={logvar.shape}")
    print(f"Loss     : total={loss.item():.4f}  recon={r.item():.4f}  kl={kl.item():.4f}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params   : {n:,}")
    assert x.shape == recon.shape, "Shape mismatch!"
    print("✓ Architecture Validation Passed")
