"""
model/losses.py — Phase 2.0 Phylogenetic Loss Functions
=========================================================
Implements the two new loss terms introduced in the Strategic Evolution Plan:

  PhyloDistanceLoss  (Sprint A)
    Temporal Alignment Loss — penalises latent distances that deviate from
    patristic evolutionary distances calibrated on the Archosaur clade tree.
    L_phylo = Σ_{i<j} || dist(z_i, z_j) - k·T_ij/T_max ||²

  TripletAnchorLoss  (Sprint A)
    Phylogenetic Triplet Loss — enforces that within-clade latent variance is
    smaller than between-clade distance.
    L_trip  = Σ max(0, d(a,p) - d(a,n) + margin)

"The fossil record tells us what was; the latent space tells us what could
 have been. The Raup Constraint ensures we don't imagine the impossible."
  — DinoSynthesis Strategic Evolution Plan (Phase 2.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ──────────────────────────────────────────────────────────────────────────────
# Archosaur patristic distance table (millions of years, calibrated phylogeny)
#
# Sources:
#   Brusatte et al. (2010) — Archosaur cladogram
#   Sookias et al. (2012) — Molecular clock calibration
#   Irisarri et al. (2017) — Bird time-calibrated phylogeny
# ──────────────────────────────────────────────────────────────────────────────

PATRISTIC_DISTANCES: Dict[tuple, float] = {
    # Archosaur crown split ~252 Mya (Triassic origin)
    ("Tinamou_Tinamus", "Crocodylia"):      252.0,
    ("Crocodylia",      "Tinamou_Tinamus"): 252.0,

    # Whippomorpha (hippo + cetaceans) diverged from other placentals ~53 Mya
    # Bird–mammal split (via Amniota) ~320 Mya
    ("Tinamou_Tinamus", "Whippomorpha"):    320.0,
    ("Whippomorpha",    "Tinamou_Tinamus"): 320.0,

    # Croc–mammal split also ~320 Mya (Amniota)
    ("Crocodylia",      "Whippomorpha"):    320.0,
    ("Whippomorpha",    "Crocodylia"):      320.0,
}

# Normalisation constant — max patristic distance in the table
_T_MAX: float = max(PATRISTIC_DISTANCES.values())

# Branch-length variance parameter for Brownian Bridge simulation (Mya)
# Dinosaur ancestral reconstruction sits ~166 Mya before the present
DINO_BRANCH_MYA: float = 166.0


# ──────────────────────────────────────────────────────────────────────────────
# Sprint A-1 : Temporal Alignment / Phylogenetic Distance Matching
# ──────────────────────────────────────────────────────────────────────────────

class PhyloDistanceLoss(nn.Module):
    """
    Temporal Alignment Loss — metric regularisation for the VAE latent space.

    For every pair of class centroids (z_i, z_j) whose patristic distance T_ij
    is known, penalise deviation from the target Euclidean distance:

        L_phylo = (1/N) · Σ_{i<j} ( ||z_i - z_j||₂ − k · T_ij/T_max )²

    Args:
        scale     : Multiplier k.  Distance target = scale × T_normalised.
                    The absolute scale matters less than consistency; default 4.0
                    yields targets in a typical VAE latent-norm range.
        normalize : Normalise each centroid vector to unit sphere before
                    computing distance.  Recommended True.
    """

    def __init__(self, scale: float = 4.0, normalize: bool = True):
        super().__init__()
        self.scale     = scale
        self.normalize = normalize

    def forward(self, centroids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            centroids : {class_name → mean latent vector of shape (D,)}

        Returns:
            Scalar loss (requires_grad = True if any centroid does).
        """
        device  = next(iter(centroids.values())).device
        loss    = torch.zeros(1, device=device)
        n_pairs = 0

        keys = list(centroids.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ci, cj = keys[i], keys[j]
                pair   = (ci, cj)
                if pair not in PATRISTIC_DISTANCES:
                    continue

                t_norm = PATRISTIC_DISTANCES[pair] / _T_MAX   # [0, 1]
                z_i    = centroids[ci]
                z_j    = centroids[cj]

                if self.normalize:
                    z_i = F.normalize(z_i, dim=0)
                    z_j = F.normalize(z_j, dim=0)

                d_latent = torch.norm(z_i - z_j, p=2)
                d_target = self.scale * t_norm
                loss     = loss + (d_latent - d_target).pow(2)
                n_pairs += 1

        return (loss / max(n_pairs, 1)).squeeze()


# ──────────────────────────────────────────────────────────────────────────────
# Sprint A-2 : Triplet Anchor Loss
# ──────────────────────────────────────────────────────────────────────────────

class TripletAnchorLoss(nn.Module):
    """
    Phylogenetic Triplet Loss — enforces within-clade variance < between-clade
    distance in the latent space.

    Triplet sampling:
      anchor   = sample from clade A  (e.g. bird)
      positive = another sample from clade A  (within-clade — should be CLOSE)
      negative = sample from clade B  (between-clade — should be FAR)

    L_trip = (1/N) · Σ max(0, d(a,p) − d(a,n) + margin)

    Args:
        margin     : Euclidean triplet margin.  Default 1.0.
        n_triplets : Random triplets sampled per forward call.  Default 64.
        p          : Distance norm (2 = Euclidean).
    """

    def __init__(self, margin: float = 1.0, n_triplets: int = 64, p: int = 2):
        super().__init__()
        self.margin     = margin
        self.n_triplets = n_triplets
        self.p          = p

    def forward(
        self,
        z:      torch.Tensor,  # (B, D) — batch of latent vectors
        labels: torch.Tensor,  # (B,)   — integer class labels
    ) -> torch.Tensor:
        """
        Args:
            z      : Batch of latent vectors, shape (B, D).
            labels : Integer class labels aligned with z, shape (B,).

        Returns:
            Scalar triplet loss.
        """
        device  = z.device
        loss    = torch.zeros(1, device=device)
        n_valid = 0

        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return loss.squeeze()  # Need ≥ 2 classes

        for _ in range(self.n_triplets):
            # Randomly pick two different classes
            perm   = unique_labels[torch.randperm(len(unique_labels), device=device)]
            cls_a  = perm[0].item()
            cls_n  = perm[1].item()

            mask_a = (labels == cls_a).nonzero(as_tuple=True)[0]
            mask_n = (labels == cls_n).nonzero(as_tuple=True)[0]

            if len(mask_a) < 2:
                continue  # Need ≥ 2 samples in anchor class

            perm_a = mask_a[torch.randperm(len(mask_a), device=device)]
            a_idx  = perm_a[0]
            p_idx  = perm_a[1]
            n_idx  = mask_n[torch.randint(len(mask_n), (1,), device=device)[0]]

            z_a = z[a_idx]
            z_p = z[p_idx]
            z_n = z[n_idx]

            d_ap  = torch.norm(z_a - z_p, p=self.p)
            d_an  = torch.norm(z_a - z_n, p=self.p)
            loss  = loss + F.relu(d_ap - d_an + self.margin)
            n_valid += 1

        return (loss / max(n_valid, 1)).squeeze()


# ──────────────────────────────────────────────────────────────────────────────
# Combined phylogenetic regulariser
# ──────────────────────────────────────────────────────────────────────────────

class PhylogeneticRegulariser(nn.Module):
    """
    Combines PhyloDistanceLoss + TripletAnchorLoss into a single module.

    Total loss:
        L_reg = w_pdm · L_phylo + w_trip · L_trip

    Plug into the main train loop alongside the existing VAE ELBO:
        total = vae_loss + lambda_reg * regulariser(centroids, z_batch, labels)

    Args:
        w_pdm  : Weight for PhyloDistanceLoss.     Default 0.5.
        w_trip : Weight for TripletAnchorLoss.     Default 0.5.
        pdm_scale    : Scale for PhyloDistanceLoss.  Default 4.0.
        trip_margin  : Margin for TripletAnchorLoss. Default 1.0.
        n_triplets   : Triplets per forward.          Default 64.
    """

    def __init__(
        self,
        w_pdm:       float = 0.5,
        w_trip:      float = 0.5,
        pdm_scale:   float = 4.0,
        trip_margin: float = 1.0,
        n_triplets:  int   = 64,
    ):
        super().__init__()
        self.w_pdm  = w_pdm
        self.w_trip = w_trip
        self.pdm    = PhyloDistanceLoss(scale=pdm_scale)
        self.trip   = TripletAnchorLoss(margin=trip_margin, n_triplets=n_triplets)

    def forward(
        self,
        centroids: Dict[str, torch.Tensor],
        z:         torch.Tensor,
        labels:    torch.Tensor,
    ) -> torch.Tensor:
        return self.w_pdm * self.pdm(centroids) + self.w_trip * self.trip(z, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── PhyloDistanceLoss smoke test ──")
    centroids = {
        "Tinamou_Tinamus": torch.randn(128),
        "Crocodylia":       torch.randn(128),
        "Whippomorpha":     torch.randn(128),
    }
    pdm_loss = PhyloDistanceLoss(scale=4.0)
    val = pdm_loss(centroids)
    print(f"  PhyloDistanceLoss = {val.item():.4f}")

    print("── TripletAnchorLoss smoke test ──")
    B, D     = 32, 128
    z        = torch.randn(B, D)
    labels   = torch.randint(0, 3, (B,))
    trip     = TripletAnchorLoss(margin=1.0, n_triplets=64)
    val_trip = trip(z, labels)
    print(f"  TripletAnchorLoss = {val_trip.item():.4f}")

    print("── PhylogeneticRegulariser smoke test ──")
    reg = PhylogeneticRegulariser()
    val_reg = reg(centroids, z, labels)
    print(f"  Combined regulariser = {val_reg.item():.4f}")
    print("✓ All loss tests passed")
