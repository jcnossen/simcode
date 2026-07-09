"""
example/crlb_plot.py

CRLB comparison for SMLM, SIMFLUX and SIMCODE fitting on 6-pattern data.

Simulates N single emitters in a 10x10x6 image stack (10x10 pixels,
6 illumination patterns), then computes the theoretical Cramer-Rao lower bound
for three fitting strategies and plots CRLB versus total signal photons:

  * SMLM:    2D Gaussian fit on the sum of the 6 frames.
  * SIMFLUX: PSF + illumination modulation model on all 6 frames jointly.
  * SIMCODE: two-stage 'normally-distributed intensities' (NDI) fit -- estimate
             each pattern's intensity via an (I, bg) CRLB on the corresponding
             frame, then fit position from the modulated intensities.

Patterns are assumed known exactly. See:
  - smlmtorch/smlm/gaussian_psf.py       -- 2D Gaussian PSF derivatives
  - smlmtorch/smlm/crlb.py               -- poisson_crlb
  - smlmtorch/simflux/model.py           -- SIMFLUXModel (PSF + modulation)
  - smlmtorch/simflux/ndi_fit_torch.py   -- ndi_model, intensity_bg_crlb, ndi_crlb
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from smlmtorch.smlm.gaussian_psf import gauss_psf_2D_fixed_sigma
from smlmtorch.smlm.crlb import poisson_crlb
from smlmtorch.simflux.ndi_fit_torch import (
    ndi_model,
    ndi_crlb,
    intensity_bg_crlb,
)


PATTERN_FRAMES = np.array([[0, 1, 2], [3, 4, 5]])  # axis 0: X-modulated, axis 1: Y-modulated


def simflux_forward(theta, mod, roisize, sigma):
    """Inlined SIMFLUX PSF + modulation model (see smlmtorch.simflux.model.SIMFLUXModel).

    theta : [B, 4] = [x, y, I_total, bg_per_frame]
    mod   : [B, K, 6] = [kx, ky, kz, depth, phase, relint]
    Returns (mu, jac) with mu: [B, K, H, W], jac: [B, K, H, W, 4].
    """
    B = theta.shape[0]
    dev = theta.device
    pos = theta[:, :2]                     # [B, 2]
    I = theta[:, [2]]                      # [B, 1]
    bg = theta[:, [3]]                     # [B, 1]

    k = mod[:, :, :2]                      # [B, K, 2]
    depth = mod[:, :, 3]
    phase_shift = mod[:, :, 4]
    relint = mod[:, :, 5]

    em_phase = (pos[:, None] * k).sum(-1) - phase_shift               # [B, K]
    mod_intensity = (1 + depth * torch.sin(em_phase)) * relint        # [B, K]
    mod_deriv = depth[:, :, None] * k * torch.cos(em_phase)[:, :, None] * relint[:, :, None]

    # Unit-intensity, zero-background PSF at (x, y).
    psf_params = torch.cat(
        [pos, torch.ones(B, 1, device=dev), torch.zeros(B, 1, device=dev)], dim=-1)
    psf_ev, psf_deriv = gauss_psf_2D_fixed_sigma(psf_params, roisize, sigma, sigma)
    # psf_ev: [B, H, W]; psf_deriv: [B, H, W, 4] (dx, dy, dI, dbg)

    phot = I[:, None, None]                                            # [B, 1, 1, 1]
    mu = bg[:, None, None] + mod_intensity[:, :, None, None] * phot * psf_ev[:, None]

    # d mu / d(x,y): both the PSF and the modulation depend on position.
    deriv_pos = phot[..., None] * (
        mod_deriv[:, :, None, None] * psf_ev[:, None, ..., None]
        + mod_intensity[:, :, None, None, None] * psf_deriv[:, None, ..., :2]
    )                                                                  # [B, K, H, W, 2]
    deriv_I = (psf_ev[:, None] * mod_intensity[:, :, None, None])[..., None]
    deriv_bg = torch.ones_like(deriv_I)
    jac = torch.cat([deriv_pos, deriv_I, deriv_bg], dim=-1)             # [B, K, H, W, 4]
    return mu, jac


def make_patterns(pitch_px, depth=0.9):
    """Return [K, 6] modulation array: columns are [kx, ky, kz, depth, phase, relint].
    K = 6: 3 phase steps along X, then 3 along Y."""
    k = 2 * np.pi / pitch_px
    K = 6
    phases = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    mod = np.zeros((K, 6), dtype=np.float32)
    mod[:3, 0] = k                     # kx for axis 0
    mod[3:, 1] = k                     # ky for axis 1
    mod[:3, 4] = phases                # phases for axis 0
    mod[3:, 4] = phases                # phases for axis 1
    mod[:, 3] = depth
    mod[:, 5] = 1.0 / K                # equal relative intensity per pattern
    return mod


def smlm_crlb(theta, roisize, sigma, n_frames):
    """CRLB when the 6 frames are summed into one.

    theta : [B, 4] with columns [x, y, I_total, bg_per_frame].
    """
    theta_sum = theta.clone()
    theta_sum[:, 3] = theta_sum[:, 3] * n_frames  # summed background per pixel
    mu, jac = gauss_psf_2D_fixed_sigma(theta_sum, roisize, sigma, sigma)
    return poisson_crlb(mu, jac)  # [B, 4]


def simflux_crlb(theta, roisize, sigma, mod):
    """CRLB from the full PSF + modulation model over all 6 frames.

    theta : [B, 4] with columns [x, y, I_total, bg_per_frame].
    mod   : [K, 6] modulation table.
    """
    mod_b = mod[None].expand(theta.shape[0], -1, -1)   # [B, K, 6]
    mu, jac = simflux_forward(theta, mod_b, roisize, sigma)
    return poisson_crlb(mu, jac)                       # [B, 4]


def simcode_crlb(theta, roisize, sigma, mod, pattern_frames):
    """CRLB for the SIMCODE two-stage NDI fit.

    Stage 1: for each of the K frames, estimate intensity (and background)
             from the PSF pixels -- gives per-frame sigma_I via the 2x2 (I, bg)
             CRLB (intensity_bg_crlb).
    Stage 2: fit position and per-axis intensity to the K measured intensities,
             treated as Gaussian samples with sigma_I -- gives the final CRLB
             on (x, y, I_axis0, I_axis1) via ndi_crlb.
    """
    dev = theta.device
    B = theta.shape[0]
    K = mod.shape[0]
    naxes, nphase = pattern_frames.shape
    assert K == naxes * nphase

    # Group modulation table by (axis, phase step): [B, naxes, nphase, 6]
    pf = torch.as_tensor(pattern_frames, dtype=torch.long, device=dev)
    mod_grouped = mod[pf][None].expand(B, -1, -1, -1).contiguous()

    I_total = theta[:, 2]
    bg_per_frame = theta[:, 3]

    # Expected per-frame intensity (before PSF) from the NDI model.
    # Split the total signal equally between the two modulation axes.
    ndi_params = torch.zeros(B, 2 + naxes, device=dev)
    ndi_params[:, :2] = theta[:, :2]
    ndi_params[:, 2:] = (I_total / naxes)[:, None]

    I_ev, _ = ndi_model(ndi_params, mod_grouped, ndims=2)  # [B, naxes, nphase]

    # Stage 1: 2x2 (I, bg) CRLB per frame, using a unit-intensity PSF.
    psf_params = torch.stack(
        [theta[:, 0], theta[:, 1],
         torch.ones(B, device=dev), torch.zeros(B, device=dev)], dim=-1)
    psf, _ = gauss_psf_2D_fixed_sigma(psf_params, roisize, sigma, sigma)  # [B, H, W]

    psf_rep = psf.repeat_interleave(K, dim=0)                   # [B*K, H, W]
    I_rep = I_ev.reshape(B * K)                                 # [B*K]
    bg_rep = bg_per_frame.repeat_interleave(K)                  # [B*K]
    ibg_crlb = intensity_bg_crlb(psf_rep, I_rep, bg_rep)        # [B*K, 2]
    I_sig = ibg_crlb[:, 0].reshape(B, naxes, nphase)

    # Stage 2: CRLB on (x, y, I_axis0, I_axis1) from the K intensity samples.
    return ndi_crlb(ndi_params, I_sig, mod_grouped, ndims=2)    # [B, 4]


def sample_images(theta, roisize, sigma, mod):
    """Draw a Poisson realization of a 10x10x6 stack (visual sanity check)."""
    mod_b = mod[None].expand(theta.shape[0], -1, -1)
    mu, _ = simflux_forward(theta, mod_b, roisize, sigma)
    return torch.poisson(mu)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    N = 500                                # emitters per photon count
    roisize = 10
    sigma = 1.3                            # PSF sigma in pixels
    pitch_px = 220.0 / 100.0               # 220 nm pitch at 100 nm/px
    K = 6
    bg_per_frame = 4.0                     # photons/pixel/frame
    mod = torch.tensor(make_patterns(pitch_px, depth=0.9), device=dev)
    photon_counts = np.logspace(2, 4, 12)  # total signal photons over 6 frames

    def make_theta(n, I_total):
        theta = torch.zeros(n, 4, device=dev)
        # Position centered in ROI, jittered within one pattern period.
        theta[:, 0] = roisize / 2 + (torch.rand(n, device=dev) - 0.5) * pitch_px
        theta[:, 1] = roisize / 2 + (torch.rand(n, device=dev) - 0.5) * pitch_px
        theta[:, 2] = I_total
        theta[:, 3] = bg_per_frame
        return theta

    # Show a sample stack at the midpoint photon count.
    sample_theta = make_theta(1, photon_counts[len(photon_counts) // 2])
    stack = sample_images(sample_theta, roisize, sigma, mod)[0].cpu().numpy()
    fig, axs = plt.subplots(1, K, figsize=(2 * K, 2.2))
    for i in range(K):
        axs[i].imshow(stack[i], cmap='gray')
        axs[i].set_title(f'pattern {i}')
        axs[i].axis('off')
    fig.suptitle(f'Example 10x10x6 stack '
                 f'(I={photon_counts[len(photon_counts) // 2]:.0f}, bg={bg_per_frame}/frame)')
    fig.tight_layout()

    # Sweep photon counts and collect CRLBs.
    smlm_c, sf_c, sc_c = [], [], []
    for ph in photon_counts:
        theta = make_theta(N, ph)
        smlm_c.append(smlm_crlb(theta, roisize, sigma, K).mean(0).cpu())
        sf_c.append(simflux_crlb(theta, roisize, sigma, mod).mean(0).cpu())
        sc_c.append(simcode_crlb(theta, roisize, sigma, mod, PATTERN_FRAMES).mean(0).cpu())

    smlm_c = torch.stack(smlm_c).numpy()   # [nphot, 4] -> [x, y, I, bg]
    sf_c = torch.stack(sf_c).numpy()
    sc_c = torch.stack(sc_c).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, name in enumerate(['x', 'y']):
        ax = axes[i]
        ax.loglog(photon_counts, smlm_c[:, i], 'ko-', label='SMLM (summed)')
        ax.loglog(photon_counts, sf_c[:, i], 'bs-', label='SIMFLUX (PSF+mod)')
        ax.loglog(photon_counts, sc_c[:, i], 'r^-', label='SIMCODE (NDI)')
        ax.set_xlabel('Total signal [photons]')
        ax.set_ylabel(f'{name} CRLB [pixels]')
        ax.set_title(f'{name} CRLB  (bg={bg_per_frame} ph/px/frame, sigma={sigma} px)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig('crlb_plot.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
