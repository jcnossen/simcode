"""
example/estimation_vs_crlb.py

Adds measured estimator RMSD to the CRLB comparison from crlb_plot.py.

Simulates single emitters on a 16x16 ROI (10x10 usable centre + 3-pixel border
so the SIMCODE convnet has spatial context), then runs three estimators:

  * SMLM MLE     -- LM_MLE and LM_MLE_Adaptive on the sum of the 6 frames.
  * SIMFLUX MLE  -- LM_MLE and LM_MLE_Adaptive on the PSF + modulation model, all 6 frames.
  * SIMCODE NN   -- SFLocalizationModel loaded from
                    model/sf_conv_g1.3_tio2_L64_2/. Position is read from the
                    pixel with the highest emitter probability, exactly as in
                    notebooks/colab_plot_crlb.ipynb (see eval_single_emitters).

All three CRLB curves are overlaid for reference. Patterns are known exactly.
"""

import os
import sys
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from smlmtorch.smlm.gaussian_psf import gauss_psf_2D_fixed_sigma
from smlmtorch.smlm.levmar import LM_MLE, LM_MLE_Adaptive
from smlmtorch.simflux.ndi_fit_torch import ndi_fit
from smlmtorch.util.config_dict import config_dict
from smlmtorch.nn.sf_model import SFLocalizationModel

# Reuse the CRLB machinery from the previous example.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crlb_plot import (
    PATTERN_FRAMES,
    make_patterns,
    simflux_forward,
    smlm_crlb,
    simflux_crlb,
    simcode_crlb,
)


# ---------------------------------------------------------------------------
# MLE model wrappers (LM_MLE calls model(params, const_) -> (mu, jac))

class GaussPSFModule(nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = int(roisize)
        self.sigma = float(sigma)

    def forward(self, params, const_=None):
        return gauss_psf_2D_fixed_sigma(params, self.roisize, self.sigma, self.sigma)


class SIMFLUXPSFModule(nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = int(roisize)
        self.sigma = float(sigma)

    def forward(self, params, const_):
        return simflux_forward(params, const_, self.roisize, self.sigma)


# ---------------------------------------------------------------------------
# NN model

def load_simcode_model(device):
    """Load the shipped SIMCODE model (SFLocalizationModel, sigma=1.3 px, K=6)."""
    root = os.path.dirname(os.path.abspath(__file__))
    model_zip = os.path.join(root, '..', 'model', 'model_weights_sf_conv_g1.3_tio2_L64_2.zip')
    model_dir = os.path.join(root, '..', 'model', 'sf_conv_g1.3_tio2_L64_2')
    if not os.path.exists(os.path.join(model_dir, 'checkpoint_1.pt')):
        with zipfile.ZipFile(model_zip, 'r') as zf:
            zf.extractall(os.path.dirname(model_zip))

    cfg = config_dict.load(os.path.join(model_dir, 'config.yaml'))
    model = SFLocalizationModel(**cfg.model)
    ck = torch.load(os.path.join(model_dir, 'checkpoint_1.pt'),
                    map_location='cpu', weights_only=False)
    state = ck['model_state_dict']
    state.pop('output_scale', None)
    model.load_state_dict(state)
    return model.eval().to(device), cfg


def eval_single_emitters(output, edge=3, frame_ix=-1):
    """Pick the pixel with the highest emitter probability inside the cropped
    interior of the ROI, and return its per-feature predictions.
    Returns [batch_size, n_frames, 2*n_gauss] where the last dim is
    (params ..., sigmas ...).  For SFLocalizationModel: params = [x, y, bg, N0..N5]."""
    batch_size, n_frames, n_features, height, width = output.shape
    output = output[:, :, :, edge:-edge, edge:-edge]

    n_gauss = (n_features - 1) // 2
    prob = output[:, :, 0]
    max_ix = prob[:, frame_ix].reshape(batch_size, -1).argmax(-1)
    return output[:, :, 1:].reshape(batch_size, n_frames, n_gauss * 2, -1)[
        torch.arange(batch_size), :, :, max_ix
    ]


def nn_estimate(model, images, edge=3, batch_size=64):
    """Run the NN, return (predicted, predicted_sigma), each [B, n_gauss].
    predicted[:, :2] = (x, y) in ROI pixel coordinates."""
    outs = []
    with torch.no_grad():
        for chunk in torch.split(images, batch_size):
            out, _ = model(chunk)
            outs.append(out)
    out = torch.cat(outs, 0)
    locs = eval_single_emitters(out, edge=edge)[:, -1]  # [B, 2*n_gauss]
    n = locs.shape[-1] // 2
    return locs[:, :n], locs[:, n:]


def simcode_ndi_refine(pred, pred_sig, mod, pattern_frames, roisize):
    """Refine the NN position by fitting the NN's per-pattern intensities to
    the known modulation model (second stage of the SIMCODE pipeline).

    Uses the same defaults as ``smlmtorch.simflux.pattern_estimator.ndi_fit_dataset``
    (which is what simulation_example.py runs), so this analysis stays consistent
    with the real pipeline:
      * ``lambda_=10`` (ndi_fit default)
      * ``iterations=100`` (ndi_fit default)
      * ``normalize_scale=True``
    """
    dev = pred.device
    B = pred.shape[0]
    naxes, nphase = pattern_frames.shape

    # NN outputs (after prob) order: [x, y, bg, N0..N5]
    x_nn = pred[:, 0]
    y_nn = pred[:, 1]
    N = pred[:, 3:3 + naxes * nphase]           # [B, K]
    N_sig = pred_sig[:, 3:3 + naxes * nphase]

    # Group frames into (axis, phase) using PATTERN_FRAMES, [B, naxes, nphase]
    pf = torch.as_tensor(pattern_frames, dtype=torch.long, device=dev)
    I_mu = N[:, pf.flatten()].reshape(B, naxes, nphase)
    I_sig = torch.clamp(N_sig[:, pf.flatten()].reshape(B, naxes, nphase), min=1.0)

    initial = torch.zeros(B, 2 + naxes, device=dev)
    initial[:, 0] = x_nn
    initial[:, 1] = y_nn
    initial[:, 2:] = I_mu.sum(-1)  # per-axis total intensity

    mod_grouped = mod[pf][None].expand(B, -1, -1, -1).contiguous()
    param_limits = torch.tensor([
        [0.0, roisize - 1.0],
        [0.0, roisize - 1.0],
        [1.0, 1e9],
        [1.0, 1e9],
    ], device=dev)

    params, _, _, _ = ndi_fit(initial, I_mu, I_sig, mod_grouped, param_limits,
                              ndims=2, lambda_=10, iterations=100, normalize_scale=True)
    return params  # [B, 4] = [x, y, I_ax0, I_ax1]


# ---------------------------------------------------------------------------
# Simulation

def simulate_stack(theta, mod, roisize, sigma):
    """theta: [B, 4] = [x, y, I_total, bg_per_frame]; returns Poisson [B, K, H, W]."""
    mod_b = mod[None].expand(theta.shape[0], -1, -1)
    mu, _ = simflux_forward(theta, mod_b, roisize, sigma)
    return torch.poisson(mu)


def rmsd(errors, keep=1.0):
    """errors: [B, D]. Returns per-dim RMSD.
    If keep < 1.0, remove the (1-keep) fraction with largest ||error|| first --
    this reveals the 'typical' performance when a few fits jump to the wrong
    pixel and dominate the naive RMSD."""
    if keep >= 1.0:
        return (errors ** 2).mean(0).sqrt()
    norms = errors.norm(dim=-1)
    thresh = torch.quantile(norms, keep)
    mask = norms <= thresh
    return (errors[mask] ** 2).mean(0).sqrt()


# ---------------------------------------------------------------------------
# Main

def sweep_photon_counts(photon_counts, bg_per_frame, N, roisize, edge, sigma,
                        mod, K, nn_model, smlm_mle, sf_mle,
                        smlm_mle_a, sf_mle_a, dev):
    """For each photon count: draw N emitters, run every estimator, and keep the
    raw per-emitter position errors so RMSD can be computed at any trim level.

    Two LM variants are fit per model: the fixed-lambda ``LM_MLE`` (suffix _err)
    and the adaptive-damping ``LM_MLE_Adaptive`` (suffix _err_a)."""
    def make_theta(n, I_total):
        theta = torch.zeros(n, 4, device=dev)
        theta[:, 0] = edge + torch.rand(n, device=dev) * (roisize - 2 * edge)
        theta[:, 1] = edge + torch.rand(n, device=dev) * (roisize - 2 * edge)
        theta[:, 2] = I_total
        theta[:, 3] = bg_per_frame
        return theta

    error_keys = ['smlm_err', 'smlm_err_a', 'sf_err', 'sf_err_a',
                  'sc_err', 'sc_ndi_err']
    crlb_keys = ['smlm_c', 'sf_c', 'sc_c']
    out = {k: [] for k in crlb_keys + error_keys + ['sc_pred_sig']}

    for ph in photon_counts:
        theta = make_theta(N, ph)

        out['smlm_c'].append(smlm_crlb(theta, roisize, sigma, K).mean(0)[:2].cpu())
        out['sf_c'].append(simflux_crlb(theta, roisize, sigma, mod).mean(0)[:2].cpu())
        out['sc_c'].append(simcode_crlb(theta, roisize, sigma, mod, PATTERN_FRAMES).mean(0)[:2].cpu())

        stack = simulate_stack(theta, mod, roisize, sigma)

        # Same jittered init fed to both LM variants for a fair comparison.
        init_smlm = theta.clone()
        init_smlm[:, 3] *= K
        init_smlm[:, :2] += (torch.rand(N, 2, device=dev) - 0.5) * 1.0

        init_sf = theta.clone()
        init_sf[:, :2] += (torch.rand(N, 2, device=dev) - 0.5) * 1.0

        # SMLM MLE, both LM variants
        est = smlm_mle(stack.sum(1), init_smlm.clone(), None)
        out['smlm_err'].append((est[:, :2] - theta[:, :2]).cpu())
        est = smlm_mle_a(stack.sum(1), init_smlm.clone(), None)
        out['smlm_err_a'].append((est[:, :2] - theta[:, :2]).cpu())

        # SIMFLUX MLE, both LM variants
        mod_b = mod[None].expand(N, -1, -1)
        est = sf_mle(stack, init_sf.clone(), mod_b)
        out['sf_err'].append((est[:, :2] - theta[:, :2]).cpu())
        est = sf_mle_a(stack, init_sf.clone(), mod_b)
        out['sf_err_a'].append((est[:, :2] - theta[:, :2]).cpu())

        # SIMCODE NN and NN + NDI refinement (independent of LM choice)
        pred, pred_sig = nn_estimate(nn_model, stack, edge=edge)
        out['sc_err'].append((pred[:, :2] - theta[:, :2]).cpu())
        out['sc_pred_sig'].append(pred_sig[:, :2].mean(0).cpu())

        ndi_est = simcode_ndi_refine(pred, pred_sig, mod, PATTERN_FRAMES, roisize)
        out['sc_ndi_err'].append((ndi_est[:, :2] - theta[:, :2]).cpu())

        # Quick log at the reference (10% trim, x-precision).
        def r(key):
            return rmsd(out[key][-1], keep=0.9).numpy()[0]
        print(f'  bg={bg_per_frame:4.1f}  ph={ph:7.1f}  trim10%  '
              f'SMLM LM {r("smlm_err"):.3f} / adapt {r("smlm_err_a"):.3f}  '
              f'SF LM {r("sf_err"):.3f} / adapt {r("sf_err_a"):.3f}  '
              f'NN {r("sc_err"):.3f}  NN+NDI {r("sc_ndi_err"):.3f}')

    return {
        **{k: torch.stack(out[k]).numpy() for k in crlb_keys},
        **{k: torch.stack(out[k]) for k in error_keys},  # keep on CPU as tensors
        'sc_pred_sig': torch.stack(out['sc_pred_sig']).numpy(),
    }


def rmsd_over_photons(errors, keep=1.0):
    """errors: [nphot, N, D]. Returns [nphot, D] per-photon-count RMSD."""
    return np.stack([rmsd(errors[i], keep).numpy() for i in range(errors.shape[0])])


def plot_comparison(photon_counts, res, bg, sigma, keep, savepath):
    """1x2 figure (x, y).  CRLB dashed for reference; one RMSD line per estimator.
    MLE curves always use full RMSD -- LM converges (or fails to) as a whole,
    it doesn't have "picks the wrong pixel" outliers that trim removes.  Trim
    is applied only to the NN curves, which is where genuine detection failures
    dominate the raw RMSD."""
    trim_label = 'no trim' if keep >= 1.0 else f'NN trim {int(round((1 - keep) * 100))}%'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(['x', 'y']):
        ax = axes[i]
        ax.loglog(photon_counts, res['smlm_c'][:, i], 'k--', label='SMLM CRLB', alpha=0.7)
        ax.loglog(photon_counts, res['sf_c'][:, i],   'b--', label='SIMFLUX CRLB', alpha=0.7)
        ax.loglog(photon_counts, res['sc_c'][:, i],   'r--', label='SIMCODE CRLB (NDI)', alpha=0.7)

        # (key, marker style, label, is_mle) -- MLE lines get keep=1.0 always.
        for key, fmt, lab, is_mle in [
            ('smlm_err',   dict(marker='o', mfc='k',   mec='k', color='k', ls='-'),  'SMLM MLE (LM)',         True),
            ('smlm_err_a', dict(marker='o', mfc='none', mec='k', color='k', ls=':'), 'SMLM MLE (LM adapt)',   True),
            ('sf_err',     dict(marker='s', mfc='b',   mec='b', color='b', ls='-'),  'SIMFLUX MLE (LM)',      True),
            ('sf_err_a',   dict(marker='s', mfc='none', mec='b', color='b', ls=':'), 'SIMFLUX MLE (LM adapt)',True),
            ('sc_err',     dict(marker='^', mfc='g',   mec='g', color='g', ls='-'),  'SIMCODE NN',            False),
            ('sc_ndi_err', dict(marker='D', mfc='r',   mec='r', color='r', ls='-'),  'SIMCODE NN+NDI',        False),
        ]:
            k = 1.0 if is_mle else keep
            r = rmsd_over_photons(res[key], k)
            ax.loglog(photon_counts, r[:, i], **fmt, label=f'{lab} RMSD')

        ax.loglog(photon_counts, res['sc_pred_sig'][:, i], 'g:', alpha=0.6,
                  label='SIMCODE NN pred. sigma')
        ax.set_xlabel('Total signal [photons]')
        ax.set_ylabel(f'{name} error [pixels]')
        ax.set_title(f'{name} precision  (bg={bg} ph/px/frame, sigma={sigma} px, {trim_label})')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7, loc='lower left', ncol=1)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_bg_sweep(photon_counts, bg_results, sigma, keep, savepath,
                  sf_err_key='sf_err_a', lm_label='adaptive LM'):
    """One curve per background for both SIMFLUX MLE and SIMCODE NN+NDI, plus
    the SIMCODE CRLB reference.  SIMFLUX MLE always uses full RMSD (no trim
    bias); NN+NDI uses the requested trim.  ``sf_err_key`` selects which LM's
    SIMFLUX MLE errors to plot ('sf_err' fixed-lambda, 'sf_err_a' adaptive)."""
    trim_label = 'no trim' if keep >= 1.0 else f'NN trim {int(round((1 - keep) * 100))}%'
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f'SIMFLUX MLE + SIMCODE NN+NDI vs background  '
                 f'({lm_label}, sigma={sigma} px, {trim_label})', fontsize=11)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bg_results)))
    for i, name in enumerate(['x', 'y']):
        ax = axes[i]
        for (bg, res), c in zip(bg_results.items(), colors):
            r_sf = rmsd_over_photons(res[sf_err_key], 1.0)   # MLE: no trim
            r_nn = rmsd_over_photons(res['sc_ndi_err'], keep)  # NN: user-requested trim
            ax.loglog(photon_counts, res['sc_c'][:, i], '--', color=c, alpha=0.5,
                      label=f'SIMCODE CRLB, bg={bg}')
            ax.loglog(photon_counts, r_sf[:, i], '^--', color=c, alpha=0.75, mfc='none',
                      label=f'SIMFLUX MLE, bg={bg}')
            ax.loglog(photon_counts, r_nn[:, i], 'o-', color=c,
                      label=f'NN+NDI, bg={bg}')
        ax.set_xlabel('Total signal [photons]')
        ax.set_ylabel(f'{name} error [pixels]')
        ax.set_title(f'{name} precision')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7, loc='lower left', ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    # Match the trained SIMCODE model.
    roisize = 16
    edge = 4
    sigma = 1.3
    pixelsize_nm = 100.0
    pitch_nm = 220.0
    pitch_px = pitch_nm / pixelsize_nm
    K = 6

    mod = torch.tensor(make_patterns(pitch_px, depth=0.9), device=dev)

    print('Loading SIMCODE model...')
    nn_model, cfg = load_simcode_model(dev)
    print(f'  param_names: {nn_model.param_names}')

    # Fit each spot with both LM variants:
    #   * LM_MLE with lambda=1e3 -- the tuned fixed-damping version.
    #   * LM_MLE_Adaptive        -- per-sample step acceptance, no lambda tuning.
    param_range = torch.tensor([
        [0.0, roisize - 1.0],
        [0.0, roisize - 1.0],
        [1.0, 1e9],
        [1e-6, 1e6],
    ], device=dev)
    # Per-model lambda tuning: SMLM MLE fits the summed frame, so its Fisher
    # matrix is roughly K times larger than SIMFLUX's per-frame model.  The
    # scale-invariant damping heuristic in LM_MLE therefore scales up too much
    # for SMLM at lambda=1e3, leaving fits stuck at init at low SNR.  Lambda=1e2
    # for SMLM is the smallest damping that stays stable across the range.
    # LM_MLE_Adaptive avoids this coupling entirely by adapting per sample.
    smlm_mle = LM_MLE(GaussPSFModule(roisize, sigma), param_range,
                      iterations=50, lambda_=1e2).to(dev)
    sf_mle = LM_MLE(SIMFLUXPSFModule(roisize, sigma), param_range,
                    iterations=50, lambda_=1e3).to(dev)
    smlm_mle_a = LM_MLE_Adaptive(GaussPSFModule(roisize, sigma), param_range,
                                 iterations=50).to(dev)
    sf_mle_a = LM_MLE_Adaptive(SIMFLUXPSFModule(roisize, sigma), param_range,
                               iterations=50).to(dev)

    photon_counts = np.logspace(2, 3.5, 10)
    N = 500

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Sweeps at every background we care about.  bg=2 is the "sweet spot" for
    # this trained model (see estimation_vs_crlb_bg.png).
    bg_values = [0.5, 2.0, 4.0, 10.0]
    bg_results = {}
    for bg in bg_values:
        print(f'\n=== bg={bg} sweep ===')
        bg_results[bg] = sweep_photon_counts(
            photon_counts, bg, N, roisize, edge, sigma,
            mod, K, nn_model, smlm_mle, sf_mle,
            smlm_mle_a, sf_mle_a, dev)

    # Per-background comparison plots (SMLM / SIMFLUX / SIMCODE).
    #   * The reference bg=4 in the same three trim variants as bg=2.
    #   * bg=2 with no trim, 5% trim, 10% trim as requested.
    for bg in [4.0, 2.0]:
        res = bg_results[bg]
        for keep, tag in [(1.0, 'notrim'), (0.95, 'trim05'), (0.90, 'trim10')]:
            fn = os.path.join(plot_dir, f'estimation_vs_crlb_bg{bg:g}_{tag}.png')
            plot_comparison(photon_counts, res, bg, sigma, keep, fn)
            print(f'  wrote {os.path.relpath(fn)}')

    # Per-background SIMFLUX MLE + SIMCODE NN+NDI, one plot per LM variant so
    # the LM effect on the MLE curves (SF MLE, which does depend on LM) is
    # visible; the NN+NDI curves are LM-independent and should look identical
    # across the two plots (any residual diff is CUDA reduction non-determinism).
    for keep, tag in [(1.0, 'notrim'), (0.90, 'trim10')]:
        for sf_key, lm_tag, lm_lab in [('sf_err',   'lmfixed', 'fixed-lambda LM'),
                                        ('sf_err_a', 'lmadapt', 'adaptive LM')]:
            fn = os.path.join(plot_dir,
                              f'estimation_vs_crlb_bg_sweep_{tag}_{lm_tag}.png')
            plot_bg_sweep(photon_counts, bg_results, sigma, keep, fn,
                          sf_err_key=sf_key, lm_label=lm_lab)
            print(f'  wrote {os.path.relpath(fn)}')


if __name__ == '__main__':
    main()
