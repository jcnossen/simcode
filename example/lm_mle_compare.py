"""
example/lm_mle_compare.py

Compare the current LM_MLE against LM_MLE_Adaptive on the same simulated
16x16x6 stacks used by estimation_vs_crlb.py.

Two problems the adaptive version tries to solve:
  * The current LM_MLE applies every step unconditionally, so a low-SNR sample
    whose linearised step overshoots simply walks off to the ROI edge and stays
    there.  The adaptive version evaluates the Poisson NLL before/after each
    step and only accepts steps that decrease it.
  * Damping is per-sample and adaptive: shrinks on accept, grows on reject.
    Each emitter finds its own trust region instead of sharing a global lambda.

We test three configs of the current LM (lambda=1e2, 1e3, 1e4) alongside the
adaptive version, on both SMLM MLE (summed 2D Gaussian) and SIMFLUX MLE.

Reads the CRLB machinery from crlb_plot.py and the MLE wrappers from
estimation_vs_crlb.py.  Saves plots to example/plots/lm_compare_bg{bg}.png.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

from smlmtorch.smlm.levmar import LM_MLE, LM_MLE_Adaptive

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crlb_plot import PATTERN_FRAMES, make_patterns, smlm_crlb, simflux_crlb
from estimation_vs_crlb import (
    GaussPSFModule,
    SIMFLUXPSFModule,
    simulate_stack,
    rmsd,
)


def run_mle(mle, smp, init, const_=None):
    return mle(smp, init, const_)


def sweep(photon_counts, bg, N, roisize, edge, sigma, mod, K, fitters, dev):
    """Return dict of per-photon errors [nphot, N, 2] per (fitter_name, kind)."""
    def make_theta(n, I_total):
        theta = torch.zeros(n, 4, device=dev)
        theta[:, 0] = edge + torch.rand(n, device=dev) * (roisize - 2 * edge)
        theta[:, 1] = edge + torch.rand(n, device=dev) * (roisize - 2 * edge)
        theta[:, 2] = I_total
        theta[:, 3] = bg
        return theta

    errs = {name: {'smlm': [], 'sf': []} for name in fitters}
    crlbs = {'smlm': [], 'sf': []}

    mod_b = mod[None].expand(N, -1, -1)

    for ph in photon_counts:
        theta = make_theta(N, ph)
        stack = simulate_stack(theta, mod, roisize, sigma)

        crlbs['smlm'].append(smlm_crlb(theta, roisize, sigma, K).mean(0)[:2].cpu())
        crlbs['sf'].append(simflux_crlb(theta, roisize, sigma, mod).mean(0)[:2].cpu())

        # Same jittered init for every fitter -> fair comparison.
        init_smlm = theta.clone()
        init_smlm[:, 3] = init_smlm[:, 3] * K
        init_smlm[:, :2] += (torch.rand(N, 2, device=dev) - 0.5) * 0.5

        init_sf = theta.clone()
        init_sf[:, :2] += (torch.rand(N, 2, device=dev) - 0.5) * 0.5

        row = [f'bg={bg}  ph={ph:7.1f}']
        for name, (fit_smlm, fit_sf) in fitters.items():
            est = fit_smlm(stack.sum(1), init_smlm, None)
            errs[name]['smlm'].append((est[:, :2] - theta[:, :2]).cpu())

            est = fit_sf(stack, init_sf, mod_b)
            errs[name]['sf'].append((est[:, :2] - theta[:, :2]).cpu())
            r_full = rmsd(errs[name]['sf'][-1])
            r_trim = rmsd(errs[name]['sf'][-1], keep=0.9)
            row.append(f'{name}: SF full={r_full[0]:.3f} trim={r_trim[0]:.3f}')
        print('  ' + '  |  '.join(row))

    return (
        {k: torch.stack(v).numpy() for k, v in crlbs.items()},
        {name: {k: torch.stack(errs[name][k]) for k in errs[name]}
         for name in fitters},
    )


def rmsd_over_photons(err_stack, keep=1.0):
    return np.stack([rmsd(err_stack[i], keep).numpy() for i in range(err_stack.shape[0])])


def plot_compare(photon_counts, crlbs, errs, bg, sigma, keep, savepath, model):
    """One figure per model (SMLM / SF), 2 panels (x, y), each fitter as one line."""
    trim_label = 'no trim' if keep >= 1.0 else f'trim {int(round((1 - keep) * 100))}%'
    styles = {
        'LM lam=1e2': ('#7fb069', 'o'),
        'LM lam=1e3': ('#e6aa04', 's'),
        'LM lam=1e4': ('#d1495b', '^'),
        'LM adaptive': ('#2e4057', 'D'),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, name in enumerate(['x', 'y']):
        ax = axes[i]
        ax.loglog(photon_counts, crlbs[model][:, i], 'k--',
                  label=f'{"SMLM" if model == "smlm" else "SIMFLUX"} CRLB', alpha=0.7)
        for fname, per_model in errs.items():
            r = rmsd_over_photons(per_model[model], keep)
            color, marker = styles[fname]
            ax.loglog(photon_counts, r[:, i],
                      marker=marker, color=color, ls='-', label=fname)
        ax.set_xlabel('Total signal [photons]')
        ax.set_ylabel(f'{name} error [pixels]')
        ax.set_title(f'{name} precision  ({"SMLM MLE" if model == "smlm" else "SIMFLUX MLE"}, '
                     f'bg={bg} ph/px/frame, sigma={sigma} px, {trim_label})')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    roisize, edge, sigma = 16, 3, 1.3
    K = 6
    pitch_px = 220.0 / 100.0
    mod = torch.tensor(make_patterns(pitch_px, depth=0.9), device=dev)

    param_range = torch.tensor([
        [0.0, roisize - 1.0],
        [0.0, roisize - 1.0],
        [1.0, 1e9],
        [1e-6, 1e6],
    ], device=dev)

    gauss = GaussPSFModule(roisize, sigma)
    sf = SIMFLUXPSFModule(roisize, sigma)

    def make_pair(mle_smlm, mle_sf):
        return (mle_smlm.to(dev), mle_sf.to(dev))

    fitters = {
        'LM lam=1e2':  make_pair(
            LM_MLE(gauss, param_range, iterations=50, lambda_=1e2),
            LM_MLE(sf, param_range, iterations=50, lambda_=1e2)),
        'LM lam=1e3':  make_pair(
            LM_MLE(gauss, param_range, iterations=50, lambda_=1e3),
            LM_MLE(sf, param_range, iterations=50, lambda_=1e3)),
        'LM lam=1e4':  make_pair(
            LM_MLE(gauss, param_range, iterations=50, lambda_=1e4),
            LM_MLE(sf, param_range, iterations=50, lambda_=1e4)),
        'LM adaptive': make_pair(
            LM_MLE_Adaptive(gauss, param_range, iterations=50),
            LM_MLE_Adaptive(sf, param_range, iterations=50)),
    }

    photon_counts = np.logspace(2, 3.5, 10)
    N = 500

    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for bg in [2.0, 4.0]:
        print(f'\n=== bg={bg} sweep ===')
        crlbs, errs = sweep(photon_counts, bg, N, roisize, edge, sigma,
                            mod, K, fitters, dev)

        for keep, tag in [(1.0, 'notrim'), (0.9, 'trim10')]:
            for model, model_lab in [('smlm', 'smlm'), ('sf', 'simflux')]:
                fn = os.path.join(plot_dir,
                                  f'lm_compare_{model_lab}_bg{bg:g}_{tag}.png')
                plot_compare(photon_counts, crlbs, errs, bg, sigma, keep, fn, model)
                print(f'  wrote {os.path.relpath(fn)}')


if __name__ == '__main__':
    main()
