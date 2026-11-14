"""FRC of the paper's SIMCODE v2 table (pass 2, sigma <= 12 nm) with and without the post-hoc CDF equalisation of the
sub-pixel coordinate (per network-photon bin, fit on the table itself), same rows, same drift (the Fig 3 drift cache).
Also the folded sub-pixel histograms before / after.  usage: python analysis/pixbias/frc_corr.py 1809_2 [2.5nM_2]"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pblib as B
import frc_fair
from simcode import strict as SP
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

RES = {}
for ds in sys.argv[1:]:
    run = {'1809_2': 'v2_matched', '2.5nM_2': 'v2_matched', '20nM_2': 'v2_m25'}[ds]
    T = np.load(os.path.join(B.REPO, 'out', 'paper2', ds, run, 'locs_p2.npz'))
    m = SP.select(T, crop=SP.run_crop(os.path.join(B.REPO, 'out', 'paper2', ds, run)))
    x, y, ph = T['x'][m], T['y'][m], T['photons'][m].astype(np.float64)
    groups = [b for _, b in B.ph_bin_masks(ph)]
    xc, yc = B.cdf_equalise(x, groups), B.cdf_equalise(y, groups)
    dc = os.path.join(B.REPO, 'out', 'paper2', 'fig', f'{ds}_fig3_matched_sig12_v2_drift.npy')
    nframes = int(T['frame'].max()) + 6
    r = {}
    for lab, (xx, yy) in (('raw', (x, y)), ('cdf_eq', (xc, yc))):
        D = dict(x=xx, y=yy, f=T['frame'][m].astype(np.int64), sx=T['sx'][m].astype(np.float64), sy=T['sy'][m].astype(np.float64))
        E = frc_fair.evaluate(D, nframes, drift_cache=dc if os.path.exists(dc) else None, secondary=False)
        r[lab] = dict(frc_nm=float(E['frc_nm']), n_frc=E['n_frc'], c1x=B.ustats(B.sub(xx))['c1_abs'], c1y=B.ustats(B.sub(yy))['c1_abs'])
        r[lab + '_curve'] = (E['freq'], E['curve'])
    dx = (xc - x) * B.PX; dy = (yc - y) * B.PX
    r['shift_nm'] = dict(rms_x=float(np.sqrt(np.mean(dx ** 2))), rms_y=float(np.sqrt(np.mean(dy ** 2))), max_abs=float(max(np.abs(dx).max(), np.abs(dy).max())))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for lab, c in (('raw', 'C0'), ('cdf_eq', 'C3')):
        fq, cv = r[lab + '_curve']; ax[0].plot(fq, cv, color=c, label=f'{lab}: {r[lab]["frc_nm"]:.2f} nm')
    ax[0].axhline(1 / 7, color='k', lw=0.5); ax[0].set_xlabel('1/nm'); ax[0].legend(); ax[0].set_title(f'{ds} SIMCODE v2 P2 sigma<=12nm FRC', fontsize=9)
    B.hist1d(ax[1], B.sub(x), label='raw x', color='C0'); B.hist1d(ax[1], B.sub(xc), label='eq x', color='C3'); ax[1].legend(); ax[1].set_ylim(0.8, 1.2)
    fs = np.linspace(-0.5, 0.5, 41)
    for b_ in groups:
        pass
    ax[2].hist(dx, 200, range=(-10, 10), histtype='step', label='x'); ax[2].hist(dy, 200, range=(-10, 10), histtype='step', label='y'); ax[2].set_xlabel('correction shift (nm)'); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(B.OUT, f'fig_frc_corr_{ds}.png'), dpi=90); plt.close(fig)
    RES[ds] = {k: v for k, v in r.items() if not k.endswith('_curve')}
    print(ds, json.dumps(RES[ds]), flush=True)
json.dump(RES, open(os.path.join(B.OUT, 'frc_corr.json'), 'w'), indent=1)
