"""SIMCODE v2 on a real movie: the PSF-v2 run vs the previous v2 production (the paper's 12 nm rule, Dataset.select).

Per run: rows, the 12 nm selection (p >= 0.63, max(sx, sy) <= 12 nm, run crop), the fair-FRC set (from the figure's
_values.json: disjoint windows, sigma < 0.15 px, crop, own DME v2 drift), the FRC (fair protocol), the localization
precision distribution of the selection (calibrated sigma: mean(sx, sy) and max(sx, sy), quantiles), photons per 6-frame
window (the previous production's photons are in the old camera units, gain 0.56 e/ADU; converted x 0.4256 / 0.56 = 0.76 to
the v2 units for the comparison), and the emitter height distribution (pass-1 z; only runs with a 3D head).

old values json '-': the fair FRC of the old run is computed here (needs --drift_old as the drift cache, --nframes).
usage: python analysis/paper/v2psf_compare.py <ds> <old run dir> <new run dir> <old values json> <new values json> <out prefix>
       [--calib <calib json of the new run>] [--drift_old npy --drift_new npy --nframes F]
       (with the drift caches of the figures: the fair FRC is recomputed with the crossing linearly interpolated between the
       5-nm-render frequency bins, frc_interp_nm: the protocol's FRC is the first bin below 1/7, steps of ~0.03 nm here)
-> <out prefix>.json, .md (table), _sigma.png (precision / photons / z histograms)"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from simcode.dataset import Dataset

PX = 108.33
GAIN_OLD, GAIN_NEW = 0.56, 0.4256
Q = [5, 10, 25, 50, 75, 90, 95]


def stats(run, label, photon_scale, focus_um=None, pix_um=PX / 1000):
    L = Dataset.load(run)
    S = L.select(12)
    F = S.numpy()
    s = F['crlb'][:, :2].astype(np.float64) * PX
    smean, smax = s.mean(1), s.max(1)
    ph = F['photons'].astype(np.float64)
    r = dict(label=label, run=run, n_rows=len(L), n_sel12=len(S),
             sigma_mean_nm=dict(zip([f'q{q:02d}' for q in Q], np.percentile(smean, Q).round(3).tolist())),
             sigma_max_nm=dict(zip([f'q{q:02d}' for q in Q], np.percentile(smax, Q).round(3).tolist())),
             photons_raw_median=float(np.median(ph)), photon_scale=photon_scale,
             photons_v2units=dict(zip([f'q{q:02d}' for q in Q], np.percentile(ph * photon_scale, Q).round(1).tolist())))
    hist = dict(smax=smax, ph=ph * photon_scale)
    if F['pos'].shape[1] > 2 and focus_um is not None:
        z = focus_um + F['pos'][:, 2].astype(np.float64) * pix_um
        ok = np.isfinite(z)
        r['z_um'] = dict(zip([f'q{q:02d}' for q in Q], np.percentile(z[ok], Q).round(4).tolist()))
        r['z_nan_frac'] = float(1 - ok.mean()); r['z_mad_um'] = float(np.median(np.abs(z[ok] - np.median(z[ok]))))
        r['z_note'] = f'height = focus {focus_um} um + z_px * {pix_um:.5f} um (pass-1 z of the linked pass-1 row)'
        hist['z'] = z[ok]
    return r, hist


def from_values(fn):
    v = json.load(open(fn)); i = len(v['labels']) - 1          # the v2 entry is the last column
    return dict(values=fn, fig_label=v['labels'][i], frc_nm=v['frc_nm'][i], n_frc=v['n_frc'][i], n_disjoint=v['n_disjoint'][i],
                n_all_loaded=v['n_all'][i], frc_nm_nofilter=v['frc_nm_nofilter'][i], frc_nm_30frame_halves=v['frc_nm_30frame_halves'][i],
                pearson=v['pearson'][i], drift_ptp_nm=v['drift_ptp_nm'][i])


def from_fair(run, drift_npy, nframes):
    """no figure values (e.g. the previous 20 nM production had no figure): the fair protocol here (own DME v2 drift, cached at
    drift_npy); no Pearson"""
    sys.path.insert(0, HERE)
    import frc_fair as FF
    T = FF.load('v2', run, v2_sigma_max=12)
    keep = T['f'] + 6 <= nframes
    T = {k: v[keep] for k, v in T.items()}
    R = FF.evaluate(T, nframes, drift_cache=drift_npy)
    return dict(values=None, fig_label='SIMCODE v2', frc_nm=R['frc_nm'], n_frc=R['n_frc'], n_disjoint=R['n_disjoint'],
                n_all_loaded=R['n_all'], frc_nm_nofilter=R['frc_nm_nofilter'], frc_nm_30frame_halves=R['frc_nm_30frame_halves'],
                pearson=float('nan'), drift_ptp_nm=R['drift_ptp_nm'])


def frc_interp(run, drift_npy, nframes, check):
    sys.path.insert(0, HERE)
    import frc_fair as FF
    T = FF.load('v2', run, v2_sigma_max=12)
    keep = T['f'] + 6 <= nframes
    T = {k: v[keep] for k, v in T.items()}
    R = FF.evaluate(T, nframes, drift_cache=drift_npy, secondary=False)
    assert abs(R['frc_nm'] - check) < 1e-9, (R['frc_nm'], check)
    f, c = R['freq'], R['curve']; thr = 1 / 7
    k = 3 + np.nonzero(c[3:] < thr)[0][0]
    fx = f[k - 1] + (c[k - 1] - thr) / (c[k - 1] - c[k]) * (f[k] - f[k - 1])
    return float(1 / fx)


def main():
    ap = argparse.ArgumentParser()
    for k in ('ds', 'old', 'new', 'old_values', 'new_values', 'out'): ap.add_argument(k)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--drift_old', default=None); ap.add_argument('--drift_new', default=None)
    ap.add_argument('--nframes', type=int, default=None)
    a = ap.parse_args()
    focus = None
    if a.calib:
        from simcode.config import resolve
        focus = json.load(open(resolve(a.calib)))['focus_um']
    ro, ho = stats(a.old, 'previous v2 production', GAIN_NEW / GAIN_OLD)
    rn, hn = stats(a.new, 'v2, PSF-v2 models', 1.0, focus_um=focus)
    ro.update(from_values(a.old_values) if a.old_values != '-' else from_fair(a.old, a.drift_old, a.nframes))
    rn.update(from_values(a.new_values))
    if a.drift_old and a.drift_new:
        ro['frc_interp_nm'] = frc_interp(a.old, a.drift_old, a.nframes, ro['frc_nm'])
        rn['frc_interp_nm'] = frc_interp(a.new, a.drift_new, a.nframes, rn['frc_nm'])
    res = dict(dataset=a.ds, old=ro, new=rn, gain_old=GAIN_OLD, gain_new=GAIN_NEW)
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    f1 = lambda x: f'{x:.2f}'
    rows = [('rows (pass 2, all)', f"{ro['n_rows']:,}", f"{rn['n_rows']:,}"),
            ('12 nm selection (p >= 0.63, max sigma <= 12 nm, crop)', f"{ro['n_sel12']:,}", f"{rn['n_sel12']:,}"),
            ('fair-FRC set n_frc (disjoint windows, sigma < 0.15 px, crop)', f"{ro['n_frc']:,}", f"{rn['n_frc']:,}"),
            ('FRC fair protocol [nm]', f1(ro['frc_nm']), f1(rn['frc_nm'])),
            ('  FRC without sigma filter / 30-frame halves [nm]', f"{ro['frc_nm_nofilter']:.2f} / {ro['frc_nm_30frame_halves']:.2f}",
             f"{rn['frc_nm_nofilter']:.2f} / {rn['frc_nm_30frame_halves']:.2f}"),
            ('  FRC, crossing interpolated between bins [nm]', f"{ro.get('frc_interp_nm', np.nan):.2f}", f"{rn.get('frc_interp_nm', np.nan):.2f}"),
            ('Pearson corr. (RSP)', f"{ro['pearson']:.4f}", f"{rn['pearson']:.4f}"),
            ('drift peak-to-peak x / y [nm]', '/'.join(f'{v:.1f}' for v in ro['drift_ptp_nm']), '/'.join(f'{v:.1f}' for v in rn['drift_ptp_nm'])),
            ('median sigma mean(sx, sy) of the selection [nm]', f1(ro['sigma_mean_nm']['q50']), f1(rn['sigma_mean_nm']['q50'])),
            ('  q10 / q25 / q75 / q90 [nm]', ' / '.join(f1(ro['sigma_mean_nm'][k]) for k in ('q10', 'q25', 'q75', 'q90')),
             ' / '.join(f1(rn['sigma_mean_nm'][k]) for k in ('q10', 'q25', 'q75', 'q90'))),
            ('median max(sx, sy) of the selection [nm]', f1(ro['sigma_max_nm']['q50']), f1(rn['sigma_max_nm']['q50'])),
            ('median photons / 6-frame window, v2 units (gain 0.4256)', f"{ro['photons_v2units']['q50']:.0f} (= {ro['photons_raw_median']:.0f} old units x 0.76)",
             f"{rn['photons_v2units']['q50']:.0f}"),
            ('  (photons = network intensity = the PSF model intensity parameter, not detected photons; the', '', ''),
            ('   previous production: the psf3d PSF model, so the two scales differ by the PSF normalisation too)', '', ''),
            ('  photons q25 / q75 (v2 units)', f"{ro['photons_v2units']['q25']:.0f} / {ro['photons_v2units']['q75']:.0f}",
             f"{rn['photons_v2units']['q25']:.0f} / {rn['photons_v2units']['q75']:.0f}")]
    if 'z_um' in rn:
        z = rn['z_um']
        rows.append(('emitter height (pass-1 z) median [nm]', 'n.a. (2D networks)', f"{z['q50'] * 1e3:.0f}"))
        rows.append(('  q25-q75 / q05-q95 [nm]; MAD [nm]', '', f"{z['q25'] * 1e3:.0f}-{z['q75'] * 1e3:.0f} / {z['q05'] * 1e3:.0f}-{z['q95'] * 1e3:.0f}; {rn['z_mad_um'] * 1e3:.0f}"))
        rows.append(('  rows without a pass-1 z', '', f"{rn['z_nan_frac'] * 100:.2f} %"))
    md = [f'# {a.ds}: SIMCODE v2 PSF-v2 run vs the previous v2 production', '',
          f'old: `{a.old}`  \nnew: `{a.new}`', '', '| | previous v2 | v2, PSF-v2 models |', '|---|---|---|']
    md += [f'| {r[0]} | {r[1]} | {r[2]} |' for r in rows]
    open(a.out + '.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = 3 if 'z' in hn else 2
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
    b = np.linspace(0, 12, 61)
    for h, r, c in ((ho, ro, 'grey'), (hn, rn, 'red')):
        ax[0].hist(h['smax'], b, histtype='step', color=c, density=True, label=f"{r['label']} (median {r['sigma_max_nm']['q50']:.2f} nm)")
        ax[1].hist(h['ph'], np.linspace(0, 4000, 81), histtype='step', color=c, density=True, label=f"{r['label']} (median {r['photons_v2units']['q50']:.0f})")
    ax[0].set_xlabel('calibrated max(sx, sy) [nm], 12 nm selection'); ax[0].legend(fontsize=7)
    ax[1].set_xlabel('photons / 6-frame window (v2 units)'); ax[1].legend(fontsize=7)
    if n == 3:
        ax[2].hist(hn['z'] * 1e3, np.linspace(-200, 600, 81), histtype='step', color='red', density=True)
        ax[2].axvline(rn['z_um']['q50'] * 1e3, color='k', ls=':'); ax[2].set_xlabel('emitter height (pass-1 z) [nm]')
        ax[2].set_title(f"median {rn['z_um']['q50'] * 1e3:.0f} nm", fontsize=9)
    for x in ax: x.set_yticks([])
    fig.suptitle(f'{a.ds}: SIMCODE v2 selection (p >= 0.63, max sigma <= 12 nm)', fontsize=10)
    plt.tight_layout(); plt.savefig(a.out + '_sigma.png', dpi=130)


if __name__ == '__main__':
    main()
