"""SI figure si_dme_drift: each method's OWN DME v2 drift on one movie (default 1809_2) under the fair protocol of
analysis/paper/frc_fair.py (README.md in this directory): x and y vs frame, one trace per method (SMLM, SIMFLUX, DECODE,
SIMCODE v2), each with its mean removed and a small vertical offset for clarity, in the preprint's 2-panel layout
('Estimated Drift', X [nm] / Y [nm] vs Frame). --style extended (-> si_dme_drift_extended.*) adds the difference of each
baseline's drift to the SIMCODE v2 drift (x: C, y: D; no offsets).

Drift files = the frc_fair drift caches (<prefix>_<kind>_drift.npy, written by frc_fair.evaluate via fig_exp_new.py):
float32 [F, 2] = (x, y) in px (sensor frame), indexed by time = window start + 3 (a localization of window start w is
corrected by drift[w + 3]); F = the movie's frame count. Rows that entered each estimate: that method's rows after protocol
steps 1-4 (disjoint 6-frame windows, max(sigma_x, sigma_y) < 0.15 px with the method's own sigma, crop x, y in [5, 115) px),
crlb = its sigma; DME v2 = simcode.util.drift.dme_v2 (dme.dme_prior.dme_prior_estimate: temporal resolution chosen from the
data, learned temporal prior). The values json of the main figure (--values) supplies the DME info logged at estimation time.

Numbers (caption txt / values json): per method and axis the peak-to-peak range (nm); pairwise rms differences between the
methods' drift traces after removing each trace's mean (independent estimates from different localization tables of the same
sample, so the rms difference bounds the estimation error of the pair), also after a 2000-frame running mean (slow part) and
the remainder (fast part).

  PYTHONPATH=. python analysis/paper/si_dme_drift.py [--smlm f] [--sf f] [--decode f] [--v2 f] [--outdir d] [--offset_nm 15]

--dme fixed (2026-11-15): instead of the figure's drift caches (each method's DME v2 with its OWN data-chosen temporal resolution),
estimate the drift here for every method with IDENTICAL DME settings, logged in the values json (dme_settings): DME v2 =
dme.dme_grid.dme_grid_estimate at --fpb frames/bin (estimatePrecision False) + the learned-prior refinement dme.dme_prior.refine
with --seg-frame segments and kernel scales --scales (the two stages dme_prior_estimate runs, with the choice fixed instead of
cross-validated); sigma = the mean of the method's own per-localization sigma of its rows. Rows = frc_fair protocol steps 1-4 of
the tables given with --tables (kind:path per method; SIMCODE v2 = the paper selection at --v2_sigma_max nm), frames f + 6 <= F.
--auto_log 1 also logs each method's own cross-validated choice (dme_prior_estimate, same rows). Drifts are cached as
<outdir>/<name>_<k>_drift.npy (same layout as the frc_fair caches).
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python analysis/paper/si_dme_drift.py --dme fixed --tables smlm=smlm:<npy>,sf=sf:<hdf5>,decode=decode:<npz>,v2=v2:<run>
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common  # noqa: F401  (paper matplotlib settings)
import matplotlib.pyplot as plt

PX = 108.33
D = '/home/jc/simcode2/out/paper_v2psf/1809_2_fig3_psfv2'
OUT = '/home/jc/simcode2/out/paper_figs_v2'
METHODS = [('smlm', 'SMLM', 'blue'), ('sf', 'SIMFLUX', 'green'), ('decode', 'DECODE', 'orange'), ('v2', 'SIMCODE v2', 'red')]


def runmean(v, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(v, (n // 2, n - 1 - n // 2), mode='edge'), k, mode='valid')


def fixed_dme(tables, nframes, fpb, seg, scales, v2_sigma_max, outbase, auto_log=False, redo=False):
    """identical-setting DME v2 for every method (module doc) -> drift caches <outbase>_<k>_drift.npy, info per method"""
    import time
    import frc_fair as FF
    from simcode.util import drift as DR
    DR.dme_package()
    from dme.dme_grid import dme_grid_estimate
    from dme.dme_prior import refine, dme_prior_estimate
    from dme.dme_gpu import default_device
    dev = default_device(None)
    info = {}
    for k, (kind, path) in tables.items():
        cache = f'{outbase}_{k}_drift.npy'
        T = FF.load(kind, path, **(dict(v2_sigma_max=v2_sigma_max) if kind == 'v2' else {}))
        keep = T['f'] + 6 <= nframes; T = {kk: v[keep] for kk, v in T.items()}
        _, filt = FF.masks(T)
        pos, fr, crlb = DR.drift_rows(T['x'][filt], T['y'][filt], T['f'][filt] + 3, T['sx'][filt], T['sy'][filt], nframes, t_offset=0)
        sig = crlb.mean(0).astype(np.float32)
        I = dict(kind=kind, table=os.path.abspath(path), n_rows=int(len(pos)), sigma_mean_px=sig.tolist(), cache=cache)
        t0 = time.time()
        if os.path.exists(cache) and not redo:
            I['cached'] = True
        else:
            d = dme_grid_estimate(pos, fr.astype(np.int32), sig, fpb, estimatePrecision=False, numframes=nframes, device=dev, verbose=False)
            d = refine(pos, fr.astype(np.int32), sig, nframes, np.asarray(d, np.float32), seg, tuple(scales), dev)
            np.save(cache, np.asarray(d, np.float32)[:nframes, :2]); I['s'] = round(time.time() - t0, 1)
        if auto_log:
            _, ai = dme_prior_estimate(pos, fr.astype(np.int32), crlb, numframes=nframes, device=dev, verbose=False, return_info=True)
            I['auto_choice'] = dict(kind='smooth' if ai['segment_frames'] else 'dme', start_framesperbin=ai['start_framesperbin'],
                                    segment_frames=ai['segment_frames'])
        info[k] = I; print(k, json.dumps(I), flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k, lab, _ in METHODS:
        ap.add_argument(f'--{k}', default=f'{D}_{k}_drift.npy', help=f'{lab} drift .npy ([F, 2] px, t = window start + 3)')
    ap.add_argument('--values', default=f'{D}_values.json', help="main figure's values json (DME info, tables); optional")
    ap.add_argument('--outdir', default=OUT)
    ap.add_argument('--style', default='preprint', choices=['preprint', 'extended'],
                    help="preprint: the preprint's 2-panel layout (X, Y); extended: + difference-to-SIMCODE-v2 panels")
    ap.add_argument('--name', default=None, help='output base name (default si_dme_drift / si_dme_drift_extended)')
    ap.add_argument('--offset_nm', type=float, default=15.0, help='vertical offset between successive traces (nm)')
    ap.add_argument('--slow', type=int, default=2000, help='running-mean window (frames) for the slow / fast split')
    ap.add_argument('--dme', default='cache', choices=['cache', 'fixed'], help='cache: the figure drift caches; fixed: module doc')
    ap.add_argument('--tables', default=None, help='--dme fixed: k=kind:path,... for k in smlm, sf, decode, v2')
    ap.add_argument('--nframes', type=int, default=120000)
    ap.add_argument('--fpb', type=int, default=2000, help='--dme fixed: frames per bin of the spline start')
    ap.add_argument('--seg', type=int, default=96, help='--dme fixed: segment length (frames) of the learned-prior refinement')
    ap.add_argument('--scales', default='4,2,1,1,1', help='--dme fixed: kernel scales of the refinement (dme_prior default)')
    ap.add_argument('--v2_sigma_max', type=float, default=12.0, help='--dme fixed: SIMCODE v2 rows = paper selection at this nm')
    ap.add_argument('--auto_log', type=int, default=1, help="--dme fixed: also log each method's own cross-validated choice")
    ap.add_argument('--redo', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.name is None: a.name = 'si_dme_drift' if a.style == 'preprint' else 'si_dme_drift_extended'
    fixed_info = None
    if a.dme == 'fixed':
        tables = {}
        for spec in a.tables.split(','):
            k, kp = spec.split('=', 1); kind, path = kp.split(':', 1); tables[k] = (kind, path)
        assert set(tables) == {k for k, _, _ in METHODS}, tables
        sc = [int(v) for v in a.scales.split(',')]
        fixed_info = dict(settings=dict(estimator='DME v2, fixed choice: dme_grid_estimate(framesperbin=fpb, estimatePrecision=False) '
                                                  '+ dme_prior.refine(segment L, scales); sigma = mean per-localization sigma of the rows',
                                        fpb=a.fpb, segment_frames=a.seg, scales=sc, nframes=a.nframes, t='window start + 3',
                                        rows='frc_fair steps 1-4: disjoint 6-frame windows, max(sigma) < 0.15 px, crop [5, 115) px; f + 6 <= F',
                                        v2_selection_sigma_max_nm=a.v2_sigma_max),
                          methods=fixed_info)
        fixed_info['methods'] = fixed_dme(tables, a.nframes, a.fpb, a.seg, sc, a.v2_sigma_max, f'{a.outdir}/{a.name}', a.auto_log, a.redo)
        try:
            from simcode.util import drift as DR; fixed_info['settings']['dme_version'] = DR.version()
        except Exception:
            pass
        for k, _, _ in METHODS: setattr(a, k, fixed_info['methods'][k]['cache'])
        a.values = None

    dr = {}; files = {}
    for k, lab, _ in METHODS:
        f = getattr(a, k); files[lab] = dict(path=os.path.abspath(f), mtime=os.path.getmtime(f))
        dr[k] = np.load(f).astype(np.float64)[:, :2] * PX
    F = min(len(v) for v in dr.values())
    dr = {k: v[:F] - v[:F].mean(0) for k, v in dr.items()}
    t = np.arange(F)

    nm = len(METHODS)
    if a.style == 'preprint':
        # the preprint's layout: 'Estimated Drift', X [nm] / Y [nm] vs Frame, legend in the lower panel
        fig, axs = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw=dict(hspace=0.12))
        for ai, ax in enumerate(axs):
            for i, (k, lab, c) in enumerate(METHODS):
                off = (nm - 1 - i) * a.offset_nm
                ax.plot(t, dr[k][:, ai] + off, color=c, lw=0.8, label=f'{lab} (+{off:.0f} nm)' if off else lab)
            ax.set_ylabel(f'{"XY"[ai]} [nm]')
        axs[0].set_title('Estimated Drift')
        axs[1].legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
        axs[1].set_xlabel('Frame'); axs[1].set_xlim(0, F)
    else:
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                                gridspec_kw=dict(height_ratios=[1, 1, 0.55, 0.55], hspace=0.14))
        for ai, ax in enumerate(axs[:2]):
            for i, (k, lab, c) in enumerate(METHODS):
                off = (nm - 1 - i) * a.offset_nm
                ax.plot(t, dr[k][:, ai] + off, color=c, lw=0.9, label=f'{lab} (+{off:.0f} nm)' if off else lab)
            ax.set_ylabel(f'{"xy"[ai]} drift (nm)')
        axs[0].legend(loc='upper center', fontsize=11, ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.17), handlelength=1.5,
                      columnspacing=1.0)
        for ai, ax in enumerate(axs[2:]):
            for k, lab, c in (METHODS[2], METHODS[0], METHODS[1]):        # DECODE first (densest), SIMFLUX on top
                ax.plot(t, dr[k][:, ai] - dr['v2'][:, ai], color=c, lw=0.7, alpha=0.85)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_ylabel(f'$\\Delta${"xy"[ai]} to\nSIMCODE v2 (nm)', fontsize=13)
        axs[3].set_xlabel('frame'); axs[3].set_xlim(0, F)
        for i, ax in enumerate(axs):
            ax.text(-0.17, 1.02, 'ABCD'[i], transform=ax.transAxes, fontsize=20, weight='bold', va='top')
    fig.savefig(f'{a.outdir}/{a.name}.pdf', bbox_inches='tight'); fig.savefig(f'{a.outdir}/{a.name}.svg', bbox_inches='tight')
    fig.savefig(f'{a.outdir}/{a.name}.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # numbers
    labs = {k: lab for k, lab, _ in METHODS}
    V = dict(files=files, nframes=F, offset_nm=a.offset_nm, units='nm (px x 108.33)',
             ptp_nm={labs[k]: np.ptp(v, 0).round(1).tolist() for k, v in dr.items()},
             range_nm={labs[k]: [v.min(0).round(1).tolist(), v.max(0).round(1).tolist()] for k, v in dr.items()}, pair_rms_nm={})
    ks = [k for k, _, _ in METHODS]
    for i in range(nm):
        for j in range(i + 1, nm):
            d = dr[ks[i]] - dr[ks[j]]; d = d - d.mean(0)
            slow = np.stack([runmean(d[:, c], a.slow) for c in (0, 1)], 1)
            V['pair_rms_nm'][f'{labs[ks[i]]} - {labs[ks[j]]}'] = dict(
                all=np.sqrt((d ** 2).mean(0)).round(2).tolist(), slow=np.sqrt((slow ** 2).mean(0)).round(2).tolist(),
                fast=np.sqrt(((d - slow) ** 2).mean(0)).round(2).tolist())
    if a.values and os.path.exists(a.values):
        vj = json.load(open(a.values))
        V['main_figure_values'] = a.values
        V['tables'] = vj.get('paths'); V['dme_info'] = dict(zip(vj['labels'], vj['drift_info']))
        V['n_frc_rows'] = dict(zip(vj['labels'], vj['n_frc'])); V['drift_reused_from'] = vj.get('drift_reused_from')
    if fixed_info is not None: V['dme_fixed'] = fixed_info
    json.dump(V, open(f'{a.outdir}/{a.name}_values.json', 'w'), indent=1, default=float)
    open(f'{a.outdir}/{a.name}_caption.txt', 'w').write(caption(V, a))
    print(json.dumps(V, indent=1, default=float))


def caption(V, a):
    import time
    P = V['pair_rms_nm']; R = V['ptp_nm']
    pr = lambda k: f'{P[k]["all"][0]:.1f} / {P[k]["all"][1]:.1f} nm'
    L = []
    L.append('DRAFT CAPTION')
    L.append('Estimated sample drift of the low-density DNA-PAINT movie of main-text Fig. 3 (1809/cover1/2, 120 000 frames), '
             "estimated separately for every method on its own localizations with DME v2 (Drift at Minimum Entropy, "
             + ('github.com/jcnossen/drift-estimation) with identical settings for all methods ('
                f'{V["dme_fixed"]["settings"]["fpb"]}-frame spline bins refined with the learned temporal prior on '
                f'{V["dme_fixed"]["settings"]["segment_frames"]}-frame segments), ' if 'dme_fixed' in V else
                'github.com/jcnossen/drift-estimation; temporal resolution chosen from the data with a learned temporal prior), ')
             + 'on the localizations of the fair comparison of Fig. 3: disjoint 6-frame windows, max(sigma_x, sigma_y) < 0.15 px (16.2 nm) '
             "with each method's own uncertainty, x, y in [5, 115) px, time = window start + 3. "
             + (f'X (top) and Y (bottom) drift' if a.style == 'preprint' else '(A) x and (B) y drift') +
             f' for SMLM, SIMFLUX, DECODE and SIMCODE v2; each trace has its mean removed and is offset '
             f'by {a.offset_nm:.0f} nm from the next for clarity.' + ('' if a.style == 'preprint' else
             ' (C, D) Difference of the SMLM, SIMFLUX and DECODE drift to the SIMCODE v2 drift (x, y; no offset).') +
             f' The four independent estimates agree: rms difference (x / y) to SIMCODE v2 '
             f'{pr("SMLM - SIMCODE v2")} (SMLM), {pr("DECODE - SIMCODE v2")} (DECODE) and {pr("SIMFLUX - SIMCODE v2")} '
             f'(SIMFLUX, which has the fewest localizations), against a drift range of '
             f'{R["SIMCODE v2"][0]:.0f} nm (x) and {R["SIMCODE v2"][1]:.0f} nm (y).')
    L.append('')
    L.append('NUMBERS (si_dme_drift_values.json)')
    L.append('Peak-to-peak drift (x, y) nm: ' + '; '.join(f'{k} {v[0]:.1f}, {v[1]:.1f}' for k, v in R.items()))
    L.append(f'Pairwise rms difference after removing the mean (x / y nm), all / slow ({a.slow}-frame running mean) / fast (remainder):')
    for k, v in P.items():
        L.append(f'  {k}: {v["all"][0]:.2f} / {v["all"][1]:.2f}; slow {v["slow"][0]:.2f} / {v["slow"][1]:.2f}; fast {v["fast"][0]:.2f} / {v["fast"][1]:.2f}')
    L.append('  (the difference of two independent estimates; its rms is ~sqrt(e1^2 + e2^2) of the two estimation errors, plus any '
             'genuine difference from the different localization sets, e.g. position-dependent bias.)')
    if 'dme_fixed' in V:
        S = V['dme_fixed']['settings']
        L.append('DME SETTINGS (identical for every method; values json dme_fixed): ' + json.dumps(S))
        for k, I in V['dme_fixed']['methods'].items():
            L.append(f'  {k}: table {I["table"]}, rows {I["n_rows"]}, mean sigma {I["sigma_mean_px"][0]:.4f} / {I["sigma_mean_px"][1]:.4f} px'
                     + (f'; its own cross-validated DME v2 choice would be {I["auto_choice"]}' if 'auto_choice' in I else ''))
    if 'n_frc_rows' in V:
        L.append('Rows entering each estimate (= the Fig. 3 FRC set, protocol steps 1-4): ' +
                 ', '.join(f'{k} {v}' for k, v in V['n_frc_rows'].items()))
        L.append('DME info logged in the main figure values json: ' + json.dumps(V['dme_info']))
        if V['n_frc_rows'].get('SMLM') == 554544 and V['n_frc_rows'].get('SIMFLUX') == 213936:
            L.append('  SMLM / SIMFLUX: the same rows (same counts) gave DME v2 choice (smooth, 2000 frames/bin start, segments 96 / 24) in '
                 'out/paper2/fig/1809_2_fig3_q045_provisional_values.json; DME v2.0.0 (external/drift-estimation git 374300f).')
        L.append(f'Tables: {json.dumps(V.get("tables"))}; drift caches originally reused from {V.get("drift_reused_from")}')
    L.append('')
    L.append('FILES USED')
    for k, v in V['files'].items():
        L.append(f'  {k}: {v["path"]} (mtime {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["mtime"]))})')
    L.append('  Array layout: float32 [120000, 2] = (x, y) px, index = time (window start + 3); plotted x 108.33 nm/px.')
    L.append('  NB DECODE: the DECODE table is decode/locs_fixed.npz (frame-label fix: rows shift by one 6-frame window); '
             'rerun this script after any DECODE drift recompute (--decode <npy>).')
    L.append('')
    L.append('CODE: analysis/paper/si_dme_drift.py (this figure); analysis/paper/frc_fair.py (protocol, evaluate -> drift cache); '
             'simcode/util/drift.py (dme_v2, drift_rows); analysis/paper/fig_exp_new.py (writes the caches, --reuse_drift).')
    L.append('Rerun: cd <repo>; PYTHONPATH=. python analysis/paper/si_dme_drift.py [--smlm/--sf/--decode/--v2 <npy>] [--outdir d]')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
