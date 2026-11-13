"""Preprint value vs reproduced (Gaussian 1.3 px PSF movies, baselines as the preprint ran them) vs SIMCODE v2 (net set --netset,
main selection sigma <= 12 nm; u <= 0.45 as the earlier rule) for every number the preprint quotes for its simulation figures (out/paper/INVENTORY.md):
  Fig 2 (density 1): FRC SIMCODE 20.7 / DECODE 28.2 / SIMFLUX 28.4 / SMLM 44.4 nm; Jaccard SIMCODE 0.89 / DECODE 0.68
  SI S2 (density 4): Jaccard SIMCODE 0.76 / DECODE 0.60 / SIMFLUX 0.41 / SMLM 0.37
  SI S8 (moving): mean Jaccard at 10 / 20 / 40 nm (values not transcribed in INVENTORY.md; the reproduced numbers are listed)
Two FRC protocols per row: 'fair' (this project's: disjoint windows, sigma < 0.15 px, alternating 300-frame halves) and
'preprint' (run_pipeline.get_frc: crlb_filter 0.2 px, window starts % 6 == 0, random half split of the rows, zoom 20 = 5.4 nm
pixels, 4-bin smoothing). Jaccard: the preprint's definition at 10 nm over the full 60 000 frames (and at 20 000 frames, the
frame count in 5a4fda9 fig_densities2's S2 code, nf = 20000).
Methods: SMLM / SIMFLUX = Gaussian sigma 1.3 px fits (SIMFLUX with the GT pattern, as the preprint's sims); DECODE = the
preprint's own model (decode-3f-g1.3); SIMCODE (preprint) = v1 as published; SIMCODE v2 = the given net set.
usage: python analysis/papersim/repro_table.py [--netset g13] -> out/papersim/repro_table_<netset>.md / .json"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M
sys.path.insert(0, os.path.join(C.REPO, 'analysis', 'paper')); import frc_fair

PRE = {'fig2_frc': dict(v1=20.7, decode_pre=28.2, sf_gtp=28.4, smlm=44.4), 'fig2_jac': dict(v1=0.89, decode_pre=0.68),
       's2_jac_d4': dict(v1=0.76, decode_pre=0.60, sf_gtp=0.41, smlm=0.37)}
LAB = dict(smlm='SMLM', sf_gtp='SIMFLUX (GT pattern)', sf='SIMFLUX (est. pattern)', decode_pre='DECODE (preprint model)', v1='SIMCODE (preprint, v1)',
           v2q45='(SIMCODE v2, u <= 0.45, earlier rule)', v2sig='SIMCODE v2, σ <= 12 nm', v2='SIMCODE v2 strict')


def frc_preprint(T, nframes, seed=0):
    m = np.isfinite(T['x']) & (np.maximum(T['sx'], T['sy']) < 0.2) & (T['f'] % 6 == 0) & (T['f'] + 6 <= nframes)
    x, y = T['x'][m], T['y'][m]; h = np.random.default_rng(seed).random(m.sum()) < 0.5
    return float(frc_fair.frc_curve(x[h], y[h], x[~h], y[~h], px=C.PX_NM / 20)[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--netset', default='g13'); a = ap.parse_args()
    rows = []; out = {}
    for d, key in ((1, 'fig2'), (4, 's2')):
        run = os.path.join(C.OUT, f'D{d}_gauss', 'full'); R = json.load(open(os.path.join(run, f'metrics_{a.netset}.json')))
        for k in ('smlm', 'sf_gtp', 'sf', 'decode_pre', 'v1', 'v2sig', 'v2q45', 'v2'):
            if k not in R['methods']: continue
            r = R['methods'][k]; T = M.load_method(k, r['path'])
            fp = frc_preprint(T, R['nframes'])
            j20 = r['jaccard'][int(round(20000 / R['nframes'] * len(r['jaccard']))) - 1]
            e = dict(density=d, method=k, frc_fair=r.get('frc_nm'), frc_preprint_protocol=fp, jaccard_60k=r['jaccard_end'], jaccard_20k=j20, n_all=r['n_all'],
                     rms_nm=r['gt_match'].get('rms_nm'), err_gt30=r['gt_match'].get('frac_err_gt30'))
            if d == 1: e.update(preprint_frc=PRE['fig2_frc'].get(k), preprint_jaccard=PRE['fig2_jac'].get(k))
            else: e.update(preprint_jaccard=PRE['s2_jac_d4'].get(k))
            rows.append(e)
    out['density'] = rows
    mv = os.path.join(C.OUT, 'moving_gauss', 'full')
    if os.path.exists(os.path.join(mv, f'metrics_{a.netset}.json')):
        R = json.load(open(os.path.join(mv, f'metrics_{a.netset}.json'))); out['moving'] = {k: R['methods'][k]['jaccard_mean'] for k in R['methods']}
        mm = os.path.join(mv, f'moving_matched_{a.netset}.json')
        if os.path.exists(mm): out['moving_matched'] = json.load(open(mm))['methods']
    json.dump(out, open(os.path.join(C.OUT, f'repro_table_{a.netset}.json'), 'w'), indent=1, default=float)
    f = lambda v, p=1: '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f'{v:.{p}f}' if isinstance(v, float) else str(v))
    L = ['# Preprint vs reproduced (Gaussian 1.3 px PSF) vs SIMCODE v2', '', f'v2 net set: {a.netset}', '',
         '| figure | method | preprint FRC | FRC, preprint protocol | FRC, fair protocol | preprint Jaccard | Jaccard 60k frames | Jaccard 20k frames | RMS vs GT nm | err > 30 nm |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for e in rows:
        L.append(f"| {'Fig 2 (density 1)' if e['density'] == 1 else 'SI S2 (density 4)'} | {LAB[e['method']]} | {f(e.get('preprint_frc'))} | {f(e['frc_preprint_protocol'])} | "
                 f"{f(e['frc_fair'])} | {f(e.get('preprint_jaccard'), 2)} | {f(e['jaccard_60k'], 3)} | {f(e['jaccard_20k'], 3)} | "
                 f"{','.join(f'{v:.1f}' for v in (e['rms_nm'] or []))} | {f(e['err_gt30'], 3)} |")
    if 'moving' in out:
        L += ['', '| SI S8 (moving) | Jaccard 10 / 20 / 40 nm (all rows, preprint definition) | matched count per block |', '|---|---|---|']
        for k, v in out['moving'].items():
            mm = out.get('moving_matched', {}).get(k, {}).get('jaccard_mean')
            L.append(f"| {LAB.get(k, k)} | {' / '.join(f'{x:.3f}' for x in v)} | {' / '.join(f'{x:.3f}' for x in mm) if mm else '-'} |")
        for k in [k for k in out.get('moving_matched', {}) if k.endswith('_w6')]:
            L.append(f"| {k} (window starts every 6 frames) | - | {' / '.join(f'{x:.3f}' for x in out['moving_matched'][k]['jaccard_mean'])} |")
    open(os.path.join(C.OUT, f'repro_table_{a.netset}.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
