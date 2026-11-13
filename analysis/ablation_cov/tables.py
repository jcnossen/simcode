"""Covariance ablation: markdown tables from the result json files -> stdout (pasted into out/ablation_cov/REPORT.md)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import variants as V
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'out', 'ablation_cov')
VS = ['A', 'B', 'H', 'F', 'G', 'C', 'D', 'E1', 'E']
SCN = [('iso', 'isolated (exact model)'), ('pair2', 'pairs 2 px (exact)'), ('pair1', 'pairs 1 px (exact)'), ('rs05qm', 'realistic sparse (bright)'),
       ('rs05pm', 'realistic sparse (paper photons)'), ('rs25qm', 'realistic dense')]


def gate(front):
    R = json.load(open(f'{OUT}/gate_abl_{front}.json'))['variants']
    for b, bl in (('200-2000', '200-2000 photons'), ('2000-4000', '2000-4000 photons')):
        print(f'\n**{ {"s2": "Pass 2", "s1": "Pass 1"}[front]}, {bl}: RMS / CRLB.** A: value [95 % CI]; others: paired difference to A [95 % CI] (FOV bootstrap).\n')
        print('| scene | n | A | ' + ' | '.join(VS[1:]) + ' |'); print('|---|---|---|' + '---|' * (len(VS) - 1))
        for sc, sl in SCN:
            if sc not in R['A'] or b not in R['A'][sc]: continue
            a = R['A'][sc][b]; row = f'| {sl} | {a["n"]} | {a["rms"]:.3f} [{a["ci"][0]:.2f}, {a["ci"][1]:.2f}] |'
            for v in VS[1:]:
                r = R[v][sc][b]; sig = r['d_ci'][0] > 0 or r['d_ci'][1] < 0
                row += f' {"**" if sig else ""}{r["d_rms"]:+.3f}{"**" if sig else ""} [{r["d_ci"][0]:+.3f}, {r["d_ci"][1]:+.3f}] |'
            print(row)
    for key, title, fmt in (('c5', 'gross errors, % of rows with |error| > 5 CRLB', '{:.2f}'), ('zcal', 'calibration of the reported sigma after the per-variant rescale: robust sd of error / sigma (1 = calibrated)', '{:.2f}'),
                            ('rms45_nm', 'RMS error (nm) of the 45 % of rows with the smallest reported sigma (the ranking use; se of A in brackets)', '{:.2f}'),
                            ('spearman', 'Spearman rank correlation of sigma and |error|', '{:.2f}')):
        for b in ('200-2000', '2000-4000'):
            print(f'\n**{ {"s2": "Pass 2", "s1": "Pass 1"}[front]}, {b} photons: {title}.**\n')
            print('| scene | ' + ' | '.join(VS) + ' |'); print('|---|' + '---|' * len(VS))
            for sc, sl in SCN:
                if sc not in R['A'] or b not in R['A'][sc]: continue
                row = f'| {sl} |'
                for v in VS:
                    r = R[v][sc][b]; row += ' ' + fmt.format(r[key]) + (f' (±{r["rms45_se"]:.2f})' if key == 'rms45_nm' and v == 'A' else '') + ' |'
                print(row)
    print(f'\n**{ {"s2": "Pass 2", "s1": "Pass 1"}[front]}, 200-2000 photons: bias (mean error x, y in nm; se of A in brackets).**\n')
    print('| scene | ' + ' | '.join(VS) + ' |'); print('|---|' + '---|' * len(VS))
    for sc, sl in SCN:
        if sc not in R['A']: continue
        row = f'| {sl} |'
        for v in VS:
            r = R[v][sc]['200-2000']; row += f' {r["bias_nm"][0]:+.2f}, {r["bias_nm"][1]:+.2f}' + (f' (±{r["bias_se_nm"][0]:.2f})' if v == 'A' else '') + ' |'
        print(row)
    print(f'\n**{ {"s2": "Pass 2", "s1": "Pass 1"}[front]}: without the position prior (prior sigma x 1000), 200-2000 photons: RMS / CRLB and gross %.**\n')
    print('| scene | A | A without prior | D | D without prior |'); print('|---|---|---|---|---|')
    for sc, sl in SCN:
        if sc not in R['A'] or sc not in R.get('An', {}): continue
        g = lambda v: f'{R[v][sc]["200-2000"]["rms"]:.2f} / {R[v][sc]["200-2000"]["c5"]:.1f} %'
        print(f'| {sl} | {g("A")} | {g("An")} | {g("D")} | {g("Dn")} |')


def paired(Q, QA, key):
    d = np.array([q[key] for q in Q]) - np.array([q[key] for q in QA]); return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def movie(setn):
    R = json.load(open(f'{OUT}/movie_abl_{setn}.json')); A = R['A']
    if setn == 'real':
        print('\n**Real 1809_2 (120000 frames), rows u <= 0.45, fair FRC protocol, own DME v2 drift per variant.** FRC = protocol value (quantised to the frequency bin, step 0.24 nm here); '
              'FRC interp. = the same 1/7 crossing interpolated between bins; quarter difference = mean ± se over 4 movie quarters of (variant − A), interpolated FRC of each quarter.\n')
        print('| variant | FRC nm | FRC interp. nm | quarters: variant − A (nm) | n in FRC | n u <= 0.45 | median sigma nm | median photons |'); print('|---|---|---|---|---|---|---|---|')
        for v in VS:
            if v not in R: continue
            r = R[v]; d = paired(r['quarters'], A['quarters'], 'frc_i_nm')
            print(f'| {V.LABEL[v]} | {r["frc_nm"]:.2f} | {r["frc_i_nm"]:.2f} | {"—" if v == "A" else f"{d[0]:+.2f} ± {d[1]:.2f}"} | {r["n_frc"]} | {r["n_u045"]} | {r["median_sigma_nm"]:.2f} | {r["median_photons"]:.0f} |')
    else:
        print(f'\n**Simulated movie {setn} (papersim, measured PSF, 60000 frames), rows u <= 0.45 (the v2q45 row of analysis/papersim/metrics.py).** Quarter differences = mean ± se over 4 movie quarters of (variant − A).\n')
        print('| variant | FRC nm | FRC interp. nm | quarters ΔFRC interp. | Jaccard (10 nm) | quarters ΔJaccard | RMS vs GT nm | quarters ΔRMS nm | n in FRC | err > 30 nm % | median abs(err)/sigma |'); print('|---|---|---|---|---|---|---|---|---|---|---|')
        for v in VS:
            if v not in R: continue
            r = R[v]; df = paired(r['quarters'], A['quarters'], 'frc_i_nm'); dj = paired(r['quarters'], A['quarters'], 'jaccard'); dr = paired(r['quarters'], A['quarters'], 'rms_nm')
            f = lambda d, p: '—' if v == 'A' else f'{d[0]:+.{p}f} ± {d[1]:.{p}f}'
            gm = r['gt_match']
            print(f'| {V.LABEL[v]} | {r["frc_nm"]:.2f} | {r["frc_i_nm"]:.2f} | {f(df, 2)} | {r["jaccard_end"]:.4f} | {f(dj, 4)} | {r["rms_nm"]:.2f} | {f(dr, 2)} | {r["n_frc"]} | {100 * gm["frac_err_gt30"]:.2f} | {np.mean(gm["median_abs_err_over_sigma"]):.3f} |')


def sigu():
    R = json.load(open(f'{OUT}/movie_abl_m05r3.json'))
    print('\n**Selection-aware sigma rescale g(u) per variant (fit: sim movie m05r3 frames 0-2400) and the held-out check on frames 2400-4800** (robust zcal, 1 = calibrated; RMS vs GT of the rows u <= 0.45 and of the strict rows).\n')
    print('| variant | g(u) first / last bin | zcal strict before | zcal strict after | zcal u <= 0.45 after | zcal per u bin after (min-max) | RMS u <= 0.45 nm | RMS strict nm | RMS all good nm |'); print('|---|---|---|---|---|---|---|---|---|')
    for v in VS:
        if v not in R: continue
        r = R[v]; h = r['heldout']
        print(f'| {V.LABEL[v]} | {r["g"][0]:.2f} / {r["g"][-1]:.2f} | {h["zc_strict_before"]:.3f} | {h["zc_strict"]:.3f} | {h["zc_u45"]:.3f} | {min(h["zc_by_u"]):.2f}-{max(h["zc_by_u"]):.2f} | {h["rms_u45_nm"]:.2f} | {h["rms_strict_nm"]:.2f} | {h["rms_all_nm"]:.2f} |')


pm = lambda t: f' ± {t.std(ddof=1):.3f}' if t.std() > 0 else ''


def autoc():
    R = json.load(open(f'{OUT}/autoc_est.json'))
    print('\n**Automatic estimate theta = (s, c, c_a), simcode2/autoc.py.** Real and pipeline-sim rows: mean ± sd over 3 repeats on disjoint chunk subsets; gate rows: all rows, ± sd over 3 FOV groups where computed (control, per-frame 0.10, model-error-only).\n')
    print('| data | put in (c, c_a) | s | c | c_a | isolated non-kinetic rows | best single scale with c = 0 |'); print('|---|---|---|---|---|---|---|')
    for k, r in R.items():
        if k.startswith('gate:'):
            T = np.array(r['reps']); th = r['theta']
            print(f'| sim gate: {r["desc"]} | {r["truth"][0]:.2f}, {r["truth"][1]:.2f} | {th[0]:.2f} | {th[1]:.3f}{pm(T[:, 1])} | {th[2]:.3f}{pm(T[:, 2])} | {r["n_rows"]} | {r["picture"]["s_only"]:.2f} |')
        else:
            m, s = r['mean'], r['sd']
            print(f'| {k} | {"0, 0" if k.startswith("D1") else "unknown"} | {m[0]:.2f} ± {s[0]:.2f} | {m[1]:.3f} ± {s[1]:.3f} | {m[2]:.3f} ± {s[2]:.3f} | {int(np.mean([q["n_rows"] for q in r["reps"]]))} per repeat | {r["picture"]["s_only"]:.2f} |')


if __name__ == '__main__':
    for w in sys.argv[1:]:
        if w in ('s1', 's2'): gate(w)
        elif w == 'sigu': sigu()
        elif w == 'autoc': autoc()
        else: movie(w)
