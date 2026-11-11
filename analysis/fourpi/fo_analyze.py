"""Analyse frameorder.py output: per file, head (frames 0-239) and tail (last 120 frames) step offset s, match score,
per-angle phase rotation psi_a vs the reference file head, counter continuity, timing; writes a table + figure."""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quickloc import slot_coeffs, refine_k
from frameorder import K0, REF

OUT = '/home/jc/simcode2/out/4pi/smlm/frameorder'


def best_shift(Z, Zr):
    """score(s) = sum_a |sum_j Z[j,a] conj(Zr[(j+s)%6, a])| / norms; returns s, scores, psi_a(best)"""
    sc, psi = [], []
    for s in range(6):
        Zs = np.roll(Zr, -s, 0)
        c = (Z * np.conj(Zs)).sum(0)
        sc.append(np.abs(c).sum() / (np.sqrt((np.abs(Z) ** 2).sum() * (np.abs(Zs) ** 2).sum()) + 1e-12))
        psi.append(np.angle(c))
    s = int(np.argmax(sc))
    return s, np.array(sc), psi[s]


def analyse(cell):
    R = list(np.load(f'{OUT}/fo_{cell}.npy', allow_pickle=True))
    ref = [r for r in R if r['fi'] == REF[cell]][0]
    h = ref['head']
    k0 = refine_k(h['x'], h['y'], h['I'], K0[cell][0], [0, 1, 2]); k1 = refine_k(h['x'], h['y'], h['I'], K0[cell][1], [3, 4, 5])
    Zr, nr = slot_coeffs(h['x'], h['y'], h['I'], k0, k1)
    global PREF
    fm = h['fmean']; rr = (fm[:, 0] + fm[:, 2]) / (fm[:, 1] + fm[:, 3]); rr = rr / rr.mean()
    PREF = rr[:len(rr) // 6 * 6].reshape(-1, 6).mean(0) - 1
    rows = []
    for r in R:
        row = dict(fi=r['fi'], n=r['n'], fc_ok=(r['fc1'] - r['fc0'] == r['n'] - 1), dt_ms=(r['t1'] - r['t0']) / (r['n'] - 1) * 1e3,
                   t0=r['t0'], t1=r['t1'])
        for tag in ('head', 'tail'):
            q = r[tag]
            Z, ne = slot_coeffs(q['x'], q['y'], q['I'], k0, k1)
            s, sc, psi = best_shift(Z, Zr)
            # tail windows start at n - 120 (a multiple of 6), so the expected s is 0 as well
            # background polarization signature: (ch0 + ch2) / (ch1 + ch3) per frame, slot profile vs reference
            fm = q['fmean']; rr = (fm[:, 0] + fm[:, 2]) / (fm[:, 1] + fm[:, 3]); rr = rr / rr.mean()
            prof = rr[:len(rr) // 6 * 6].reshape(-1, 6).mean(0) - 1
            cc = [float(np.dot(prof, np.roll(PREF, -k))) for k in range(6)]
            sa = int(np.argmax(cc))
            row[tag] = dict(s=s, score=sc[s], score2=np.sort(sc)[-2], psi=np.rad2deg(psi), n=ne,
                            amp=np.abs(Z).round(3), s_bg=sa, bg_prof=prof.round(3))
        rows.append(row)
    rows.sort(key=lambda r: r['fi'])
    gaps = [rows[i + 1]['t0'] - rows[i]['t1'] for i in range(len(rows) - 1)]
    lines = [f'# {cell}: k0 = ({k0[0]:.4f}, {k0[1]:.4f}), k1 = ({k1[0]:.4f}, {k1[1]:.4f}) rad/px (refined on file {REF[cell]} head, {nr} emitters)',
             '| file | counter continuous | dt (ms) | gap before (s) | head: s, score (2nd) | head s (bg signature) | head psi0, psi1 (deg) | tail: s, score (2nd) | tail s (bg) | tail psi0, psi1 (deg) | n emitters head/tail |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows):
        g = f'{gaps[i - 1]:.2f}' if i > 0 else ''
        H, T = r['head'], r['tail']
        lines.append(f'| {r["fi"]:03d} | {r["fc_ok"]} | {r["dt_ms"]:.3f} | {g} | {H["s"]}, {H["score"]:.2f} ({H["score2"]:.2f}) | {H["s_bg"]} | '
                     f'{H["psi"][0]:+.0f}, {H["psi"][1]:+.0f} | {T["s"]}, {T["score"]:.2f} ({T["score2"]:.2f}) | {T["s_bg"]} | {T["psi"][0]:+.0f}, {T["psi"][1]:+.0f} | {H["n"]}/{T["n"]} |')
    open(f'{OUT}/frameorder_{cell}.md', 'w').write('\n'.join(lines) + '\n')
    # figure: psi per file head/tail, scores
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fi = np.array([r['fi'] for r in rows])
    for a in range(2):
        ax[0].plot(fi, [r['head']['psi'][a] for r in rows], 'o-', ms=3, label=f'head angle {a}')
        ax[0].plot(fi + 0.5, [r['tail']['psi'][a] for r in rows], 's', ms=3, label=f'tail angle {a}')
    ax[0].set_ylabel('pattern phase vs ref (deg)'); ax[0].legend(fontsize=7)
    ax[1].plot(fi, [r['head']['s'] for r in rows], 'o', label='head s'); ax[1].plot(fi + .3, [r['tail']['s'] for r in rows], 's', label='tail s')
    ax[1].set_ylabel('step offset s'); ax[1].legend(fontsize=7)
    ax[2].plot(fi, [r['head']['score'] for r in rows], 'o', ms=3, label='head best'); ax[2].plot(fi, [r['head']['score2'] for r in rows], 'x', ms=3, label='head 2nd')
    ax[2].plot(fi, [r['tail']['score'] for r in rows], 's', ms=3, label='tail best'); ax[2].plot(fi, [r['tail']['score2'] for r in rows], '+', ms=3, label='tail 2nd')
    ax[2].set_ylabel('match score'); ax[2].set_xlabel('file index'); ax[2].legend(fontsize=7)
    fig.suptitle(f'{cell}: pattern-step sync per file (head = frames 0-239, tail = last 120)')
    fig.savefig(f'/home/jc/simcode2/out/4pi/smlm/fig/frameorder_{cell}.png', dpi=80, bbox_inches='tight')
    print('\n'.join(lines))
    return rows, (k0, k1)


if __name__ == '__main__':
    for c in sys.argv[1:]:
        analyse(c)
