"""Analyse timeblocks.py output: per sampled block (first / last NB frames of every file) and angle: SIM period, step
phases, step differences, median depth, coherence; background step signature; population 4Pi phase. Plot vs absolute
frame index. usage: python tb_analyze.py <cell>"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pattern import refine_k, slotZ, emitter_phase, wrap, cmean
from fourpi import chan_phase

OUT = f'{P4}/smlm'
K0 = {'MT': (np.array([3.8388, -0.0238]), np.array([0.0031, 3.8511])), 'NPC': (np.array([3.8262, -0.0279]), np.array([0.0131, 3.8448]))}
FIRST = {'MT': 0, 'NPC': 2}


def run(cell):
    G = list(np.load(f'{OUT}/timeblocks/tb_{cell}.npy', allow_pickle=True))
    blocks = {}
    for meta, tag, c0, d in G:
        b = blocks.setdefault((meta['fi'], tag), dict(meta=meta, parts=[]))
        b['parts'].append((c0, d))
    fz = json.load(open(f'{OUT}/diag/fourpi_{cell}.json'))
    phc = np.array(fz['bead_r3']['phc']); tau = np.array(fz['tau_movie'])
    # reference background step profile from the first block
    rows = []
    for (fi, tag), b in sorted(blocks.items()):
        parts = sorted(b['parts'], key=lambda p: p[0])
        c0 = parts[0][0]
        x = np.concatenate([p[1]['x'] for p in parts]); y = np.concatenate([p[1]['y'] for p in parts])
        I = np.concatenate([p[1]['I'] for p in parts if len(p[1]['x'])]); fm = np.concatenate([p[1]['fmean'] for p in parts])
        It = I.sum(2); m = It.mean(1); ok = (m > 150) & (It.min(1) > 0.25 * m)
        X = np.c_[x, y][ok]
        r = dict(fi=int(fi), tag=tag, frame_abs=int((fi - FIRST[cell]) * 18000 + c0), n=int(ok.sum()))
        rr = (fm[:, 0] + fm[:, 2]) / (fm[:, 1] + fm[:, 3]); rr = rr / rr.mean()
        r['bg_prof'] = (rr[:len(rr) // 6 * 6].reshape(-1, 6).mean(0) - 1).round(3).tolist()
        for a in range(2):
            I3 = It[ok][:, 3 * a:3 * a + 3]
            k = refine_k(I3, X, K0[cell][a], span=0.03, n=21, rounds=2)
            Z = slotZ(I3, X, k); psi = -np.angle(Z)
            ph, mm, _ = emitter_phase(I3, psi); d_ = wrap(ph - X @ k)
            # the OTHER angle's k evaluated on these frames (an angle swap would show here)
            Zx = slotZ(I3, X, K0[cell][1 - a])
            r[f'a{a}'] = dict(k=k.tolist(), period=float(2 * np.pi / np.linalg.norm(k)), psi=np.rad2deg(psi).tolist(),
                              dpsi=np.rad2deg(wrap(np.diff(psi))).tolist(), slotZ=np.abs(Z).round(3).tolist(),
                              slotZ_other_k=np.abs(Zx).round(3).tolist(), m_med=float(np.nanmedian(mm)), R=float(abs(cmean(d_))))
        Phi, V, Q = chan_phase(I[ok].sum(1), tau, phc)
        c = cmean(Phi[V > 0.1]); r['phi4pi'] = float(np.rad2deg(np.angle(c))); r['R4pi'] = float(abs(c))
        rows.append(r)
    json.dump(rows, open(f'{OUT}/timeblocks/tb_{cell}.json', 'w'))
    fa = np.array([r['frame_abs'] for r in rows])
    fig, ax = plt.subplots(6, 1, figsize=(14, 17), sharex=True)
    for a in range(2):
        for j in range(3):
            ax[0].plot(fa, [r[f'a{a}']['psi'][j] for r in rows], 'o' if a == 0 else 's', ms=3, label=f'angle {a} step {j}')
        ax[1].plot(fa, [r[f'a{a}']['dpsi'][0] for r in rows], 'o' if a == 0 else 's', ms=3, label=f'angle {a}: psi1-psi0')
        ax[1].plot(fa, [r[f'a{a}']['dpsi'][1] for r in rows], 'x' if a == 0 else '+', ms=4, label=f'angle {a}: psi2-psi1')
        ax[2].plot(fa, [r[f'a{a}']['m_med'] for r in rows], 'o-', ms=3, label=f'angle {a} median depth')
        ax[2].plot(fa, [r[f'a{a}']['R'] for r in rows], 's--', ms=3, label=f'angle {a} coherence R')
        ax[3].plot(fa, [r[f'a{a}']['period'] for r in rows], 'o-', ms=3, label=f'angle {a}')
    ax[4].plot(fa, [r['phi4pi'] for r in rows], 'o', ms=3, label='population 4Pi phase (deg)'); ax5 = ax[4].twinx(); ax5.plot(fa, [r['R4pi'] for r in rows], 'r.', ms=3)
    ax[5].plot(fa, [np.mean(r['bg_prof'][:3]) - np.mean(r['bg_prof'][3:]) for r in rows], 'o', ms=3, label='bg signature: slots 0-2 minus 3-5')
    ax[0].set_ylabel('step phase psi (deg)'); ax[1].set_ylabel('step difference (deg)'); ax[2].set_ylabel('depth / R'); ax[3].set_ylabel('period (px)')
    ax[4].set_ylabel('4Pi phase (deg); R red'); ax[5].set_ylabel('(ch0+ch2)/(ch1+ch3) contrast')
    for a_ in ax:
        a_.axvline(100000, c='k', ls='--'); a_.legend(fontsize=7, ncol=3)
        for p in range(1, 8): a_.axvline(27777 * p, c='grey', ls=':', lw=0.8)
    ax[-1].set_xlabel('absolute frame index (dashed: 100k; dotted: summed-TIFF part boundaries)')
    fig.suptitle(f'{cell}: pattern and 4Pi phase vs time, sampled blocks (first/last frames of every file), raw dcimg data')
    fig.savefig(f'{OUT}/fig/timeblocks_{cell}.png', dpi=70, bbox_inches='tight')
    for r in rows[:16]:
        print(r['fi'], r['tag'], r['frame_abs'], r['n'], 'a0 psi', np.round(r['a0']['psi']), 'dpsi', np.round(r['a0']['dpsi']), 'm', round(r['a0']['m_med'], 2), 'R', round(r['a0']['R'], 2),
              '| a1 psi', np.round(r['a1']['psi']), 'm', round(r['a1']['m_med'], 2), 'R', round(r['a1']['R'], 2), '| 4pi', round(r['phi4pi']), round(r['R4pi'], 2))
    return rows


if __name__ == '__main__':
    run(sys.argv[1])
