"""Step 3b: empirical per-channel PSFs. Isolated, on-through-the-window, bright emitters, binned by PSF-fit z and by
4Pi phase Phi (4 bins of 90 deg); per channel the 6-frame-summed, background-subtracted, UNWARPED ROI is accumulated on
a 2x oversampled grid centred on the emitter (sub-pixel alignment by binning pixel-centre offsets), normalised by the
emitter's total (4-channel) amplitude. Also the bead z-stack channel images. usage: python chpsf.py <cell>"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register import PolyMap

OUT = f'{P4}/smlm'
OS = 2; HW = 6.5


def run(cell):
    em = np.load(f'{OUT}/diag/fourpi_{cell}_emitters.npz')
    key = {(int(f), round(float(x), 3), round(float(y), 3)) for f, x, y in zip(em['frame'], em['x'], em['y'])}
    pm = PolyMap.load(f'{OUT}/reg/reg_insitu_{cell}.json')
    zq = np.quantile(em['z'], [0.05, 0.25, 0.5, 0.75, 0.95])
    zedges = np.array([zq[0], (zq[0] + zq[1]) / 2, zq[2] - 0.05, zq[2] + 0.05, (zq[3] + zq[4]) / 2, zq[4]])
    nzb, npb = len(zedges) - 1, 4
    nb = int(2 * HW * OS)
    acc = np.zeros((nzb, npb, 4, nb, nb)); cnt = np.zeros((nzb, npb, 4, nb, nb)); nem = np.zeros((nzb, npb), int)
    lut = {(int(f), round(float(x), 3), round(float(y), 3)): i for i, (f, x, y) in enumerate(zip(em['frame'], em['x'], em['y']))}
    fs = sorted(glob.glob(f'{OUT}/p1/p2_{cell}_*.npz'))
    for k, f2 in enumerate(fs):
        fi = int(f2[-7:-4])
        P2 = np.load(f2); P1 = np.load(f'{OUT}/p1/p1_{cell}_{fi:03d}.npz')
        frame = k * 18000 + P2['w'] * 6 + 3
        idx = [(j, lut.get((int(fr), round(float(x), 3), round(float(y), 3)))) for j, (fr, x, y) in enumerate(zip(frame, P2['x'], P2['y']))]
        idx = [(j, i) for j, i in idx if i is not None and 0.3 < em['V'][i] and em['N'][i] > 2000]
        if not idx: continue
        inv = np.full(len(P2['w']), -1); inv[P2['aidx']] = np.arange(len(P2['aidx']))   # p2 row -> amplitude row
        idx = [(j, i) for j, i in idx if inv[j] >= 0]
        J = np.array([j for j, _ in idx]); Ie = np.array([i for _, i in idx]); Ja = inv[J]; J1 = P2['idx'][J]   # p1 rows
        roi = P1['roi_ch'][J1].astype(np.float32).sum(1)                # [n,4,11,11]
        bg = P2['B'][Ja].sum(1)                                          # [n,4]
        Atot = P2['A'][Ja].sum((1, 2))
        corner = P1['corner'][J1]
        zb = np.clip(np.searchsorted(zedges, em['z'][Ie]) - 1, 0, nzb - 1)
        ok = (em['z'][Ie] >= zedges[0]) & (em['z'][Ie] <= zedges[-1])
        pb = (np.floor(((em['Phi'][Ie] + np.pi) % (2 * np.pi)) / (np.pi / 2)).astype(int)) % 4
        o = np.arange(11) - 5
        for n in np.nonzero(ok)[0]:
            nem[zb[n], pb[n]] += 1
            for c in range(4):
                xc, yc = pm.inv(c, float(P2['x'][J[n]]), float(P2['y'][J[n]]))
                rx = xc - (corner[n, c, 0] + 5); ry = yc - (corner[n, c, 1] + 5)
                du = o[None, :] - rx; dv = o[:, None] - ry
                bu = np.floor((du + HW) * OS).astype(int); bv = np.floor((dv + HW) * OS).astype(int)
                bu = np.broadcast_to(bu, (11, 11)); bv = np.broadcast_to(bv, (11, 11))
                m = (bu >= 0) & (bu < nb) & (bv >= 0) & (bv < nb)
                np.add.at(acc[zb[n], pb[n], c], (bv[m], bu[m]), ((roi[n, c] - bg[n, c]) / Atot[n])[m])
                np.add.at(cnt[zb[n], pb[n], c], (bv[m], bu[m]), 1)
    psf = acc / np.maximum(cnt, 1)
    np.savez(f'{OUT}/diag/chpsf_{cell}.npz', psf=psf, cnt=cnt, nem=nem, zedges=zedges)
    # shape metrics: per channel second moments (widths) and centroid within r = 4 px; pair contrast map
    u = (np.arange(nb) + 0.5) / OS - HW
    U, Vv = np.meshgrid(u, u); rr = np.hypot(U, Vv); w = rr <= 4
    met = []
    for i in range(nzb):
        for p in range(npb):
            row = dict(z=[float(zedges[i]), float(zedges[i + 1])], phi_bin=p, n=int(nem[i, p]))
            for c in range(4):
                g = np.clip(psf[i, p, c], 0, None) * w; s = g.sum()
                if s <= 0: continue
                cx = (g * U).sum() / s; cy = (g * Vv).sum() / s
                row[f'ch{c}'] = dict(frac=float(psf[i, p, c][w].sum() / OS ** 2), cx=float(cx), cy=float(cy),
                                     sx=float(np.sqrt((g * (U - cx) ** 2).sum() / s)), sy=float(np.sqrt((g * (Vv - cy) ** 2).sum() / s)))
            met.append(row)
    json.dump(met, open(f'{OUT}/diag/chpsf_{cell}.json', 'w'), indent=0)
    # figure: rows = z bins, columns = channel x phi bin 0 and 2 ; plus contrast (ch0-ch2)/(ch0+ch2)
    fig, ax = plt.subplots(nzb, 10, figsize=(20, 2.1 * nzb))
    for i in range(nzb):
        for c in range(4):
            for jj, p in enumerate((0, 2)):
                a_ = ax[i, 2 * c + jj]; a_.imshow(psf[i, p, c], cmap='inferno', vmin=0); a_.set_xticks([]); a_.set_yticks([])
                if i == 0: a_.set_title(f'ch{c} Phi bin {p}', fontsize=8)
        for jj, (c1, c2) in enumerate(((0, 2), (1, 3))):
            s_ = psf[i, :, c1].mean(0) + psf[i, :, c2].mean(0)
            con = (psf[i, 0, c1] - psf[i, 0, c2]) / np.maximum(psf[i, 0, c1] + psf[i, 0, c2], 1e-3 * s_.max())
            con[s_ < 0.05 * s_.max()] = 0
            a_ = ax[i, 8 + jj]; a_.imshow(con, cmap='RdBu', vmin=-1, vmax=1); a_.set_xticks([]); a_.set_yticks([])
            if i == 0: a_.set_title(f'contrast ch{c1}-ch{c2}, Phi bin 0', fontsize=8)
        ax[i, 0].set_ylabel(f'z {zedges[i]:+.2f}..{zedges[i+1]:+.2f}\nn={nem[i].sum()}', fontsize=8)
    fig.suptitle(f'{cell}: empirical per-channel PSFs (unwarped channel images, 0.5 px bins, normalised to total)')
    fig.savefig(f'{OUT}/fig/chpsf_{cell}.png', dpi=70, bbox_inches='tight')
    print(cell, 'emitters per (z, Phi) bin:', nem.tolist())
    return met


if __name__ == '__main__':
    run(sys.argv[1])
