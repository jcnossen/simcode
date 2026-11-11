"""Step 5a: drift correction (DME v2 through analysis/baselines/dme_drift.py on GPU 0 when idle, else a CPU
cross-correlation (RCC-like) fallback), renders, NPC ring analysis. usage: python render_run.py <cell> [px_nm]"""
import os, sys, json, subprocess
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import load_locs, run_dme, render, OUT


def gpu0_idle():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.strip().splitlines()
        u, m = [float(v) for v in q[0].split(',')[1:]]
        return u < 10 and m < 8000
    except Exception:
        return False


def rcc(x, y, frame, nb=12, px_nm=128.0, sr=10.0):
    """CPU fallback: cross-correlate each time bin's render with the total render (sub-pixel by parabola), linear interp."""
    F = frame.max() + 1; edges = np.linspace(0, F, nb + 1)
    b = (x.min(), x.max(), y.min(), y.max())
    tot = render(x, y, px_nm, sr, 20, bounds=b)
    sh = []
    for i in range(nb):
        q = (frame >= edges[i]) & (frame < edges[i + 1])
        im = render(x[q], y[q], px_nm, sr, 20, bounds=b)
        cc = np.fft.irfft2(np.fft.rfft2(im) * np.conj(np.fft.rfft2(tot)), s=im.shape)
        cc = np.fft.fftshift(cc); iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        def par(a, b_, c): return 0.5 * (a - c) / (a - 2 * b_ + c)
        dy = iy + par(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix]) - cc.shape[0] // 2
        dx = ix + par(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1]) - cc.shape[1] // 2
        sh.append((dx * sr / px_nm, dy * sr / px_nm))
    sh = np.array(sh); sh -= sh.mean(0); tc = 0.5 * (edges[1:] + edges[:-1])
    return np.stack([np.interp(np.arange(F), tc, sh[:, k]) for k in range(2)], 1)


def kasa(px, py):
    A = np.c_[2 * px, 2 * py, np.ones_like(px)]; b = px ** 2 + py ** 2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c[0], c[1], np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0))


def npc_rings(x, y, px_nm, sr=5.0):
    """pores = peaks of a 25 nm-blurred render; per pore the locs within 90 nm of the peak; circle fit (Kasa, 2 passes
    on locs within 35 nm of the ring). Returns radii (nm, in the assumed px_nm) and per-pore loc counts."""
    b = (x.min(), x.max(), y.min(), y.max())
    im = render(x, y, px_nm, sr, 25, bounds=b)
    pk = (im == ndi.maximum_filter(im, int(120 / sr))) & (im > np.percentile(im[im > 0], 99))
    ys, xs = np.nonzero(pk)
    X = (x - b[0]) * px_nm; Y = (y - b[2]) * px_nm
    from scipy.spatial import cKDTree
    tr = cKDTree(np.c_[X, Y])
    out = []
    for py_, px_ in zip(ys * sr, xs * sr):
        idx = tr.query_ball_point([px_, py_], 90)
        if len(idx) < 40: continue
        qx, qy = X[idx], Y[idx]
        cx, cy, r = kasa(qx, qy)
        d = np.abs(np.hypot(qx - cx, qy - cy) - r); k = d < 35
        if k.sum() < 30: continue
        cx, cy, r = kasa(qx[k], qy[k])
        rr = np.hypot(qx[k] - cx, qy[k] - cy)
        out.append((r, len(idx), np.std(rr), cx, cy))
    return np.array(out)


def run(cell, px_nm=128.0):
    os.makedirs(f'{OUT}/render', exist_ok=True)
    L = load_locs(cell)
    p, pc = L['pfit'], L['pcrlb']
    good = (p[:, 3] > 1000) & (pc[:, 0] < 0.1) & (pc[:, 1] < 0.1) & (L['pchi2'] / 225 < 3) & (np.abs(p[:, 0]) < 1.5) & (np.abs(p[:, 1]) < 1.5)
    x, y, fr = L['x'][good], L['y'][good], L['frame'][good]
    F = int(fr.max()) + 6
    res = dict(cell=cell, n=int(good.sum()), crlb_x_nm_median=float(np.median(pc[good, 0]) * px_nm), crlb_y_nm_median=float(np.median(pc[good, 1]) * px_nm))
    if gpu0_idle():
        d = run_dme(x, y, fr, pc[good, 0], pc[good, 1], F, f'{cell}', gpu='0'); res['drift'] = 'DME v2 (GPU 0)'
    else:
        d = rcc(x, y, fr, px_nm=px_nm); res['drift'] = 'CPU cross-correlation fallback (GPU 0 busy; DME v2 not run)'
    res['drift_ptp_nm'] = (np.ptp(d, 0) * px_nm).round(1).tolist()
    xc = x - d[fr, 0]; yc = y - d[fr, 1]
    np.savez(f'{OUT}/render/locs_{cell}.npz', x=xc, y=yc, frame=fr, z=p[good, 2], N=p[good, 3], crx=pc[good, 0], cry=pc[good, 1], drift=d)
    img = render(xc, yc, px_nm, 10.0, 10)
    fig = plt.figure(figsize=(14, 12)); plt.imshow(img, cmap='inferno', vmax=np.percentile(img, 99.7)); plt.title(f'{cell}: {good.sum()} locs, 10 nm px, drift: {res["drift"]}')
    plt.savefig(f'{OUT}/fig/render_{cell}.png', dpi=80, bbox_inches='tight'); plt.close()
    # zooms at the densest 2 x 2 um regions
    im2 = render(xc, yc, px_nm, 100.0, 0); sm = ndi.uniform_filter(im2, 20)
    zs = []
    for k in range(3):
        iy, ix = np.unravel_index(np.argmax(sm), sm.shape); zs.append((ix * 100 / px_nm + xc.min(), iy * 100 / px_nm + yc.min()))
        sm[max(0, iy - 25):iy + 25, max(0, ix - 25):ix + 25] = 0
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, (cx, cy) in zip(ax, zs):
        h = 1000 / px_nm
        q = (np.abs(xc - cx) < h) & (np.abs(yc - cy) < h)
        im = render(xc[q], yc[q], px_nm, 4.0, 6, bounds=(cx - h, cx + h, cy - h, cy + h))
        a.imshow(im, cmap='inferno', vmax=np.percentile(im, 99.8)); a.set_title(f'2 x 2 um at ({cx:.0f}, {cy:.0f}) px, 4 nm px')
    fig.savefig(f'{OUT}/fig/render_{cell}_zoom.png', dpi=80, bbox_inches='tight'); plt.close()
    if cell == 'NPC':
        R = npc_rings(xc, yc, px_nm)
        res['npc'] = dict(n_pores=int(len(R)))
        if len(R):
            ok = (R[:, 0] > 20) & (R[:, 0] < 90) & (R[:, 2] < 20)
            res['npc'].update(n_ring_like=int(ok.sum()), radius_nm_median=float(np.median(R[ok, 0])) if ok.any() else None,
                              radius_nm_p25_p75=np.percentile(R[ok, 0], [25, 75]).round(1).tolist() if ok.any() else None,
                              ring_width_sd_nm_median=float(np.median(R[ok, 2])) if ok.any() else None)
            # pixel-size implied if the label is Nup96 (ring radius 53.7 nm, Thevathasan et al. 2019) [assumption]
            if ok.any(): res['npc']['px_nm_if_Nup96'] = float(px_nm * 53.7 / np.median(R[ok, 0]))
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax[0].hist(R[:, 0], 60, range=(0, 120)); ax[0].set_xlabel(f'fitted ring radius (nm at {px_nm} nm/px)')
            # gallery: average of pores aligned at centre
            acc = np.zeros((60, 60))
            for r_, n_, s_, cx, cy in R[ok][:400]:
                q = np.hypot((xc - xc.min()) * px_nm - cx, (yc - yc.min()) * px_nm - cy) < 150
                h2, _, _ = np.histogram2d((yc[q] - yc.min()) * px_nm - cy, (xc[q] - xc.min()) * px_nm - cx, bins=60, range=((-150, 150), (-150, 150)))
                acc += h2
            ax[1].imshow(acc, cmap='inferno', extent=(-150, 150, 150, -150)); ax[1].set_title(f'sum of {min(400, ok.sum())} pores aligned on fitted centres (nm)')
            fig.savefig(f'{OUT}/fig/npc_rings.png', dpi=80, bbox_inches='tight'); plt.close()
    json.dump(res, open(f'{OUT}/diag/render_{cell}.json', 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 128.0)
