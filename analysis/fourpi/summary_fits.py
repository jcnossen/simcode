"""Steps 2 and 4 summary: counts, photons, background, sigma distributions (Gaussian pass), astigmatism calibration,
PSF refit vs Gaussian (log-likelihood, chi2 / pixel, CRLBs), astigmatic z vs PSF z. usage: python summary_fits.py <cell>"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import load_locs
from zcal import astig_curve

OUT = f'{P4}/smlm'
from simcode.fourpi.setup import PX_NM as PX


def z_from_sigma(sx, sy, cx, cy):
    zg = np.linspace(-0.7, 0.7, 1401)
    ax, ay = astig_curve(cx, zg), astig_curve(cy, zg)
    d = (np.sqrt(sx)[:, None] - np.sqrt(ax)[None]) ** 2 + (np.sqrt(sy)[:, None] - np.sqrt(ay)[None]) ** 2
    j = d.argmin(1)
    return zg[j], np.sqrt(d[np.arange(len(j)), j])


def run(cell):
    L = load_locs(cell)
    zc = json.load(open(f'{OUT}/zcal/zcal_{cell}.json'))
    g = L['gfit']; gc = L['gcrlb']; p = L['pfit']; pc = L['pcrlb']
    nfr = 18000 * len(np.unique(L['fi']))
    good = (g[:, 2] > 300) & (gc[:, 0] < 0.25) & (np.abs(g[:, 0]) < 2) & (np.abs(g[:, 1]) < 2) & (g[:, 4] < 4.4) & (g[:, 5] < 4.4)
    res = dict(cell=cell, files=sorted(set(L['fi'].tolist())), n_det=int(len(g)), n_good=int(good.sum()), windows=nfr // 6,
               det_per_window=float(len(g) / (nfr / 6)), good_per_window=float(good.sum() / (nfr / 6)))
    q = lambda v: np.percentile(v, [10, 50, 90]).round(3).tolist()
    G = g[good]
    res['gauss'] = dict(N_p10_50_90=q(G[:, 2]), bg_per_px_6frame_4ch=q(G[:, 3]), sx=q(G[:, 4]), sy=q(G[:, 5]),
                        crlb_x_nm=q(gc[good, 0] * PX), crlb_y_nm=q(gc[good, 1] * PX), chi2_per_px=q(L['gchi2'][good] / 225))
    zg, dist = z_from_sigma(G[:, 4], G[:, 5], zc['curve_x'], zc['curve_y'])
    res['gauss']['z_astig_stage_um'] = q(zg)
    # PSF refit
    P = p[good]
    dll = L['pll'][good] - L['gll'][good]
    res['psf'] = dict(N=q(P[:, 3]), bg=q(P[:, 4]), z=q(P[:, 2]), crlb_x_nm=q(pc[good, 0] * PX), crlb_y_nm=q(pc[good, 1] * PX),
                      crlb_z_nm=q(pc[good, 2] * 1e3), chi2_per_px=q(L['pchi2'][good] / 225),
                      dLL_psf_minus_gauss=q(dll), frac_psf_better=float(np.mean(dll > 0)),
                      N_ratio_psf_gauss=q(P[:, 3] / G[:, 2]))
    br = G[:, 2] > 3000
    res['psf']['bright_chi2_gauss'] = q(L['gchi2'][good][br] / 225); res['psf']['bright_chi2_psf'] = q(L['pchi2'][good][br] / 225)
    res['psf']['bright_dLL'] = q(dll[br])
    cz = np.corrcoef(zg, P[:, 2])[0, 1]
    A = np.polyfit(zg, P[:, 2], 1)
    res['psf']['z_psf_vs_z_astig'] = dict(corr=float(cz), slope=float(A[0]), offset=float(A[1]),
                                         resid_sd_nm=float(np.std(P[:, 2] - np.polyval(A, zg)) * 1e3))
    json.dump(res, open(f'{OUT}/diag/fits_{cell}.json', 'w'), indent=1)
    print(json.dumps(res, indent=0))
    fig, ax = plt.subplots(1, 5, figsize=(24, 4))
    ax[0].hist(np.log10(G[:, 2]), 80); ax[0].set_xlabel('log10 N (e, 6-frame 4-ch sum, Gaussian)')
    ax[1].hist2d(G[:, 4], G[:, 5], bins=100, range=((0.6, 4.4), (0.6, 4.4)), cmap='magma'); ax[1].set_xlabel('sx (px)'); ax[1].set_ylabel('sy (px)')
    zz = np.array(zc['zstage']); gd = np.array(zc['good'])
    ax[2].plot(zz[gd], np.array(zc['sx'])[gd], 'o', ms=3, label='bead sx'); ax[2].plot(zz[gd], np.array(zc['sy'])[gd], 'o', ms=3, label='bead sy')
    zf = np.linspace(-0.7, 0.7, 200); ax[2].plot(zf, astig_curve(zc['curve_x'], zf)); ax[2].plot(zf, astig_curve(zc['curve_y'], zf))
    ax[2].set_xlabel('stage z (um)'); ax[2].set_ylim(0.5, 5); ax[2].legend(); ax[2].set_title('astigmatism calibration (bead)')
    ax[3].hist2d(zg, P[:, 2], bins=100, cmap='magma'); ax[3].set_xlabel('z astig (stage units, Gaussian)'); ax[3].set_ylabel('z PSF fit (um)')
    ax[4].hist(np.clip(dll, -50, 200), 100); ax[4].set_xlabel('logL(PSF) - logL(Gauss)')
    fig.suptitle(f'{cell}: pass 1 (Gaussian) and PSF refit'); fig.savefig(f'{OUT}/fig/fits_{cell}.png', dpi=70, bbox_inches='tight')
    return res


if __name__ == '__main__':
    run(sys.argv[1])
