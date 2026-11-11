"""m20r tuning (2026-11-11 psf3d retrain): raw-frame statistics (as out/n20/density_bg.raw_stats) and network-level pass-1 statistics
(as out/n20/netstats.stats: current det_n1 + sc5nb, p >= 0.5, 110-px crop) of a trial sim movie, next to the real 20 nM / 2.5 nM movie-2
values stored in out/n20/{density_bg,netstats}.json.
usage: python analysis/psf3d_nets/m20r_stats.py <trial dir (sim_movie.tif, p1/pass1_0.npz)> [n_frames]"""
import json, os, sys
import numpy as np, tifffile
from scipy import ndimage as ndi
from simcode2 import config
N20 = os.path.join(config.root('out'), 'n20'); sys.path.insert(0, N20)
import netstats as NS
G = 0.56


def raw(path, cam, n):
    t = tifffile.TiffFile(path); n = min(n, len(t.pages))
    e = (np.stack([t.pages[i].asarray() for i in range(n)]).astype(np.float32) - np.load(cam)['offset_adu'].astype(np.float32)) * G
    rn2 = np.load(cam)['readnoise_e'].astype(np.float32) ** 2
    c = slice(5, 115); mean = e.mean(0); med = np.median(e, 0); p10 = np.percentile(e, 10, 0)
    B = 600 if n >= 600 else n - n % 6; ev = []
    for b0 in range(0, n - B + 1, B):
        for s in range(6):
            fr = e[b0 + s::6][:B // 6]; ev.append(fr.var(0, ddof=1) - fr.mean(0) - rn2)
    ev = np.mean(ev, 0)
    return dict(mean=float(mean[c, c].mean()), med_median=float(np.median(med[c, c])), med_sd=float(med[c, c].std()),
                med_hp_sd=float((med - ndi.median_filter(med, 9))[c, c].std()), p10_mean=float(p10[c, c].mean()),
                ev_mean=float(ev[c, c].mean()))


if __name__ == '__main__':
    d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    R = json.load(open(f'{N20}/density_bg.json')); N = json.load(open(f'{N20}/netstats.json'))
    s = raw(f'{d}/sim_movie.tif', config.model('insitu/camera_maps_2p5nM_m2.npz'), n)
    ns = NS.stats(f'{d}/p1/pass1_0.npz')
    rk = ['mean', 'med_median', 'med_sd', 'med_hp_sd', 'p10_mean', 'ev_mean']
    L = [f'{"":12s}' + ''.join(f'{k:>11s}' for k in rk) + f'{"det/win/px":>12s}{"ph p50":>9s}{"iso p50":>9s}{"bg p50":>8s}{"nn<1.5":>8s}{"nn>4":>7s}']
    for nm, r, q in (('real 20 m2', R['20nM_m2']['raw'], N['real20']), ('real 2.5 m2', R['2.5nM_m2']['raw'], N['real25']), ('sim', s, ns)):
        L.append(f'{nm:12s}' + ''.join(f'{r[k]:11.2f}' for k in rk) + f'{q["det_win_px"]:12.4f}{q["phot_q"][2]:9.0f}{q["phot_iso_q"][2]:9.0f}'
                 f'{q["bg_q"][1]:8.2f}{q["nn_lt15"]:8.3f}{q["frac_iso"]:7.3f}')
    print('\n'.join(L)); open(f'{d}/stats.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(dict(raw=s, net=ns), open(f'{d}/stats.json', 'w'), indent=1)
