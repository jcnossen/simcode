"""R24 reuse check, part D (in-situ fit side): the BASE in-situ PSF fit of analysis/nanoruler/nr_fit_insitu2.py on the EXISTING tracks
(tracks.npz stores raw ADU; nothing is re-extracted) with another camera offset / read noise, plus the fitted background, which the
original script does not print. The original script is executed unchanged (exec of its source with its own arguments, env
NR_OFFSET / NR_RN); afterwards its loss function is called once more to collect the linear solution (per-sample, per-orientation
background and per-frame intensities). Outputs go to --out (never to out/nanoruler/insitu_<name>).
usage: NR_OFFSET=99 NR_RN=1.6 python analysis/nanoruler/r24r_insitu_off.py <nr40|nr20|nr80> <out prefix>"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2')
from simcode2.util.proctitle import set_title
set_title('nr-r24r')
name, outp = sys.argv[1], sys.argv[2]
ROOT = '/home/jc/simcode2/out/nanoruler'
tr = f'{ROOT}/tracks_nr80.npz' if name == 'nr80' else f'{ROOT}/insitu_{name}/tracks.npz'
assert '/reuse/' in outp, 'outputs only under out/nanoruler/<name>/reuse/'
os.makedirs(os.path.dirname(outp), exist_ok=True)
src = '/home/jc/simcode2/analysis/nanoruler/nr_fit_insitu2.py'
sys.argv = [src, '--tracks', tr, '--gain', '0.45', '--intensities', '0', '--out', outp]
ns = {'__name__': '__main__', '__file__': src}
try:
    exec(compile(open(src).read(), src, 'exec'), ns)
except SystemExit:
    pass
with torch.no_grad():
    BG, II, CH = [], [], []
    for ch in ns['chunks']:
        _, I, bg, mu, var, data = ns['loss_and_linear'](ns['S'], ch)
        m = ns['S']['mask'][ch]
        BG.append(bg.cpu().numpy()); II.append(torch.where(m, I, torch.full_like(I, float('nan'))).cpu().numpy())
        CH.append(((data - mu) ** 2 / var)[m].mean().item())
bg = np.concatenate(BG); I = np.concatenate(II)
j = json.load(open(outp + '_psf.json'))
res = dict(name=name, offset=float(os.environ.get('NR_OFFSET', 100)), rn=float(os.environ.get('NR_RN', 1.6)), nll_per_px=float(ns['tot']), resid_var=j['resid_var'],
           chi2_per_px=float(np.mean(CH)), focus=j['focus'], zernike_norm=float(np.linalg.norm(j['zernike'])), zernike=j['zernike'],
           bg_median_per_orientation=np.nanmedian(bg, 0).tolist(), bg_mean=float(np.nanmean(bg)), bg_p5_p95=np.nanpercentile(bg, [5, 95]).tolist(),
           I_frame_median=float(np.nanmedian(I)), n=int(len(bg)))
json.dump(res, open(outp + '_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'zernike'}))
