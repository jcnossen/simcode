"""Consequence of field-dependent PSF deviations for the explicit solve (item 5 of the field-dependence request).

For each deviation (sizes from the measured tile maps, out/4pi/field/), the coherent 4Pi PSF is perturbed, noise-free
read-outs (T, C per frame and channel, as defined in training) are rendered for emitters over z in [-0.3, 0.3] um at
N = 3000 e per window (on-through, true pattern and phase field), and solved with the GLOBAL model (the production
Solve4Pi, 'on' model, weak prior sigma 1 px / 0.2 um so that the prior does not hide the bias). Reported: bias in
x, y, z (median and max |.| over z) vs the photon-limited precision at that photon count.
The network's read-outs are trained on the global PSF; this calculation assumes it reports the defined read-outs of the
perturbed data faithfully (the explicit-model part of the error). Network-side errors are measured separately on real
data (field maps of DAFL / chi2).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/field_bias4pi.py
"""
import copy, json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, psi_from_eps
from simcode2.fourpi.sim4pi import DELTA_A

OUT = f'{REPO}/out/4pi/field'


def variants(st):
    V = {'reference (global PSF)': st}
    def mod(name, f):
        s = copy.deepcopy(st); f(s); V[name] = s
    for e in (5, 10, 25):
        mod(f'inter-pair phase {e:+d} deg (s pair)', lambda s, e=e: s.__setitem__('phi', [s['phi'][0], s['phi'][1] + math.radians(e), s['phi'][2], s['phi'][3] + math.radians(e)]))
    mod('coherence x0.7 (visibility 0.33 -> 0.23)', lambda s: s.__setitem__('gamma', s['gamma'] * 0.7))
    for db in (-0.08, 0.08):
        mod(f'blur {db:+.2f} px^2 (width {db / (2 * 1.3) * 128:+.0f} nm)', lambda s, db=db: s.__setitem__('blur_px2', max(s['blur_px2'] + db, 0.01)))
    for dn in (-0.01, 0.01):
        def f(s, dn=dn):
            zt = list(s['zt']); zb = list(s['zb']); zt[1] += dn; zb[1] -= dn; s['zt'] = zt; s['zb'] = zb
        mod(f'astigmatism Noll 6 {dn:+.3f} waves (both arms)', f)
    def ftau(s):
        t = np.array(s['tau']) * np.array([1.0, 0.9, 1.0, 1.1]); s['tau'] = (t / t.sum()).tolist()
    mod('transmission ch1 -10 % / ch3 +10 %', ftau)
    return V


def main():
    from simcode2.util.proctitle import auto; auto('4pi-field')
    dev = 'cuda'
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    ref_lut = LUT4Pi(st, P=15, dev=dev)
    sv = Solve4Pi(ref_lut); sv.build_table()
    zs = np.linspace(-0.3, 0.3, 25)
    nxy = 16; n = len(zs) * nxy
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.rand(n, device=dev, generator=g) - 0.5); y = (torch.rand(n, device=dev, generator=g) - 0.5)
    z = torch.tensor(np.repeat(zs, nxy), device=dev, dtype=torch.float32)
    Phi0 = torch.rand(n, device=dev, generator=g) * 2 * math.pi
    eps = torch.zeros(n, 2, device=dev); m = torch.full((n,), 0.9, device=dev); psi = psi_from_eps(eps)
    doff = torch.zeros(n, 4, 2, device=dev)
    F = 3000.0 / 6
    res = {}
    o_ = torch.arange(15, device=dev, dtype=torch.float32) - 7
    oy, ox = torch.meshgrid(o_, o_, indexing='ij')
    w = torch.exp(-((ox[None] - x[:, None, None]) ** 2 + (oy[None] - y[:, None, None]) ** 2) / 2.0)
    for name, s in variants(st).items():
        lut = LUT4Pi(s, P=15, dev=dev)
        Pt, _, _ = sv.pattern(x, y, m, psi)
        Nt = F * Pt
        T = torch.zeros(n, 6, 4, device=dev); C = torch.zeros(n, 6, 4, device=dev)
        for c in range(4):
            for a in range(2):
                gimg = lut.evaluate_ch(torch.stack([x, y, z], 1), Phi0 + DELTA_A[a], c)
                for t in range(3 * a, 3 * a + 3):
                    T[:, t, c] = Nt[:, t] * gimg.sum((-1, -2)); C[:, t, c] = Nt[:, t] * (gimg * w).sum((-1, -2))
        L = torch.zeros(n, 6, 4, 2, 2, device=dev)
        L[..., 0, 0] = (T + 15).sqrt(); L[..., 1, 1] = (C + 4).sqrt()
        o = dict(T=T, C=C, L=L, dxy=torch.stack([x, y], 1), z=z, ls=torch.log(torch.tensor([1.0, 1.0, 0.2], device=dev)).expand(n, 3))
        r = sv.solve(o, Phi0, doff, m, psi, mode='on')
        e = (r['th'][:, :3] - torch.stack([x, y, z], 1)).cpu().numpy() * np.array([128, 128, 1000])
        eb = e.reshape(len(zs), nxy, 3)
        med = np.median(eb, 1)                                                     # bias vs z
        res[name] = dict(bias_med_nm=np.median(med, 0).tolist(), bias_maxabs_nm=np.abs(med).max(0).tolist(),
                         spread_over_xy_nm=np.median(eb.std(1), 0).tolist(), bias_vs_z=med.tolist())
        print(f'{name:48s} bias median x/y/z {res[name]["bias_med_nm"][0]:6.2f} {res[name]["bias_med_nm"][1]:6.2f} {res[name]["bias_med_nm"][2]:6.2f} | '
              f'max over z {res[name]["bias_maxabs_nm"][0]:5.2f} {res[name]["bias_maxabs_nm"][1]:5.2f} {res[name]["bias_maxabs_nm"][2]:6.2f} nm', flush=True)
    res['_z_um'] = zs.tolist(); res['_N_window'] = 3000
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f'{OUT}/field_bias.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
