"""(2026-11-11 in-situ 3D PSF) Summary tables of the psf3d fits: per movie the final PSF parameters (out/psf3d/final) and the held-out
evidence (cb0 fit / cb1 held-out, out/psf3d/ho_*.npz) for the model sequence base -> line blur -> + photoselection / + per-angle
blur, and the cross-prediction matrix (cb1 samples of movie B with the line-blur fit of movie A). Prints markdown; writes
out/psf3d/summary.json.  usage: python analysis/psf3d/summary.py"""
import json, os, math
import numpy as np, torch
from simcode2.insitu.psf_compare import summarize

O = '/home/jc/simcode2/out/psf3d'
MOV = [('r05', '0.5 nM cover1/1 (ref)', 'bl'), ('m1809', '1809/2 (low)', 'bli'), ('m25', '2.5 nM/2 600 mW (medium)', 'bli'), ('m20', '20 nM/2 (high)', 'bli')]
res = {}


def blur_desc(c):
    C = np.array([[c[0], c[2]], [c[2], c[1]]]); w, v = np.linalg.eigh(C)
    th = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    return dict(sigma_major_px=float(math.sqrt(max(w[1], 0))), sigma_minor_px=float(math.sqrt(max(w[0], 0))), theta_deg=th)


def cone(g):
    if g is None or g <= 0: return None
    S = math.sqrt(g / 0.4)
    if S >= 1: return 0.0
    c = (-1 + math.sqrt(1 + 8 * S)) / 2
    return math.degrees(math.acos(c))


print('## Final PSF per movie (all samples)\n')
print('| movie | focus um | Z5 | Z6 | Z11 | line blur sigma px (theta deg) | gamma | zeta | cone half-angle | resid var |')
print('|---|---|---|---|---|---|---|---|---|---|')
for m, nm, _ in MOV:
    f = f'{O}/final/{m}_blps3_psf.pt'
    if not os.path.exists(f): continue
    S = torch.load(f, weights_only=False); ex = S['extras']; J = json.load(open(f'{O}/final/{m}_blps3_psf.json'))
    z = S['zcoef'].tolist(); bd = blur_desc(ex['blur_px2'][0])
    res[m] = dict(name=nm, focus=float(S['focus']), zernike=z, blur_px2=ex['blur_px2'][0], blur=bd, gamma=ex['gamma'], zeta=ex['zeta'],
                  cone_deg=cone(ex['gamma']), resid_var=J['resid_var'])
    print(f"| {nm} | {float(S['focus']):.3f} | {z[0]:+.3f} | {z[1]:+.3f} | {z[6]:+.3f} | {bd['sigma_major_px']:.3f} ({bd['theta_deg']:+.1f}) | "
          f"{ex['gamma']:.3f} | {ex['zeta']:.3f} | {cone(ex['gamma']) or float('nan'):.0f} deg | {J['resid_var']:.3f} |")

print('\n## Held-out evidence (cb0 fit, cb1 held-out, 6000 samples; paired trimmed z, + = better)\n')
print('| movie | step | dNLL total | trim z | ddM r3 (px^2) | a0 / a1 | angle contrast data / model |')
print('|---|---|---|---|---|---|---|')
for m, nm, b in MOV:
    base = 'base' if m == 'r05' else 'basei'
    steps = [(base, None), (b, base), (b + ('Fad' if m != 'r05' else 'Fad'), b), (b + 'Fps3', b)]
    if m == 'r05': steps = [('base', None), ('bl', 'base'), ('blFad', 'bl'), ('blFps3', 'bl')]
    res.setdefault(m, {})['heldout'] = {}
    for v, ref in steps:
        fn = f'{O}/ho_{m}_{v}.npz'
        if not os.path.exists(fn): continue
        R, _ = summarize(fn, dict(np.load(f'{O}/ho_{m}_{ref}.npz')) if ref else None)
        res[m]['heldout'][v] = R
        d = f"{R['dnll_total']:+.0f}" if ref else '-'; zt = f"{R['dnll_trim_z']:+.1f}" if ref else '-'
        print(f"| {nm} | {v} vs {ref or '-'} | {d} | {zt} | {R['ddM_r3']:+.4f} ± {R['ddM_r3_se']:.4f} | {R['ddM_r3_a0']:+.4f} / {R['ddM_r3_a1']:+.4f} | "
              f"{R['angcontrast_r3_data']:+.4f} / {R['angcontrast_r3_model']:+.4f} |")

print('\n## Cross-prediction (line-blur fits; trimmed paired z of [PSF of A] vs [own PSF] on B\'s held-out; negative = A worse)\n')
print('| test movie B \\ PSF of A | ' + ' | '.join(nm for _, nm, _ in MOV) + ' |')
print('|---|' + '---|' * len(MOV))
X = {}
for B, nmB, _ in MOV:
    own = f'{O}/ho_{B}_bli.npz'
    if not os.path.exists(own): continue
    ref = dict(np.load(own)); row = []
    for A, _, _ in MOV:
        if A == B: row.append('own'); continue
        fn = f'{O}/ho_{B}_x{A}_bli.npz'
        if not os.path.exists(fn): row.append('n/a'); continue
        R, _ = summarize(fn, ref); X[f'{B}<-{A}'] = R
        row.append(f"{R['dnll_trim_per_sample']:+.3f}/smp (z {R['dnll_trim_z']:+.1f}), ddM {R['ddM_r3']:+.3f}")
    print(f'| {nmB} | ' + ' | '.join(row) + ' |')
res['cross'] = X
json.dump(res, open(f'{O}/summary.json', 'w'), indent=1, default=float)
