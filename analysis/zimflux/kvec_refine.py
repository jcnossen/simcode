"""Refine the even / odd lateral k of the 6 ms movie from the saved segment rows (fine grid in (kx, ky), then Nelder-Mead)."""
import json, numpy as np
from scipy.optimize import minimize
rows = np.load('/home/jc/simcode2/out/zimflux/inventory/kvec6ms_rows.npy')
x, y = rows[:, 0], rows[:, 1]
def R(k, ph): return abs(np.mean(np.exp(1j * (ph - x * k[0] - y * k[1]))))
out = {}
for name, col in (('even', 2), ('odd', 3)):
    ph = rows[:, col]
    kk = np.linspace(-1.7, 1.7, 1361)          # rad/px; |k| up to 1.7 -> pitch >= 3.7 px
    best = (0, None)
    for kx in kk:
        v = np.abs(np.exp(1j * (ph[None] - x[None] * kx - y[None] * 0)) * 0)  # placeholder to keep shapes simple
        ky = kk[:, None]
        z = np.exp(1j * (ph[None] - x[None] * kx - y[None] * ky)).mean(1)
        i = np.argmax(np.abs(z))
        if abs(z[i]) > best[0]: best = (abs(z[i]), (kx, kk[i]))
    r = minimize(lambda k: -R(k, ph), best[1], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8})
    k = r.x; p = 2 * np.pi / np.hypot(*k)
    out[name] = {'resultant': float(-r.fun), 'kx_rad_px': float(k[0]), 'ky_rad_px': float(k[1]), 'pitch_px': float(p),
                 'pitch_nm': float(p * 65), 'angle_deg_mod180': float(np.rad2deg(np.arctan2(k[1], k[0])) % 180)}
    print(name, out[name])
e, o = out['even'], out['odd']
ke = np.array([e['kx_rad_px'], e['ky_rad_px']]); ko = np.array([o['kx_rad_px'], o['ky_rad_px']])
print('angle between even and odd k (deg):', np.rad2deg(np.arccos(abs(ke @ ko) / np.linalg.norm(ke) / np.linalg.norm(ko))))
json.dump(out, open('/home/jc/simcode2/out/zimflux/inventory/kvec6ms_refined.json', 'w'), indent=1)
