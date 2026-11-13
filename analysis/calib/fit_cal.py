"""per-photon-bin calibration of the sc5nb pattern solve on the TUNING seed (202) of the experiment-matched gate scenes
(disjoint from the eval seeds 301-303):
  1. kap_p(bin)  = robust sd of the prior z (x_prior - truth) / s_prior          (bins of network window photons)
  2. kap_I2(bin) = robust z^2 of the full-covariance whitened I_t errors vs the realized GT I_t
  3. solve with the nuisance Sigma (cmult, cmult_ang) + 1-2 + prior_off, then sig_ph(bin) = robust zcal of e / s_cal
     (multiplies s_cal so that zcal = 1 per bin)
usage: fit_cal.py out.json scen1,scen2 '{"cmult":0.10,"cmult_ang":0.10}' [base.json]"""
import os
import sys, json, numpy as np
from simcode import config as _cfg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blib as B
EDG = [100, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]
rob = lambda v: float(1.4826 * np.median(np.abs(v[np.isfinite(v)])))
out, scen, extra = sys.argv[1], sys.argv[2].split(','), json.loads(sys.argv[3])
base = json.load(open(sys.argv[4] if len(sys.argv) > 4 else _cfg.model('calib/calib_sc_hd_sc5nbb15_pm.json')))
cal = dict(base); cal.update(extra)
Rs = [B.load(sc, (202,)) for sc in scen]
TI = [B.targets(R) for R in Rs]
zp, zI, ph = [], [], []
for R, (ti, oi) in zip(Rs, TI):
    tr = np.c_[R['gt_x'][ti], R['gt_y'][ti]]
    off = np.array(cal.get('prior_off', [0.0, 0.0]))
    zp.append((np.c_[R['sc_x'][oi], R['sc_y'][oi]] - off - tr) / np.exp(R['sc_lsxy'][oi]))
    zI.append(np.linalg.solve(R['sc_L'][oi], (R['sc_I'][oi] - R['gt_I'][ti])[..., None])[..., 0])
    ph.append(R['sc_I'][oi].sum(1))
zp, zI, ph = np.concatenate(zp), np.concatenate(zI), np.concatenate(ph)
C, KP, KI = [], [], []
for lo, hi in zip(EDG[:-1], EDG[1:]):
    m = (ph >= lo) & (ph < hi)
    if m.sum() < 100: continue
    C.append(float(np.median(ph[m]))); KP.append(rob(zp[m])); KI.append(rob(zI[m]) ** 2)
    print(f'  bin {lo:5.0f}-{hi:<6.0f} n {m.sum():6d} centre {C[-1]:7.0f} kap_p {KP[-1]:.3f} kap_I2 {KI[-1]:.3f}')
cal['kap_bins'] = dict(ph=C, kap_p=KP, kap_I2=KI)
cal.pop('sig_ph', None)
# 3. per-bin sigma factor after the solve
zs, ph2 = [], []
for R, (ti, oi) in zip(Rs, TI):
    sol = B.solve(R, oi, cal)
    tr = np.c_[R['gt_x'][ti], R['gt_y'][ti]]
    zs.append((np.c_[sol['x'], sol['y']] - tr) / np.c_[sol['sx'], sol['sy']]); ph2.append(R['sc_I'][oi].sum(1))
zs, ph2 = np.concatenate(zs), np.concatenate(ph2)
C2, F = [], []
for lo, hi in zip(EDG[:-1], EDG[1:]):
    m = (ph2 >= lo) & (ph2 < hi)
    if m.sum() < 100: continue
    C2.append(float(np.median(ph2[m]))); F.append(rob(zs[m]))
    print(f'  sigma bin {lo:5.0f}-{hi:<6.0f} n {m.sum():6d} zcal before {F[-1]:.3f}')
cal['sig_ph'] = dict(ph=C2, f=F)
cal['note'] = (f'2026-11-09 bright-regime calibration (<out>/bright/fit_cal.py): base {base.get("note", "")[:60]}...; '
               f'nuisance Sigma {extra}; per-network-photon-bin kap_p / kap_I2 / sigma factor fitted on the tuning seed 202 of {scen}')
cal['fit_scenes'] = scen
json.dump(cal, open(out, 'w'), indent=1)
print('->', out)
