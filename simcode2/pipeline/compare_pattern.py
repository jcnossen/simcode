"""
Compare an estimated pattern with a calibrated reference pattern (common file format), in rad and nm.

  k        angle (deg), pitch (nm), |dk|; the phase error that dk causes over the crop (rms of dk.(r - r_c), gauge =
           mean removed)
  phases   per step at the crop centre r_c (psi_c = psi - k.r_c, eps means included); per angle the common offset
           (gauge: a global translation of that angle's SIMFLUX coordinate) and the step-relative residuals
  depth, relint (geometric mean 1; angle ratio)
  eps(t)   both traces on every 6th frame, mean removed: rms of the difference, correlation, rms of each
  budget   rms over the crop and time of the total phase difference with the per-angle gauge removed
  tiles    (optional, --pass1 dir) per 22 px tile: circular-mean residual phase of clean pass-1 emitters under each
           pattern (consistency with the pattern-free positions; the estimate is fitted to these data)
nm = rad / |k| * px.
usage: python -m simcode2.pipeline.compare_pattern --est <pattern_est.pt> --ref <pattern.pt> --nframes N [--pass1 <dir>] --out <json>
"""
import argparse, glob, json, math
import numpy as np, torch

PX = 108.33


def eps_at(P, f):
    d = P['dphi'].numpy().astype(np.float64); kf = float(P['knot_frames'])
    tt = f / kf; i0 = np.clip(np.floor(tt).astype(int), 0, d.shape[1] - 2); w = np.clip(tt - i0, 0, 1)
    return np.stack([d[a, i0] * (1 - w) + d[a, i0 + 1] * w for a in (0, 1)])


wrap = lambda v: (v + math.pi) % (2 * math.pi) - math.pi


def compare(E, Rf, nframes, crop=(5, 5, 110)):
    kE, kR = E['k'].numpy().astype(np.float64), Rf['k'].numpy().astype(np.float64)
    x0, y0, sz = crop; rc = np.array([x0 + sz / 2, y0 + sz / 2])
    gy, gx = np.mgrid[y0:y0 + sz, x0:x0 + sz] + 0.5
    out = dict(crop_centre=rc.tolist())
    f = np.arange(0, nframes, 6, dtype=np.float64)
    eE, eR = eps_at(E, f), eps_at(Rf, f)
    for a in (0, 1):
        kn = np.hypot(*kR[a]); nm = PX / kn
        dk = kE[a] - kR[a]
        fld = dk[0] * (gx - rc[0]) + dk[1] * (gy - rc[1])
        ang = lambda k: math.degrees(math.atan2(k[1], k[0]))
        o = dict(angle_est_deg=ang(kE[a]), angle_ref_deg=ang(kR[a]), dangle_deg=ang(kE[a]) - ang(kR[a]),
                 pitch_est_nm=2 * math.pi / np.hypot(*kE[a]) * PX, pitch_ref_nm=2 * math.pi / kn * PX,
                 dk=dk.tolist(), dk_norm=float(np.hypot(*dk)),
                 k_field_rms_rad=float(fld.std()), k_field_rms_nm=float(fld.std() * nm),
                 k_field_max_nm=float(np.abs(fld - fld.mean()).max() * nm))
        o['dpitch_nm'] = o['pitch_est_nm'] - o['pitch_ref_nm']
        st = slice(3 * a, 3 * a + 3)
        psiE = E['phi'].numpy()[st] - kE[a] @ rc + eE[a].mean(); psiR = Rf['phi'].numpy()[st] - kR[a] @ rc + eR[a].mean()
        d = wrap(psiE - psiR)
        c = math.atan2(np.sin(d).mean(), np.cos(d).mean())
        rel_d = wrap(d - c)
        o.update(phase_common_offset_rad=c, phase_common_offset_nm=c * nm, phase_step_resid_rad=rel_d.tolist(),
                 phase_step_resid_nm=(rel_d * nm).tolist(), phase_step_resid_rms_nm=float(np.sqrt((rel_d ** 2).mean()) * nm),
                 spacing_est=[float(wrap(psiE[j + 1] - psiE[j]) % (2 * math.pi)) for j in (0, 1)],
                 spacing_ref=[float(wrap(psiR[j + 1] - psiR[j]) % (2 * math.pi)) for j in (0, 1)])
        o['depth_est'] = float(E['m'].numpy()[st].mean()); o['depth_ref'] = float(Rf['m'].numpy()[st].mean())
        o['ddepth'] = o['depth_est'] - o['depth_ref']
        de, dr = eE[a] - eE[a].mean(), eR[a] - eR[a].mean()
        dd = de - dr
        o.update(eps_diff_rms_rad=float(dd.std()), eps_diff_rms_nm=float(dd.std() * nm), eps_corr=float(np.corrcoef(de, dr)[0, 1]),
                 eps_est_rms_nm=float(de.std() * nm), eps_ref_rms_nm=float(dr.std() * nm),
                 eps_diff_maxabs_nm=float(np.abs(dd).max() * nm))
        # total budget: step residual (per step, equal weight) + k field + eps difference, gauge removed
        tot = (np.mean(rel_d ** 2) + fld.var() + dd.var())
        o['total_phase_rms_nm'] = float(math.sqrt(tot) * nm)
        out[f'angle{a}'] = o
    lrE = E['logrel'].numpy() - E['logrel'].numpy().mean(); lrR = Rf['logrel'].numpy() - Rf['logrel'].numpy().mean()
    out['relint_est'] = np.exp(lrE).tolist(); out['relint_ref'] = np.exp(lrR).tolist()
    out['relint_diff'] = (np.exp(lrE) - np.exp(lrR)).tolist()
    out['angle_ratio_est'] = float(np.exp(lrE[:3]).mean() / np.exp(lrE[3:]).mean())
    out['angle_ratio_ref'] = float(np.exp(lrR[:3]).mean() / np.exp(lrR[3:]).mean())
    return out


def tile_maps(pass1_glob, pats, crop=(5, 5, 110), tile=22):
    from simcode2.pipeline.pattern_est import select, angle_Z, _interp_eps
    Ls = [np.load(f) for f in sorted(glob.glob(pass1_glob))]
    cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
    keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & (L['wstart'] < int(L['own'][1]) - 3) for L in Ls])
    R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
    x0, y0, sz = crop
    R['out'] = (R['tc'] >= 1) & (R['tc'] < 4) & (R['x'] >= x0) & (R['x'] < x0 + sz) & (R['y'] >= y0) & (R['y'] < y0 + sz)
    D = select(R)
    s = D['clean']
    w = D['w'][s]; f = w[:, None] + np.arange(6)[None]; st = f % 6
    tx = ((D['x'][s] - x0) // tile).astype(int); ty = ((D['y'][s] - y0) // tile).astype(int); nt = sz // tile
    tid = np.clip(ty, 0, nt - 1) * nt + np.clip(tx, 0, nt - 1)
    res = {}
    for nm_, P in pats.items():
        k = P['k'].numpy().astype(np.float64); psi = P['phi'].numpy().astype(np.float64)
        rel = np.exp(P['logrel'].numpy() - P['logrel'].numpy().mean()); kf = float(P['knot_frames'])
        dph = P['dphi'].numpy().astype(np.float64)
        for a in (0, 1):
            A, Z, cl = angle_Z(D['I'][s], st, psi, rel, a)
            ft = np.take_along_axis(f, cl, 1).mean(1)
            e = _interp_eps(dph, kf, ft)[a]
            u = Z * np.exp(-1j * (k[a, 0] * D['x'][s] + k[a, 1] * D['y'][s] - e))
            u = u / np.maximum(np.abs(u), 1e-9) * np.minimum(A, 4000)
            S_ = np.bincount(tid, weights=u.real, minlength=nt * nt) + 1j * np.bincount(tid, weights=u.imag, minlength=nt * nt)
            ph = np.angle(S_ * np.exp(-1j * np.angle(S_.sum())))
            kn = np.hypot(*k[a])
            res[f'{nm_}_angle{a}'] = dict(map_nm=(ph * PX / kn).reshape(nt, nt).round(2).tolist(),
                                          rms_nm=float(np.sqrt((ph ** 2).mean()) * PX / kn))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True); ap.add_argument('--ref', required=True)
    ap.add_argument('--nframes', type=int, required=True)
    ap.add_argument('--pass1', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    E, Rf = torch.load(a.est), torch.load(a.ref)
    res = compare(E, Rf, a.nframes)
    if a.pass1:
        res['tiles'] = tile_maps(a.pass1 + '/pass1_*.npz', {'est': E, 'ref': Rf})
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    for an in ('angle0', 'angle1'):
        o = res[an]
        print(f'{an}: angle {o["angle_est_deg"]:.3f} vs {o["angle_ref_deg"]:.3f} deg; pitch {o["pitch_est_nm"]:.2f} vs {o["pitch_ref_nm"]:.2f} nm '
              f'(dk field rms {o["k_field_rms_nm"]:.2f} nm, max {o["k_field_max_nm"]:.2f} nm)')
        print(f'   phase: common offset {o["phase_common_offset_rad"]:+.3f} rad = {o["phase_common_offset_nm"]:+.2f} nm (gauge); '
              f'step residuals {np.round(o["phase_step_resid_nm"], 2).tolist()} nm; spacing est {np.round(o["spacing_est"], 3).tolist()} ref {np.round(o["spacing_ref"], 3).tolist()}')
        print(f'   depth {o["depth_est"]:.3f} vs {o["depth_ref"]:.3f};  eps: diff rms {o["eps_diff_rms_nm"]:.2f} nm ({o["eps_diff_rms_rad"]:.3f} rad), '
              f'corr {o["eps_corr"]:.3f}, rms est/ref {o["eps_est_rms_nm"]:.2f}/{o["eps_ref_rms_nm"]:.2f} nm;  total phase rms {o["total_phase_rms_nm"]:.2f} nm')
    print('relint est', np.round(res['relint_est'], 3).tolist(), ' ref', np.round(res['relint_ref'], 3).tolist(),
          f' angle ratio est {res["angle_ratio_est"]:.3f} ref {res["angle_ratio_ref"]:.3f}')
    if 'tiles' in res:
        for k_, v in res['tiles'].items():
            print(f'tile phase map {k_}: rms {v["rms_nm"]:.2f} nm')


if __name__ == '__main__':
    main()
