"""R24 reuse check, part C (density question, nr20): what do the nr80 detector / head / solve return for two emitters 0.31 px (20 nm)
apart?  Scene 'pair0.31' of simcode.hd.gate (gate.py takes the separation from the scene name: pair<sep px>; 9 pairs per FOV on
an 18 px grid, random orientation, both emitters 300-5000 photons per fully-on window (1:1 brightness), EACH emitter on for >= 3 of
the 6 frames (on-time 50 frames) so that about a quarter of the pairs have one emitter switching inside the window), rendered
with the movie's calib by r24r_gates.sh (seed 301), front dump of the nr80 nets, solved here with the nr80 calibration.
Per pair: the stored candidates (score >= 0.2) within 2 px of the pair centre; the top-score row is compared with the
photon-weighted mean position of the two emitters (weights = GT window photons), along and across the pair axis.
Controls from the same movie's scenes: iso (single emitters; sigma and q/dof versus network photons) and the 'good' rows
(score >= 0.63) of rs05qm (reference population for a sigma rank: u = fraction of the reference rows with a smaller sigma;
the table build's sigma rescale uses the same rank inside a 10 px tile of the movie, simcode.strict.tile_rank).
usage: [R24R_DEV=cpu] python analysis/nanoruler/r24r_pair031.py <nr40|nr20> [scene, default pair0.31]  -> out/nanoruler/<name>/reuse/pair031_<name>.txt / .json"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/jc/simcode2'); sys.path.insert(0, os.path.join(HERE, '..', 'calib'))
import plib as P
from simcode.hd import sc as SCM
from simcode.util.proctitle import set_title
set_title('nr-r24r')
PX = 65.0; P.PX = PX
name = sys.argv[1]; SCENE = sys.argv[2] if len(sys.argv) > 2 else 'pair0.31'
F = os.environ.get('R24R_FRONT', 's1')
ROOT = '/home/jc/simcode2/out/nanoruler'; FE = f'{ROOT}/{name}/reuse/fe_{name}'
CAL = json.load(open({'s1': f'{ROOT}/models/calib/calib_sc_hd_nr80.json', 's2': f'{ROOT}/models/calib/calib_p2L_nr80.json'}[F]))
dev = os.environ.get('R24R_DEV', 'cuda')


@torch.no_grad()
def solve(R, oi):
    Pt = torch.tensor(R['patt_est'], device=dev, dtype=torch.float32); tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    res = {k: [] for k in ('x', 'y', 'sx', 'sy', 'q', 'kin', 'dnll', 'xp', 'yp')}
    for c0 in range(0, len(oi), 8192):
        j = oi[c0:c0 + 8192]
        o = dict(I=tt(R['sc_I'][j]), L=tt(R['sc_L'][j]), x=tt(R['sc_x'][j]), y=tt(R['sc_y'][j]), lsxy=tt(R['sc_lsxy'][j]), x_det=tt(R['sc_x_det'][j]), y_det=tt(R['sc_y_det'][j]))
        s = SCM.sc_solve(o, Pt, torch.tensor(R['sc_b'][j], device=dev).long(), CAL)
        res['x'].append(s['r'][:, 0]); res['y'].append(s['r'][:, 1]); res['sx'].append(s['s_cal'][:, 0]); res['sy'].append(s['s_cal'][:, 1]); res['q'].append(s['q'])
        res['kin'].append(s['kin'].float()); res['dnll'].append(s['dnll'].float() if 'dnll' in s else torch.full_like(s['q'], float('nan')))
        res['xp'].append(s['r_prior'][:, 0]); res['yp'].append(s['r_prior'][:, 1])
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}


def main():
    L = [f'# {name}: nr80 front {F} on scene {SCENE} (two emitters {SCENE[4:]} px = {float(SCENE[4:]) * PX:.1f} nm apart), seed 301, nr80 calibration; lengths in nm (65 nm px)']
    out = {}
    # ---- controls
    Ri = P.load('iso', F, seeds=(301,), d=FE); ti, oi = P.targets(Ri); si = solve(Ri, oi)
    phi = Ri['sc_I'][oi].sum(1); sgi = 0.5 * (si['sx'] + si['sy']) * PX
    edges = np.exp(np.linspace(np.log(300), np.log(12000), 13)); ctr = np.sqrt(edges[1:] * edges[:-1])
    med_s = np.array([np.median(sgi[(phi >= a) & (phi < b)]) if ((phi >= a) & (phi < b)).sum() > 10 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    q95 = np.array([np.percentile(si['q'][(phi >= a) & (phi < b)], 95) if ((phi >= a) & (phi < b)).sum() > 10 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    q50 = np.array([np.median(si['q'][(phi >= a) & (phi < b)]) if ((phi >= a) & (phi < b)).sum() > 10 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    okb = np.isfinite(med_s)
    s_iso = lambda ph: np.exp(np.interp(np.log(ph), np.log(ctr[okb]), np.log(med_s[okb])))
    q95_iso = lambda ph: np.interp(np.log(ph), np.log(ctr[okb]), q95[okb]); q50_iso = lambda ph: np.interp(np.log(ph), np.log(ctr[okb]), q50[okb])
    ei = np.c_[si['x'] - Ri['gt_x'][ti], si['y'] - Ri['gt_y'][ti]] * PX
    Rr = P.load('rs05qm', F, seeds=(301,), d=FE); gi = np.flatnonzero(Rr['sc_score'] >= 0.63); sr = solve(Rr, gi)
    ref_sig = np.sort(0.5 * (sr['sx'] + sr['sy']) * PX); ref_ph = Rr['sc_I'][gi].sum(1)
    urank = lambda s: np.searchsorted(ref_sig, s) / len(ref_sig)
    L.append(f'controls: iso single emitters n {len(oi)}; rs05qm good rows (reference population of the sigma rank) n {len(gi)}, network photons p5/p50/p95 '
             f'{np.percentile(ref_ph, [5, 50, 95]).round(0).tolist()}, sigma p5/p45/p50/p95 {np.percentile(ref_sig, [5, 45, 50, 95]).round(2).tolist()} nm')
    # ---- pairs
    R = P.load(SCENE, F, seeds=(301,), d=FE)
    allc = np.arange(len(R['sc_score'])); S = solve(R, allc)
    gx, gy, gf, gph, gon = R['gt_x'], R['gt_y'], R['gt_fov'], R['gt_phot'], R['gt_onfr']
    # pair partner = nearest emitter of the same FOV
    order = np.lexsort((gx, gf)); partner = np.full(len(gx), -1)
    for f in np.unique(gf):
        ii = np.flatnonzero(gf == f); D = np.hypot(gx[ii][:, None] - gx[ii][None], gy[ii][:, None] - gy[ii][None]); np.fill_diagonal(D, 1e9)
        j = D.argmin(1); ok = D[np.arange(len(ii)), j] < 0.6
        partner[ii[ok]] = ii[j[ok]]
    a = np.flatnonzero((partner >= 0) & (np.arange(len(gx)) < partner)); b = partner[a]
    sep = np.hypot(gx[a] - gx[b], gy[a] - gy[b])
    inter = (gx[a] >= 9) & (gx[a] <= 54) & (gy[a] >= 9) & (gy[a] <= 54)
    a, b, sep = a[inter], b[inter], sep[inter]
    tot = gph[a] + gph[b]; w = gph[b] / tot
    cx = gx[a] * (1 - w) + gx[b] * w; cy = gy[a] * (1 - w) + gy[b] * w                    # photon-weighted mean
    ux = (gx[b] - gx[a]) / sep; uy = (gy[b] - gy[a]) / sep                                  # pair axis a -> b
    rows = []
    cb = R['sc_b']; cxd, cyd = S['x'], S['y']
    byf = {}
    for i, f in enumerate(cb): byf.setdefault(int(f), []).append(i)
    for k in range(len(a)):
        ii = np.array(byf.get(int(gf[a[k]]), []), int)
        if len(ii): ii = ii[np.hypot(R['sc_x_det'][ii] - cx[k], R['sc_y_det'][ii] - cy[k]) < 2.0]
        sc_ = R['sc_score'][ii] if len(ii) else np.zeros(0)
        r = dict(n02=len(ii), n05=int((sc_ >= 0.5).sum()), n063=int((sc_ >= 0.63).sum()), tot=tot[k], wmin=min(w[k], 1 - w[k]),
                 on_a=gon[a[k]], on_b=gon[b[k]], sep=sep[k])
        if len(ii):
            t = ii[sc_.argmax()]
            dx, dy = (cxd[t] - cx[k]) * PX, (cyd[t] - cy[k]) * PX
            r.update(score=float(sc_.max()), e_al=dx * ux[k] + dy * uy[k], e_pe=-dx * uy[k] + dy * ux[k], sx=S['sx'][t] * PX, sy=S['sy'][t] * PX,
                     q=S['q'][t], kin=S['kin'][t], dnll=S['dnll'][t], ph_net=R['sc_I'][t].sum(), sprior=float(np.exp(R['sc_lsxy'][t]).mean() * PX),
                     d_a=np.hypot(cxd[t] - gx[a[k]], cyd[t] - gy[a[k]]) * PX, d_b=np.hypot(cxd[t] - gx[b[k]], cyd[t] - gy[b[k]]) * PX,
                     e_mid_al=((cxd[t] - 0.5 * (gx[a[k]] + gx[b[k]])) * ux[k] + (cyd[t] - 0.5 * (gy[a[k]] + gy[b[k]])) * uy[k]) * PX,
                     # sigma along / across the pair axis (diagonal covariance in x, y)
                     s_al=np.hypot(S['sx'][t] * ux[k], S['sy'][t] * uy[k]) * PX, s_pe=np.hypot(S['sx'][t] * uy[k], S['sy'][t] * ux[k]) * PX)
            if len(ii) > 1:
                t2 = ii[np.argsort(sc_)[-2]]; r.update(score2=float(R['sc_score'][t2]), d12=float(np.hypot(cxd[t] - cxd[t2], cyd[t] - cyd[t2]) * PX))
        rows.append(r)
    T = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in set().union(*[set(r) for r in rows])}
    both = (T['on_a'] >= 5.99) & (T['on_b'] >= 5.99); sw = ~both
    L.append(f'pairs (interior): {len(rows)}; both emitters on in all 6 frames: {int(both.sum())}; one emitter switching inside the window (on 3-6 frames): {int(sw.sum())}')
    rob = lambda v: float(1.4826 * np.median(np.abs(v[np.isfinite(v)] - np.median(v[np.isfinite(v)])))) if np.isfinite(v).sum() > 3 else float('nan')
    rms = lambda v: float(np.sqrt(np.nanmean(v ** 2)))
    bands = [(600, 1500), (1500, 3000), (3000, 10000), (600, 10000)]
    for cname, cm in (('BOTH ON (6 + 6 frames)', both), ('ONE SWITCHING (neighbour on for 3-6 frames)', sw)):
        L.append(f'\n== {cname}; bands = total GT photons of the two emitters in the window')
        out[cname] = {}
        for lo, hi in bands:
            m = cm & (T['tot'] >= lo) & (T['tot'] < hi); n = int(m.sum())
            if n < 15: continue
            g = m & (T['n063'] >= 1); ng = int(g.sum())
            sig = 0.5 * (T['sx'][g] + T['sy'][g]); ph = T['ph_net'][g]
            d = dict(n=n, frac_rows_ge1_at02=float((T['n02'][m] >= 1).mean()), frac_good_ge1=float((T['n063'][m] >= 1).mean()), frac_good_eq1=float((T['n063'][m] == 1).mean()),
                     frac_good_ge2=float((T['n063'][m] >= 2).mean()), frac_05_ge2=float((T['n05'][m] >= 2).mean()),
                     net_over_gt_photons=float(np.median(ph / T['tot'][g])), bias_along=float(np.mean(T['e_al'][g])), bias_along_se=float(np.std(T['e_al'][g]) / np.sqrt(max(ng, 1))),
                     rms_along=rms(T['e_al'][g]), rms_across=rms(T['e_pe'][g]), rsd_along=rob(T['e_al'][g]), rsd_across=rob(T['e_pe'][g]),
                     sigma_median=float(np.median(sig)), sigma_iso_same_ph=float(np.median(s_iso(ph))), sigma_ratio_iso=float(np.median(sig / s_iso(ph))),
                     z_along=rob(T['e_al'][g] / T['s_al'][g]), z_across=rob(T['e_pe'][g] / T['s_pe'][g]),
                     q_median=float(np.median(T['q'][g])), q_iso_median=float(np.median(q50_iso(ph))), frac_q_gt_iso95=float(np.mean(T['q'][g] > q95_iso(ph))),
                     kin_frac=float(np.mean(T['kin'][g])), u_median=float(np.median(urank(sig))),
                     sprior_median=float(np.median(T['sprior'][g])), dist_nearer_median=float(np.median(np.minimum(T['d_a'][g], T['d_b'][g]))),
                     wmin_median=float(np.median(T['wmin'][m])), e_mid_abs_median=float(np.median(np.abs(T['e_mid_al'][g]))))
            if cname.startswith('ONE'):
                # signed along-axis offset from the brighter (longer on) emitter towards the weaker one, in units of the separation: 0 = on the bright emitter, wmin = weighted mean
                sgn = np.where(T['on_a'][g] >= T['on_b'][g], 1.0, -1.0)      # axis a -> b; brighter = a -> positive towards b
                wm = T['wmin'][g]; sepnm = T['sep'][g] * PX
                pos_from_bright = (T['e_al'][g] * sgn + wm * sepnm)            # along-axis position measured from the bright emitter
                d.update(pos_from_bright_mean=float(np.mean(pos_from_bright)), weighted_mean_from_bright=float(np.mean(wm * sepnm)))
            out[cname][f'{lo}-{hi}'] = d
            L.append(f'  {lo}-{hi} ph: {n} pairs | rows: >= 1 stored {d["frac_rows_ge1_at02"]:.3f}, good (p >= 0.63): one {d["frac_good_eq1"]:.3f}, two or more {d["frac_good_ge2"]:.3f}, none {1 - d["frac_good_ge1"]:.3f}'
                     f' (two at p >= 0.5: {d["frac_05_ge2"]:.3f})')
            L.append(f'      top good row vs the photon-weighted mean: bias along the pair axis {d["bias_along"]:+.2f} +- {d["bias_along_se"]:.2f}, rms along {d["rms_along"]:.2f} / across {d["rms_across"]:.2f}'
                     f' (robust sd {d["rsd_along"]:.2f} / {d["rsd_across"]:.2f}); median distance to the nearer emitter {d["dist_nearer_median"]:.2f}; median weight of the weaker emitter {d["wmin_median"]:.2f}')
            L.append(f'      network photons / GT total {d["net_over_gt_photons"]:.3f}; sigma median {d["sigma_median"]:.2f} (single emitter of the same network photons: {d["sigma_iso_same_ph"]:.2f}, ratio {d["sigma_ratio_iso"]:.3f});'
                     f' zcal vs weighted mean along {d["z_along"]:.2f} / across {d["z_across"]:.2f}; head prior sigma {d["sprior_median"]:.2f}')
            L.append(f'      flags: q/dof median {d["q_median"]:.2f} (single: {d["q_iso_median"]:.2f}), fraction above the single-emitter 95th percentile {d["frac_q_gt_iso95"]:.3f}; kin flag {d["kin_frac"]:.3f};'
                     f' sigma rank u median {d["u_median"]:.2f} (reference population: 0.5)'
                     + (f'; position along the axis from the fully-on emitter {d["pos_from_bright_mean"]:+.2f} (photon-weighted mean would be {d["weighted_mean_from_bright"]:+.2f})' if 'pos_from_bright_mean' in d else ''))
    # single-emitter control numbers in the same format
    L.append('\n== control: iso single emitters (same calibration), bands = GT photons')
    out['iso'] = {}
    for lo, hi in bands:
        m = (Ri['gt_phot'][ti] >= lo) & (Ri['gt_phot'][ti] < hi) & (Ri['sc_score'][oi] >= 0.63)
        if m.sum() < 15: continue
        sig = sgi[m]; d = dict(n=int(m.sum()), rms_x=rms(ei[m, 0]), rms_y=rms(ei[m, 1]), sigma_median=float(np.median(sig)), zcal=rob((ei[m] / (np.c_[si['sx'], si['sy']][m] * PX)).ravel()),
                               q_median=float(np.median(si['q'][m])), kin_frac=float(np.mean(si['kin'][m])), u_median=float(np.median(urank(sig))))
        out['iso'][f'{lo}-{hi}'] = d
        L.append(f'  {lo}-{hi} ph: n {d["n"]}  rms x / y {d["rms_x"]:.2f} / {d["rms_y"]:.2f}  sigma median {d["sigma_median"]:.2f}  zcal {d["zcal"]:.2f}  q/dof median {d["q_median"]:.2f}  kin flag {d["kin_frac"]:.3f}'
                 f'  u median {d["u_median"]:.2f}')
    tag = '' if SCENE == 'pair0.31' else '_' + SCENE
    fn = f'{ROOT}/{name}/reuse/pair031_{F}_{name}{tag}'
    json.dump(out, open(fn + '.json', 'w'), indent=1, default=float); open(fn + '.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
