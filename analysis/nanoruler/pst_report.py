"""
Part 1 numbers from fits.pt: held-out comparison of the pattern models, per-step drift curves, step-relative drift (what a
shared drift per orientation cannot represent) with uncertainty from fits on disjoint halves of the binding sites, phase steps
over time; writes the per-step pattern file.
Outputs: out/nanoruler/perstep/fit_summary.json, curves.npz, pattern_perstep.pt
usage: python analysis/nanoruler/pst_report.py <chosen per-step config> [<shared reference config>]
"""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L

PITCH = np.array([219.9, 219.9, 219.9, 219.5, 219.5, 219.5]); NM = PITCH / (2 * math.pi)
FR = np.arange(0, L.NFRAMES, 6)                      # one sample per cycle


def rel_of(Dm):
    """step-relative part [6, n] of total drift curves [6, n] (orientation mean removed), and the orientation mean [2, n]."""
    c = np.stack([Dm[:3].mean(0), Dm[3:].mean(0)])
    return Dm - np.repeat(c, 3, 0), c


def curves(state):
    P = L.Pattern.from_state(state)
    Dm = P.total_drift_np(FR)
    return P, Dm


if __name__ == '__main__':
    chosen = sys.argv[1]; shared_ref = sys.argv[2] if len(sys.argv) > 2 else 'shared_1500'
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    D = L.load_data(); n = len(D['I'])
    R = dict(n_data=n, n_samples=int(D['B']), n_tracks=int(D['ntrack']), n_sites=int(D['nsite']), chosen=chosen, shared_ref=shared_ref)

    # ---------------- held-out robust residual (pattern from one half of the sites, amplitudes refitted on the other half)
    splits = sorted({k[0] for k in F[shared_ref]['halves']})
    held = {}
    site_sel = {}
    for sp in splits:
        h0 = L.site_halves(D, seed=sp)
        for h in (0, 1):
            sel = h0 if h == 0 else ~h0                 # training half; held-out = ~sel
            site_sel[(sp, h)] = D['site'][~sel]
    rng = np.random.default_rng(0)
    base = F[shared_ref]['halves']
    for name, e in F.items():
        tot = 0; dof = 0; dsum = []; z2 = 0; mad = []
        for key, o in e['halves'].items():
            tot += o['held_huber'].sum(); dof += o['held_dof']; z2 += float((o['held_z'].astype(np.float64) ** 2).sum())
            mad.append(1.4826 * np.median(np.abs(o['held_z'])))
            # paired difference with the shared reference on the same held-out data, summed per site
            dd = (o['held_huber'].astype(np.float64) - base[key]['held_huber'].astype(np.float64))
            st = site_sel[key]; dsum.append(np.bincount(st, dd, D['nsite']))
        ds = np.concatenate(dsum)                        # per (split-half, site) loss difference
        bs = [ds[rng.integers(0, len(ds), len(ds))].sum() for _ in range(400)]
        held[name] = dict(model=e['cfg'][0], knot_frames=e['cfg'][1], knot_frames2=e['cfg'][2],
                          n_pattern_params=int(18 + (2 if e['cfg'][0] != 'perstep' else 0) * e['full']['state']['eps'].shape[1]
                                               + (6 if e['cfg'][0] != 'shared' else 0) * e['full']['state']['d'].shape[1]),
                          train_huber_per_datum=float(e['full']['train_loss']),
                          held_huber_per_dof=float(tot / dof), held_chi2_per_dof=float(z2 / dof), held_mad_z=float(np.mean(mad)),
                          # relative change of the held-out robust loss versus the shared reference (negative = better), +- site bootstrap
                          held_rel_change_vs_ref=float(ds.sum() / sum(o['held_huber'].sum() for o in base.values())),
                          held_rel_change_se=float(np.std(bs) / sum(o['held_huber'].sum() for o in base.values())))
    R['heldout'] = held
    print(f'{"config":18s} {"npar":>5s} {"train":>8s} {"held/dof":>9s} {"rel change vs " + shared_ref:>26s}')
    for name, h in held.items():
        print(f'{name:18s} {h["n_pattern_params"]:5d} {h["train_huber_per_datum"]:8.4f} {h["held_huber_per_dof"]:9.4f} '
              f'{100 * h["held_rel_change_vs_ref"]:+8.3f} +- {100 * h["held_rel_change_se"]:.3f} %')

    # ---------------- drift curves and step-relative drift for every per-step type config
    C = {}; npz = dict(frames=FR)
    for name, e in F.items():
        P, Dm = curves(e['full']['state']); npz[f'{name}__D'] = Dm
        Dh = {key: curves(o['state']) for key, o in e['halves'].items()}
        for key, (Ph, Dmh) in Dh.items(): npz[f'{name}__D_split{key[0]}_half{key[1]}'] = Dmh
        c = dict(phi=P.phi.tolist(), m=P.m.tolist(), relint=np.exp(P.logrel.numpy() - P.logrel.numpy().mean()).tolist(), k=P.k.tolist(),
                 pitch_nm=(2 * math.pi / np.linalg.norm(P.k.numpy(), axis=1) * L.PX_NM).tolist())
        # parameter scatter between the half fits (sd of the 4 half fits / sqrt(2) ~ se of the full fit)
        hs = lambda f: (np.std([f(Ph) for Ph, _ in Dh.values()], axis=0) / math.sqrt(2)).tolist()
        c['phi_se'] = hs(lambda Q: Q.phi.numpy() + 0); c['m_se'] = hs(lambda Q: Q.m.numpy())
        c['relint_se'] = hs(lambda Q: np.exp(Q.logrel.numpy() - Q.logrel.numpy().mean()))
        c['total_drift_rms_rad'] = Dm.std(1).tolist(); c['total_drift_p2p_rad'] = np.ptp(Dm, 1).tolist()
        if e['cfg'][0] != 'shared':
            rel, com = rel_of(Dm)
            rel = rel - rel.mean(1, keepdims=True)
            cross = []; noise = []; p2p_h = []
            for sp in splits:
                r0 = rel_of(Dh[(sp, 0)][1])[0]; r1 = rel_of(Dh[(sp, 1)][1])[0]
                r0 = r0 - r0.mean(1, keepdims=True); r1 = r1 - r1.mean(1, keepdims=True)
                cross.append((r0 * r1).mean(1)); noise.append(((r0 - r1) ** 2).mean(1) / 4)       # noise variance of the full fit
                p2p_h.append(np.ptp((r0 + r1) / 2, 1))
            cross = np.mean(cross, 0); noise = np.mean(noise, 0)
            rms_full = rel.std(1)
            c['rel'] = dict(
                rms_rad_full_fit=rms_full.tolist(), rms_nm_full_fit=(rms_full * NM).tolist(),
                # unbiased: covariance of the curves of two fits on disjoint halves (noise of the halves is independent)
                rms_rad_cross_half=(np.sign(cross) * np.sqrt(np.abs(cross))).tolist(),
                rms_nm_cross_half=(np.sign(cross) * np.sqrt(np.abs(cross)) * NM).tolist(),
                noise_rms_rad_full_fit=np.sqrt(noise).tolist(), noise_rms_nm_full_fit=(np.sqrt(noise) * NM).tolist(),
                p2p_rad_full_fit=np.ptp(rel, 1).tolist(), p2p_nm_full_fit=(np.ptp(rel, 1) * NM).tolist(),
                p95_range_rad=(np.percentile(rel, 97.5, 1) - np.percentile(rel, 2.5, 1)).tolist(),
                p95_range_nm=((np.percentile(rel, 97.5, 1) - np.percentile(rel, 2.5, 1)) * NM).tolist(),
                # expected peak-to-peak of pure noise with the measured noise rms is not zero: p2p of the half-difference / 2
                p2p_rad_noise_only=np.mean([np.ptp((rel_of(Dh[(sp, 0)][1])[0] - rel_of(Dh[(sp, 1)][1])[0]) / 2, 1) for sp in splits], 0).tolist(),
                corr_between_halves=np.mean([[np.corrcoef(rel_of(Dh[(sp, 0)][1])[0][s], rel_of(Dh[(sp, 1)][1])[0][s])[0, 1] for s in range(6)]
                                             for sp in splits], 0).tolist(),
                # all three steps of an orientation pooled
                rms_rad_cross_half_per_orientation=[float(np.sqrt(max(cross[:3].mean(), 0))), float(np.sqrt(max(cross[3:].mean(), 0)))],
                rms_nm_cross_half_per_orientation=[float(np.sqrt(max(cross[:3].mean(), 0)) * NM[0]), float(np.sqrt(max(cross[3:].mean(), 0)) * NM[3])])
            c['common_drift_rms_rad'] = com.std(1).tolist(); c['common_drift_p2p_rad'] = np.ptp(com, 1).tolist()
            # phase steps over time: phi_{s+1}(t) - phi_s(t)
            ph = P.phi.numpy()[:, None] + Dm
            st = np.stack([ph[1] - ph[0], ph[2] - ph[1], ph[4] - ph[3], ph[5] - ph[4]])
            sth = []
            for key, (Ph, Dmh) in Dh.items():
                q = Ph.phi.numpy()[:, None] + Dmh; sth.append(np.stack([q[1] - q[0], q[2] - q[1], q[4] - q[3], q[5] - q[4]]))
            sth = np.array(sth)
            wr = lambda v: (v + math.pi) % (2 * math.pi) - math.pi
            c['phase_steps'] = dict(order=['o0: s1-s0', 'o0: s2-s1', 'o1: s4-s3', 'o1: s5-s4'],
                                    mean_rad=wr(st.mean(1)).tolist(), mean_deg=np.rad2deg(wr(st.mean(1))).tolist(),
                                    mean_se_rad=(wr(sth.mean(2)).std(0) / math.sqrt(2)).tolist(),
                                    sd_over_time_rad=st.std(1).tolist(), min_rad=wr(st.min(1)).tolist(), max_rad=wr(st.max(1)).tolist(),
                                    p2p_rad=np.ptp(st, 1).tolist(), p2p_deg=np.rad2deg(np.ptp(st, 1)).tolist())
            npz[f'{name}__steps'] = st
        else:
            ph = P.phi.numpy()
            c['phase_steps'] = dict(mean_rad=[float(ph[1] - ph[0]), float(ph[2] - ph[1]), float(ph[4] - ph[3]), float(ph[5] - ph[4] - 2 * math.pi if ph[5] - ph[4] > math.pi else ph[5] - ph[4])])
        C[name] = c
    R['configs'] = C
    np.savez(L.OUT + 'curves.npz', **npz)

    # ---------------- per-step pattern file (chosen config), on one knot grid
    e = F[chosen]; P = L.Pattern.from_state(e['full']['state']); kf = e['cfg'][1]
    nk = (L.NFRAMES - 1) // kf + 2; knots = np.arange(nk) * kf
    Dk = P.total_drift_np(knots)                                               # [6, nk] total per-step drift at the knots
    gm = Dk.mean(1, keepdims=True)                                             # gauge: knot mean per step into phi
    relk, comk = rel_of(Dk - gm)
    S = F[shared_ref]['full']['state']
    t32 = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32)
    out = dict(
        format='per-step pattern: illumination of stored frame f (step s = f % 6, orientation a = s // 3) = rel_s (1 + m_s cos(k_a . r '
               '- phi_s - dphi_step[s](f))), r = (x = column, y = row) in movie px (65 nm), rel = exp(logrel - mean(logrel)), dphi_step '
               '[6, K] linear interpolation on frames j * knot_frames (last value held), mean 0 per step over the knots. '
               'dphi_shared [2, K] = mean of the three steps of the orientation, dphi_rel [6, K] = dphi_step - dphi_shared[a]. '
               '"pipeline" = the same pattern in the common pattern-file format read by simcode.processing.common.pattern_frames '
               '(every step its own "angle": k [6, 2], dphi [6, K], angle_of_step = [0..5]).',
        source=f'{chosen} fit of out/nanoruler/psf/bl_int_intensities.pt (analysis/nanoruler/pst_fit.py)', model=e['cfg'][0],
        k=t32(P.k.numpy()), phi=t32(P.phi.numpy() + gm[:, 0]), m=t32(P.m.numpy()), logrel=t32(P.logrel.numpy()),
        dphi_step=t32(Dk - gm), dphi_shared=t32(comk), dphi_rel=t32(relk), knot_frames=int(kf),
        pitch_nm=t32(2 * math.pi / np.linalg.norm(P.k.numpy(), axis=1) * L.PX_NM), px_nm=L.PX_NM,
        shared_reference=dict(config=shared_ref, k=S['k'].float(), phi=S['phi'].float(), m=S['m'].float(), logrel=S['logrel'].float(),
                              dphi=S['eps'].float(), knot_frames=int(S['knot_frames'])))
    out['pipeline'] = dict(k=t32(np.repeat(P.k.numpy(), 3, 0)), phi=out['phi'].clone(), m=out['m'].clone(), logrel=out['logrel'].clone(),
                           dphi=out['dphi_step'].clone(), knot_frames=int(kf), angle_of_step=[0, 1, 2, 3, 4, 5],
                           note='per-step drift: each step is its own "angle" for pattern_frames (k repeated). depth per step.')
    torch.save(out, L.OUT + 'pattern_perstep.pt')
    # check: processing.common.pattern_frames on the 'pipeline' dict reproduces the model phases
    from simcode.processing.common import pattern_frames
    ft = np.random.default_rng(0).integers(0, L.NFRAMES, 2000)
    PF = pattern_frames(out['pipeline'], ft, 0.0, 0.0, dev='cpu').numpy()
    with torch.no_grad():
        ref = (P.phi[torch.as_tensor(ft % 6)] + P.drift(torch.as_tensor(ft), torch.as_tensor(ft % 6))).numpy()
    R['pattern_file_check'] = dict(max_abs_phase_diff_rad=float(np.abs(PF[:, 2] - ref).max()),
                                   max_abs_k_diff=float(np.abs(PF[:, :2] - P.k.numpy()[(ft % 6) // 3]).max()))
    print('pattern_frames check', R['pattern_file_check'])
    json.dump(R, open(L.OUT + 'fit_summary.json', 'w'), indent=1)
    for name in C:
        if 'rel' in C[name]:
            r = C[name]['rel']
            print(f'{name:18s} rel rms nm (cross-half) {np.round(r["rms_nm_cross_half"], 2).tolist()} full {np.round(r["rms_nm_full_fit"], 2).tolist()} '
                  f'noise {np.round(r["noise_rms_nm_full_fit"], 2).tolist()} p2p nm {np.round(r["p2p_nm_full_fit"], 1).tolist()} corr {np.round(r["corr_between_halves"], 2).tolist()}')
            print(f'{"":18s} steps mean {np.round(C[name]["phase_steps"]["mean_rad"], 3).tolist()} +- {np.round(C[name]["phase_steps"]["mean_se_rad"], 3).tolist()} p2p {np.round(C[name]["phase_steps"]["p2p_rad"], 3).tolist()}')
