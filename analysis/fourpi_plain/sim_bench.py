"""Simulation benchmark of the plain-4Pi pipeline (sim first): a simulated single-frame movie of the acquisition's whole
channel crops from the training recipe's scene simulator (MultiChannelAcquisitionSim, patterns=False; the
acquisition's registration map, background, photons; a 4Pi phase field fixed over the movie with a slow drift; the
registration error of the recipe), processed by the real driver (simcode.processing.fourpi.process_file: detector,
patches, emitter network, phase field, explicit solve), evaluated against the ground truth: recall / false positives
per photon bin, x / y / z errors against the reported sigma (pull), z bias, and the 4Pi phase-field error.

  python analysis/fourpi_plain/sim_bench.py --acq simcode/pipeline/configs/fourpi_plain.py --work <plain-4Pi work folder> --data <extracted Zenodo folder> --out <dir> [--frames 600]
         [--dens 0.02,0.2] [--seed 11] [--set emitter=<ckpt> ...]
Output: <dir>/sim_movie.npz (frames ADU, GT; reused), <dir>/locs_sim.npz, <dir>/bench.json
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree


class ArrayReader:
    """ChannelReader stand-in over an in-memory movie [n, C, H, W] (ADU)"""
    def __init__(self, a): self.a, self.n = a, a.shape[0]

    def read(self, f0, f1): return self.a[f0:f1].astype(np.float32)


def simulate(c, frames, dens, seed, dev, B=25):
    from simcode.setups import get
    from simcode.simulate.multichannel import MultiChannelAcquisitionSim, CellSpec
    S_ = get(c['setup']); init, smp = S_.sim_kw(c)
    sp = S_.cell_spec(c)
    L = S_.layout(c)
    init['cells'] = [CellSpec(sp.name, sp.regmap, sp.W, sp.H, (0, L.W), (0, L.H), sp.bg)]   # scene = the whole crop (origin 0, 0)
    from simcode.psf import fourpi_state
    st = fourpi_state(c['fourpi']['psf'])
    sim = MultiChannelAcquisitionSim(st, H=L.H, W=L.W, seed=seed, dev=dev, **init)
    smp['dens'] = dens
    rng = np.random.default_rng(seed)
    cxy = rng.uniform(-1, 1, 4).tolist(); phi0 = rng.uniform(0, 2 * math.pi); drift = math.radians(rng.uniform(-60, 60)) / 1000   # rad / frame
    gain, off = c['camera']['gain_e_per_adu'], c['camera']['offset_adu']
    mov = np.zeros((frames, 4, L.H, L.W), np.float32); G = {k: [] for k in ('t', 'x', 'y', 'z', 'N', 'Phi', 'q')}
    for f0 in range(0, frames, B):
        b = min(B, frames - f0)
        S = sim.sample(B=b, det=False, phase=dict(phi_off=phi0 + drift * (f0 + b / 2), cxy=cxy), **smp)
        img = S['img'][:, 0]                                                                # [b,4,H,W] e, channel c at its crop origin
        for c_ in range(4):     # into the channel's real crop coordinates (integer origin)
            ox, oy = (int(v) for v in S['origin'][0, c_].tolist())
            fr = torch.full((b, L.H, L.W), float('nan'), device=dev)
            ys, xs = slice(max(0, oy), min(L.H, L.H + oy)), slice(max(0, ox), min(L.W, L.W + ox))
            fr[:, ys, xs] = img[:, c_, max(0, -oy):max(0, -oy) + (ys.stop - ys.start), max(0, -ox):max(0, -ox) + (xs.stop - xs.start)]
            bgm = S['bgmap'][:, 0, c_].mean((-1, -2))[:, None, None].expand_as(fr)        # outside the rendered area: background only
            fr = torch.where(torch.isnan(fr), torch.poisson(bgm.clamp(min=0)), fr)
            mov[f0:f0 + b, c_] = (fr / gain + off).cpu().numpy()
        g = S['gt']
        G['t'].append((g['b'] + f0).cpu().numpy()); G['x'].append(g['x'].cpu().numpy()); G['y'].append(g['y'].cpu().numpy())
        G['z'].append(g['z'].cpu().numpy()); G['N'].append(g['Nt'][:, 0].cpu().numpy()); G['Phi'].append(g['Phi'].cpu().numpy())
        G['q'].append(g['q'][:, 0].cpu().numpy())
    G = {k: np.concatenate(v) for k, v in G.items()}
    return mov, G


def isolated_interior(G, out_fr=None, r=3.0, margin=(8, 9)):
    """GT rows with no other emitter (> 50 photons) of the frame within r px, inside the processed field"""
    nn = np.full(len(G['x']), 99.0)
    for t in np.unique(G['t']):
        s = np.nonzero((G['t'] == t) & (G['N'] > 50))[0]
        if len(s) > 1:
            d, _ = cKDTree(np.c_[G['x'][s], G['y'][s]]).query(np.c_[G['x'][s], G['y'][s]], k=2); nn[s] = d[:, 1]
    x0, x1 = out_fr['fourpi']['xlim'] if out_fr else (0, 1e9)
    H = out_fr['channels'][0]['crop'][3] if out_fr else 1e9
    return (nn > r) & (G['x'] >= max(x0, margin[0])) & (G['x'] < x1 - 4) & (G['y'] >= margin[1]) & (G['y'] < H - margin[1])


def evaluate(L, G, px_nm, r=0.8):
    """per frame one-to-one nearest within r px: recall per photon bin, FP, errors, pulls"""
    gi, li = [], []
    for t in np.unique(G['t']):
        a = np.nonzero(G['t'] == t)[0]; b = np.nonzero(L['frame'] == t)[0]
        if len(a) == 0 or len(b) == 0: continue
        d, q = cKDTree(np.c_[L['x'][b], L['y'][b]]).query(np.c_[G['x'][a], G['y'][a]], distance_upper_bound=r)
        d2, q2 = cKDTree(np.c_[G['x'][a], G['y'][a]]).query(np.c_[L['x'][b], L['y'][b]], distance_upper_bound=r)
        for i in np.nonzero(np.isfinite(d))[0]:
            if q2[q[i]] == i: gi.append(a[i]); li.append(b[q[i]])
    gi, li = np.array(gi, int), np.array(li, int)
    out = dict(n_gt=int(len(G['x'])), n_loc=int(len(L['x'])), matched=int(len(gi)))
    bins = [0, 150, 500, 1000, 2000, 4000, 8000, 1e9]
    rows = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        s = (G['N'] >= b0) & (G['N'] < b1); sm = (G['N'][gi] >= b0) & (G['N'][gi] < b1)
        if s.sum() < 20: continue
        ex = (L['x'][li] - G['x'][gi])[sm] * px_nm; ey = (L['y'][li] - G['y'][gi])[sm] * px_nm; ez = (L['z'][li] - G['z'][gi])[sm] * 1000
        rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 10 else None
        rows.append(dict(N=[b0, b1], n_gt=int(s.sum()), recall=float(sm.sum() / s.sum()), rms_x=rs(ex), rms_y=rs(ey), rms_z=rs(ez),
                         bias_z=float(np.median(ez)) if len(ez) > 10 else None,
                         sig_x=float(np.median(L['sx'][li][sm]) * px_nm) if sm.any() else None,
                         sig_z=float(np.median(L['sz'][li][sm]) * 1000) if sm.any() else None,
                         pull_x=rs(ex / (L['sx'][li][sm] * px_nm)), pull_y=rs(ey / (L['sy'][li][sm] * px_nm)), pull_z=rs(ez / (L['sz'][li][sm] * 1000)),
                         frac_z_gt_100nm=float(np.mean(np.abs(ez) > 100)) if len(ez) else None))
    out['bins'] = rows
    out['fp'] = int(len(L['x']) - len(li))
    unm = np.ones(len(L['x']), bool); unm[li] = False
    out['fp_good'] = int((unm & L['good']).sum()); out['n_good'] = int(L['good'].sum())
    return out, gi, li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acq', required=True); ap.add_argument('--out', required=True); ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--dens', default='0.02,0.2'); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--set', nargs='*', default=[], help='overrides of the fourpi section: key=value (json values)')
    ap.add_argument('--sim_set', nargs='*', default=[], help='overrides of the simulation recipe (fourpi_train): key=value, e.g. n_psf=1 reg_shift_sd=0')
    ap.add_argument('--sim_dev', default='cuda', help="device of the movie simulation ('cpu': GPU-free)"); ap.add_argument('--sim_only', type=int, default=0)
    ap.add_argument('--work', required=True, help='work folder of the config (configs(work, data))')
    ap.add_argument('--data', required=True, help='data folder of the config (the extracted Zenodo data)')
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fourpi'))
    from train_det4pi import acquisition, setup_of
    from simcode.processing import fourpi as PF
    c = acquisition(a.acq, a.work, a.data)
    for kv in a.set:
        k, v = kv.split('=', 1); c['fourpi'][k] = json.loads(v)
    for kv in a.sim_set:
        k, v = kv.split('=', 1); c['fourpi_train'][k] = json.loads(v)
    os.makedirs(a.out, exist_ok=True)
    fm = os.path.join(a.out, 'sim_movie.npz')
    if os.path.exists(fm):
        z = np.load(fm); mov = z['mov']; G = {k[3:]: z[k] for k in z.files if k.startswith('gt_')}
    else:
        t0 = time.time()
        if a.sim_dev == 'cpu': torch.set_num_threads(16)
        mov, G = simulate(c, a.frames, tuple(float(v) for v in a.dens.split(',')), a.seed, a.sim_dev)
        np.savez(fm, mov=mov, **{f'gt_{k}': v for k, v in G.items()})
        print(f'simulated {a.frames} frames, {len(G["x"])} emitter-frames in {time.time() - t0:.0f}s', flush=True)
    if a.sim_only: return
    S = setup_of(c)
    ctx = PF.context(c)
    fr = S.front(c, 0, 'cuda')
    st = PF.settings(c)
    out, fld, summ = PF.process_file(ArrayReader(mov), fr, ctx, st, 0, mov.shape[0])
    np.savez(os.path.join(a.out, 'locs_sim.npz'), **out)
    # 4Pi phase-field error at the GT emitters (the solved Phi0 of the matched localizations)
    res, gi, li = evaluate(out, G, c['optics']['pixel_um'] * 1000)
    # the same on isolated interior GT emitters (no other on-emitter within 3 px; NPC-like rings put several emitters
    # within 1 px in one frame, which no single-frame method separates)
    iso = isolated_interior(G, out_fr=c)
    Gi = {k: v[iso] for k, v in G.items()}
    res['isolated'], gi2, li2 = evaluate(out, Gi, c['optics']['pixel_um'] * 1000)
    # sigma calibration: robust pull (error / reported sigma) of the matched isolated emitters with >= 1000 photons and
    # chi2 'good' -> the factors fourpi.sig_scale that make the pulls 1 (divide by the current sig_scale)
    s = (Gi['N'][gi2] >= 1000) & out['good'][li2]
    rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))
    pulls = [rs((out[k][li2] - Gi[k][gi2])[s] / out[sk][li2][s]) for k, sk in (('x', 'sx'), ('y', 'sy'), ('z', 'sz'))]
    res['sig_scale_fit'] = dict(n=int(s.sum()), pull=pulls, current=c['fourpi']['sig_scale'],
                                sig_scale=[p * q for p, q in zip(pulls, c['fourpi']['sig_scale'])])
    dphi = np.angle(np.exp(1j * (out['Phi0'][li] - G['Phi'][gi])))
    res['phase_field'] = dict(offset_deg=float(np.degrees(np.angle(np.mean(np.exp(1j * dphi))))), R=float(np.abs(np.mean(np.exp(1j * dphi)))),
                              sd_deg=float(np.degrees(np.std(np.angle(np.exp(1j * (dphi - np.angle(np.mean(np.exp(1j * dphi))))))))))
    sg = out['good'][li]
    res['good'] = dict(matched_good=int(sg.sum()), frac_matched_good=float(sg.mean()) if len(sg) else None)
    res['summary'] = summ
    json.dump(res, open(os.path.join(a.out, 'bench.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'summary'}, indent=1, default=float))


if __name__ == '__main__':
    main()
