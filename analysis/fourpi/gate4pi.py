"""4Pi gate suite (analogue of simcode2/hd/gate.py): fixed-seed full-scene benchmarks of the 4Pi front end + solve.

Scenes (simcode2/fourpi/realsim4pi.py, 64 x 64 px, real registration, PSF jitter bank, per-angle offset fixed -12 deg):
  iso    : uniform emitters, 0.02 ON /um^2/frame
  pair   : close pairs 0.3-3 px apart, dz up to +-0.3 um, 0.1 /um^2/frame
  dense  : mixed structures, 0.3-1.0 /um^2/frame
  ring   : NPC-like rings, 0.2 /um^2/frame
Seeds: 202 = tuning (sigma calibration only), 301-303 = evaluation (pre-registered, PLAN_FULL_PIPELINE.md section 5).
Front end: --det <det4pi checkpoint> or 'gt' (GT-jitter candidates, sd 0.35 px); head: v2 / v3 (Head4Pi) or p1
(Head4PiP1, with neighbour / detector inputs). Solve: production Solve4Pi with the scene's TRUE pattern and 4Pi phase field
(network + solve test; the in-situ estimates are tested on the simulated movies), kinetics (two starts) vs free.
Metrics per GT photon bin (window photons): recall (match within 1 px), precision, robust and RMS (no 4Pi fringe error)
errors x / y / z, 4Pi fringe errors (|dz| > 100 nm), lateral SIM-fringe hops (> 60 nm); CRLB = single-emitter pixel CRLB
with the true per-frame photons (kinetics and pattern included); for pairs / dense it is the isolated bound (a lower bound).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/gate4pi.py --head v3 --det gt --seeds 301,302,303 --tag v3_gt
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from train_head4pi_p1 import candidates, build_inputs
from simcode2.fourpi.realsim4pi import RealSim4Pi, RegMap
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, kin_starts, best_of
from simcode2.fourpi.lm import crlb

GATES = {'iso': dict(kind='uniform', dens=(0.02, 0.02)), 'pair': dict(kind='pair', dens=(0.1, 0.1)),
         'dense': dict(kind=None, dens=(0.3, 1.0)), 'ring': dict(kind='ring', dens=(0.2, 0.2))}
EDGES = (300, 1000, 2000, 4000, 8000, 30000)
DELTA = -12.0


def load_head(tag, dev):
    from simcode2.fourpi.head4pi import Head4Pi
    from simcode2.fourpi.head4pi_p1 import Head4PiP1
    f = f'{REPO}/out/4pi/models/head4pi_{tag}.pt' if not tag.endswith('.pt') else tag
    ck = torch.load(f, weights_only=False, map_location=dev)
    st = ck['state']
    if any(k.startswith('out.') for k in st) and st['out.weight'].shape[0] > 150:
        a = ck['args']; init = torch.load(f'{REPO}/{a["init"]}', weights_only=False)
        net = Head4PiP1(ch=init['args']['ch'], nblk=init['args']['nblk'])
        p1 = True
    else:
        net = Head4Pi(ch=ck['args']['ch'], nblk=ck['args']['nblk']); p1 = False
    net.load_state_dict(st); return net.to(dev).eval(), p1


def crlb_units(G, lut, idx, dev, H, W):
    """single-emitter pixel CRLB (x, y, z) of GT units idx with the true per-frame photons (kinetics + pattern)"""
    from simcode2.fourpi.lm import fisher_j
    ang = [0, 0, 0, 1, 1, 1]; stp = [0, 1, 2, 0, 1, 2]
    out = []
    for i0 in range(0, len(idx), 1000):
        j = idx[i0:i0 + 1000]
        n = len(j)
        lx, ly = G['lx'][j], G['ly'][j]                                   # [n,4] channel-tile coordinates
        X, Y, z = G['X'][j], G['Y'][j], G['z'][j]
        Nt = G['Nt'][j]; k = G['k'][j]; psi = G['psi'][j]; m = G['m'][j]; Phi = G['Phi'][j]; dl = G['delta'][j]
        dlog = []
        for t in range(6):
            a = ang[t]; arg = k[:, a, 0] * X + k[:, a, 1] * Y - psi[:, a, stp[t]]
            Pv = 1 + m * torch.cos(arg)
            dlog.append(torch.stack([-m * k[:, a, 0] * torch.sin(arg) / Pv, -m * k[:, a, 1] * torch.sin(arg) / Pv], -1))
        dlog = torch.stack(dlog, 1)
        bg = torch.full((n, 4), 0.8, device=dev)

        def fj(th):
            dx, dy, zz, s = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
            mus, Js = [], []
            for c in range(4):
                pos = torch.stack([lx[:, c] - torch.round(lx[:, c]) + dx, ly[:, c] - torch.round(ly[:, c]) + dy, zz], 1)
                Gc = [lut.evaluate_ch_grad(pos, Phi + (dl if a else 0.0), c) for a in range(2)]
                mt, Jt = [], []
                for t in range(6):
                    g, gx, gy, gz, _ = Gc[ang[t]]
                    Ns = (s * Nt[:, t])[:, None, None]
                    mt.append(Ns * g + th[:, 4 + c, None, None])
                    jb = [torch.full_like(g, 1.0 if cc == c else 0.0) for cc in range(4)]
                    Jt.append(torch.stack([Ns * (gx + dlog[:, t, 0, None, None] * g), Ns * (gy + dlog[:, t, 1, None, None] * g),
                                           Ns * gz, Nt[:, t, None, None] * g] + jb, -1))
                mus.append(torch.stack(mt, 1)); Js.append(torch.stack(Jt, 1))
            return torch.stack(mus, 2).reshape(n, -1), torch.stack(Js, 2).reshape(n, -1, 8)
        th = torch.cat([torch.zeros(n, 2, device=dev), z[:, None], torch.ones(n, 1, device=dev), bg], 1)
        out.append(crlb(fisher_j(fj, th, 1.14 ** 2))[:, :3])
    return torch.cat(out) if out else torch.zeros(0, 3, device=dev)


def run_gate(name, seed, n_batches, head, p1, det, lut_solve, sv, reg, rs_state, dev):
    cfg = GATES[name]
    rs = RealSim4Pi(rs_state, seed=seed, n_psf=6)
    rows = []
    for ib in range(n_batches):
        S = rs.sample(B=16, dens=cfg['dens'], kind=cfg['kind'], delta_deg=DELTA)
        o = dict(img=S['img'], origin=S['origin'], ox=S['ox'], oy=S['oy'], det_in=S['det_in'], cell=S['cell'], gt=S['gt'])
        b, cx, cy, p, pmap = candidates(o, det, dev)
        if len(b) == 0: continue
        pt, doff, extra = build_inputs(o, reg, b, cx, cy, p, pmap, dev)
        with torch.no_grad():
            outs = [head(pt[i:i + 2048], doff[i:i + 2048], extra[i:i + 2048]) if p1 else head(pt[i:i + 2048], doff[i:i + 2048])
                    for i in range(0, len(b), 2048)]
        out = {k: torch.cat([u[k] for u in outs]) for k in outs[0] if outs[0][k] is not None}
        sc = S['scene']; H, W = 64, 64
        X = cx.float() + S['ox'][b]; Y = cy.float() + S['oy'][b]
        kk = sc['k'][b]; ang = [0, 0, 0, 1, 1, 1]; stp = [0, 1, 2, 0, 1, 2]
        psi = torch.stack([sc['psi'][b][:, ang[t], stp[t]] - (kk[:, ang[t], 0] * X + kk[:, ang[t], 1] * Y) for t in range(6)], 1)
        m = sc['m'][b]
        Phi0 = RealSim4Pi.phase_at(sc, b, cx.float() + out['dxy'][:, 0], cy.float() + out['dxy'][:, 1], H, W)
        # main-axis k per angle (local pattern); per-scene k differs by <= 0.3 %: use the batch mean
        sv.k = (float(kk[:, 0, 0].mean()), float(kk[:, 1, 1].mean()))
        idx = torch.arange(len(b), device=dev)
        res = []
        for i in range(0, len(b), 4096):
            j = idx[i:i + 4096]; oo = {k: v[j] for k, v in out.items()}
            rf = sv.solve(oo, Phi0[j], doff[j], m[j], psi[j], mode='free')
            k1, k2 = kin_starts(rf, m[j], psi[j], sv)
            r1 = sv.solve(oo, Phi0[j], doff[j], m[j], psi[j], mode='kin', kin0=k1)
            r2 = sv.solve(oo, Phi0[j], doff[j], m[j], psi[j], mode='kin', kin0=k2)
            rk = best_of(r1, r2)
            use = rk['chi2'] - rf['chi2'] < 7.81
            th = torch.where(use[:, None], rk['th'][:, :3], rf['th'][:, :3])
            res.append(dict(th=th, use=use))
        th = torch.cat([r['th'] for r in res]); use = torch.cat([r['use'] for r in res])
        xs, ys, zs = cx.float() + th[:, 0], cy.float() + th[:, 1], th[:, 2]
        G = S['gt']
        act = (G['q'].sum(1) >= 1) & (G['Nt'].sum(1) >= 60) & (G['x'] >= 8) & (G['x'] < 56) & (G['y'] >= 8) & (G['y'] < 56)
        gi = torch.nonzero(act)[:, 0]
        # greedy match per scene within 1 px on the final positions
        d2 = (xs[:, None] - G['x'][gi][None]) ** 2 + (ys[:, None] - G['y'][gi][None]) ** 2
        d2 = torch.where(b[:, None] == G['b'][gi][None], d2, torch.full_like(d2, 1e9))
        mg = torch.full((len(gi),), -1, device=dev, dtype=torch.long)
        if d2.numel():
            D = d2.clone()
            for _ in range(min(D.shape)):
                v, flat = D.min(), torch.argmin(D)
                if v > 1.0: break
                i_, j_ = flat // D.shape[1], flat % D.shape[1]
                mg[j_] = i_; D[i_, :] = 1e9; D[:, j_] = 1e9
        cr = crlb_units(G, lut_solve, gi, dev, H, W)
        rows.append(dict(N=G['Nt'][gi].sum(1), mg=mg, gx=G['x'][gi], gy=G['y'][gi], gz=G['z'][gi], cr=cr,
                         xs=xs, ys=ys, zs=zs, use=use, n_loc=len(b)))
    return rows


def summarise(rows, name):
    N = torch.cat([r['N'] for r in rows]); cr = torch.cat([r['cr'] for r in rows])
    ex, ey, ez, found, usek = [], [], [], [], []
    nloc = sum(r['n_loc'] for r in rows); nmatch = 0
    for r in rows:
        mm = r['mg'] >= 0; nmatch += int(mm.sum())
        e = torch.full((len(r['mg']), 3), float('nan'), device=r['N'].device)
        j = r['mg'][mm]
        e[mm, 0] = (r['xs'][j] - r['gx'][mm]) * 128; e[mm, 1] = (r['ys'][j] - r['gy'][mm]) * 128; e[mm, 2] = (r['zs'][j] - r['gz'][mm]) * 1000
        u = torch.zeros(len(r['mg']), dtype=torch.bool, device=e.device); u[mm] = r['use'][j]
        ex.append(e); found.append(mm); usek.append(u)
    e = torch.cat(ex).cpu().numpy(); f = torch.cat(found).cpu().numpy(); uk = torch.cat(usek).cpu().numpy()
    N = N.cpu().numpy(); cr = cr.cpu().numpy() * np.array([128, 128, 1000])
    out = dict(gate=name, n_gt=int(len(N)), n_loc=int(nloc), precision=float(nmatch / max(nloc, 1)), bins=[])
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (N >= lo) & (N < hi)
        if s.sum() < 30: continue
        sm = s & f
        rb = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 10 else None
        nf = sm & (np.abs(e[:, 2]) <= 100)
        rm = lambda k: float(np.sqrt(np.mean(e[nf, k] ** 2))) if nf.sum() > 10 else None
        crm = lambda k: float(np.sqrt(np.mean(cr[sm, k] ** 2))) if sm.sum() > 10 else None
        out['bins'].append(dict(lo=lo, hi=hi, n=int(s.sum()), recall=float(f[s].mean()), kin_frac=float(uk[sm].mean()) if sm.sum() else None,
                                robust=[rb(e[sm, k]) for k in range(3)], rms=[rm(k) for k in range(3)], crlb=[crm(k) for k in range(3)],
                                rms_over_crlb=[(rm(k) / crm(k)) if rm(k) and crm(k) else None for k in range(3)],
                                fringe=float(np.mean(np.abs(e[sm, 2]) > 100)) if sm.sum() else None,
                                lat_hop=float(np.mean(np.abs(e[sm, :2]) > 60)) if sm.sum() else None,
                                median=[float(np.median(e[sm, k])) if sm.sum() > 10 else None for k in range(3)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', default='v3'); ap.add_argument('--det', default='gt'); ap.add_argument('--tag', default='')
    ap.add_argument('--seeds', default='301,302,303'); ap.add_argument('--gates', default='iso,pair,dense,ring')
    ap.add_argument('--n_batches', type=int, default=6); ap.add_argument('--psf', default='NPC_f_dxy15')
    a = ap.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-gate')
    dev = 'cuda'
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    head, p1 = load_head(a.head, dev)
    det = None
    if a.det != 'gt':
        from simcode2.det.up2 import load as dload
        det, _ = dload(a.det)
    lut = LUT4Pi(st, P=15, dev=dev)
    sv = Solve4Pi(lut, delta=(0.0, math.radians(DELTA))); sv.build_table()
    reg = {c: RegMap(c, dev) for c in ('MT', 'NPC')}
    res = {}
    for gname in a.gates.split(','):
        rows = []
        for sd in a.seeds.split(','):
            rows += run_gate(gname, int(sd), a.n_batches, head, p1, det, lut, sv, reg, st, dev)
        res[gname] = summarise(rows, gname)
        r = res[gname]
        print(f"\n{gname}: GT {r['n_gt']}, locs {r['n_loc']}, precision {r['precision']:.3f}", flush=True)
        for bn in r['bins']:
            f = lambda v: '-' if v is None else f'{v:.2f}'
            print(f"  {bn['lo']}-{bn['hi']}: n {bn['n']} recall {bn['recall']:.2f} kin {f(bn['kin_frac'])} | robust x/y/z {'/'.join(f(v) for v in bn['robust'])} | "
                  f"RMS/CRLB {'/'.join(f(v) for v in bn['rms_over_crlb'])} | CRLB {'/'.join(f(v) for v in bn['crlb'])} | fringe {f(bn['fringe'])} hop {f(bn['lat_hop'])} | median {'/'.join(f(v) for v in bn['median'])}", flush=True)
    os.makedirs(f'{REPO}/out/4pi/gates', exist_ok=True)
    json.dump(res, open(f'{REPO}/out/4pi/gates/gate_{a.tag or a.head}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
