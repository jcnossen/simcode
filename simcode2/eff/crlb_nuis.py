"""Marginal (nuisance) multi-emitter CRLB for the gate scenes (task 4, 2026-11-09, sim realism worker).
Model = CRLB_SF_multi of simcode2/sim/crlb.crlb_all ('sf': per emitter x, y, z, B, t_on, t_off with the true
pattern; all active emitters within 9 px in the parameter set; one bg per angle; exact vectorial PSF; Poisson + read
noise; the same expected image), extended per emitter k with multiplicative intensity nuisances
    I_kt = B_k a_kt il_kt (1 + eps_kt) (1 + eta_k,a(t))
  eps_kt  per-frame factor, eps ~ N(0, c^2) independent per frame (covers realsim frame_sd and the per-event step
          factors es, which are independent across the 6 steps of one window); c = --c (default sqrt(0.10^2+0.07^2))
  eta_ka  per-angle factor, prior sd --ca (default 0.14 = realsim ang_sd). Our solve has NO free per-angle intensity
          (simcode2/hd/sc.py:216 solve_select_cov(mode='shared'); the PM path simcode2/hd/msolve.py:52
          free = [x, y, B, 0 (B_1 tied to B_0), t_on, t_off]) -> the per-angle factor is a nuisance with a prior, not a
          free parameter. (B free absorbs the common mode, so eta_0 - eta_1 has prior sd 0.14 sqrt 2 = realsim's
          normalized ea = exp(0.14 g) / mean.)
  Fisher: J_eps_kt = B a il psf_k (frame t only), J_eta_ka = B a il psf_k (frames of angle a); prior information
  1 / c^2, 1 / ca^2 on their diagonal; inverse restricted to the target x, y = Schur complement over every nuisance
  (marginal bound). Evaluated at eps = eta = 0 (same expected image as CRLB_SF_multi). c = ca = 0 -> columns dropped
  (= CRLB_SF_multi exactly).
Recomputed on the deterministic gate scenes (hd/gate.gen) for the targets of an existing gate npz (--ref tag; gt_x
checked equal); output <out>/gate/nuis/<scene>__s<seed>_c<c>_ca<ca>.npz: gt_x, gt_y, gt_phot, gt_target,
sf_m (recomputed, check), nu_m (nuisance bound) [n,2] px, NaN for non-targets.
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode2.eff.crlb_nuis --scen rs05qm --c 0.122 --ca 0.14
"""
import argparse, os, time
import numpy as np, torch
from simcode2.sim import simtools as ST
from simcode2.sim.simtools import CD, dev, S
from simcode2.nn.blocks import extract_patches
from simcode2.hd import gate as GT
from simcode2 import config

R, HR, BOXP, BOXM = CD.R, CD.HR, CD.BOXP, CD.BOXM


def crlb_nuis(G, o, psS, tg, T, c, ca):
    nb = o['nb']; Np = len(G['x'])
    pxi, pyi = torch.round(G['x']).long(), torch.round(G['y']).long()
    patt = o['patt']; ang_f = (patt[..., 0].abs() < patt[..., 1].abs()).long()
    pt_e = patt[G['fov']]
    il, dilx, dily = CD.illum_at(pt_e, G['x'], G['y'])
    a_, da0, da1 = CD.onfrac(G['t0'], G['t1'], T)
    Bv = G['bright']
    Ie = Bv[:, None] * a_ * il
    pad = BOXM; Sp = S + 2 * pad
    canvas = torch.zeros(nb, T, Sp, Sp, device=dev)
    oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
    lin = (G['fov'][:, None, None] * T * Sp * Sp + torch.arange(T, device=dev)[None, :, None] * Sp * Sp
           + ((pyi[:, None, None] + pad - HR + oy) * Sp + pxi[:, None, None] + pad - HR + ox).reshape(Np, 1, -1))
    canvas.view(-1).index_add_(0, lin.reshape(-1), (Ie[:, :, None] * psS[:, 0].reshape(Np, 1, -1)).reshape(-1))
    mu_full = canvas[:, :, pad:pad + S, pad:pad + S] + o['bg']
    rn2_full = (o['rn'] ** 2).expand(nb, 1, S, S)
    n = len(tg)
    tb, tcx, tcy = G['fov'][tg], pxi[tg], pyi[tg]
    mu_m = extract_patches(mu_full, tb, tcy, tcx, R).reshape(n, T, -1)
    rn2_r = extract_patches(rn2_full, tb, tcy, tcx, R).reshape(n, 1, -1)
    sets = []
    for i in range(n):
        j = tg[i]
        cand = torch.nonzero((G['fov'] == G['fov'][j]) & ((pxi - pxi[j]).abs() <= BOXP) & ((pyi - pyi[j]).abs() <= BOXP))[:, 0]
        cand = cand[cand != j]
        sets.append(torch.cat([j[None], cand]))
    Ks = torch.tensor([len(s_) for s_ in sets])
    angT = ang_f[tb]
    bgcol = torch.stack([(angT == 0).float(), (angT == 1).float()], 1)[..., None].expand(n, 2, T, R * R)
    use_e, use_a = c > 0, ca > 0
    q = 6 + (T if use_e else 0) + (2 if use_a else 0)
    res = dict(sf=torch.full((n, 2), float('nan'), device=dev), nu=torch.full((n, 2), float('nan'), device=dev))

    def run(idx, K):
        m_ = len(idx)
        S_ = torch.stack([torch.cat([sets[i], torch.full((K - len(sets[i]),), -1, dtype=torch.long, device=dev)]) for i in idx.tolist()])
        valid = (S_ >= 0).float(); Sc = S_.clamp(min=0); Kk = S_.shape[1]
        ps = CD.shift_region(psS[Sc], pxi[Sc] - tcx[idx][:, None], pyi[Sc] - tcy[idx][:, None]) * valid[:, :, None, None, None]
        ps = ps.reshape(m_, Kk, 4, 1, R * R)
        B_ = Bv[Sc][..., None]; A_ = a_[Sc]; IL = il[Sc]
        w = 1.0 / (mu_m[idx] + rn2_r[idx])
        BA = B_ * A_
        cols = [(BA * dilx[Sc])[..., None] * ps[:, :, 0] + (BA * IL)[..., None] * ps[:, :, 1],
                (BA * dily[Sc])[..., None] * ps[:, :, 0] + (BA * IL)[..., None] * ps[:, :, 2],
                (BA * IL)[..., None] * ps[:, :, 3],
                (A_ * IL)[..., None] * ps[:, :, 0],
                (B_ * da0[Sc] * IL)[..., None] * ps[:, :, 0],
                (B_ * da1[Sc] * IL)[..., None] * ps[:, :, 0]]
        sig = (BA * IL)[..., None] * ps[:, :, 0]                        # [m,K,T,RR] = dmu / d(log I_kt)
        if use_e:
            eye = torch.eye(T, device=dev)
            cols += [sig * eye[t][None, None, :, None] for t in range(T)]
        if use_a:
            angS = ang_f[G['fov'][Sc]]                                   # [m,K,T]
            cols += [sig * (angS == a)[..., None].float() for a in (0, 1)]
        J = torch.stack(cols, 2) * valid[:, :, None, None, None]        # [m,K,q,T,RR]
        J = torch.cat([J.reshape(m_, Kk * q, T, R * R), bgcol[idx]], 1).reshape(m_, Kk * q + 2, -1).double()
        FI = torch.einsum('npk,nqk->npq', J * w.reshape(m_, 1, -1).double(), J)
        del J
        out = {'sf': None}
        # plain SF bound: the 6 base params per emitter + bg
        base = torch.tensor([k * q + j for k in range(Kk) for j in range(6)] + [Kk * q, Kk * q + 1], device=dev)
        out['sf'] = CD.scaled_solve(FI[:, base][:, :, base], [0, 1])
        pri = torch.zeros(q, device=dev, dtype=torch.float64)
        o_ = 6
        if use_e: pri[o_:o_ + T] = 1.0 / c ** 2; o_ += T
        if use_a: pri[o_:o_ + 2] = 1.0 / ca ** 2
        prv = torch.cat([pri.repeat(Kk), torch.zeros(2, device=dev, dtype=torch.float64)])
        # padded (invalid) emitters: their columns are zero; the prior keeps the nuisance block invertible
        FI = FI + torch.diag_embed(prv.expand(m_, -1))
        out['nu'] = CD.scaled_solve(FI, [0, 1])
        return out

    order = torch.argsort(Ks); i0 = 0
    while i0 < n:
        K = int(Ks[order[i0]]); npar = K * q + 2
        nbat = max(1, min(512, int(2e7 // (npar * T * R * R))))
        idx = order[i0:i0 + nbat]
        K = int(Ks[idx].max()); npar = K * q + 2
        nbat = max(1, min(len(idx), int(2e7 // (npar * T * R * R)))); idx = idx[:nbat].to(dev)
        out = run(idx, K)
        for k_ in ('sf', 'nu'):
            res[k_][idx] = out[k_].diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
        i0 += len(idx)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='rs05qm')
    ap.add_argument('--seeds', default='301,302,303')
    ap.add_argument('--c', type=float, default=float(np.hypot(0.10, 0.07)))
    ap.add_argument('--ca', type=float, default=0.14)
    ap.add_argument('--ref', default='_hd_sc5nb15pm')
    ap.add_argument('--gate', default=os.path.join(config.root('out'), 'gate'))
    a = ap.parse_args()
    od = f'{a.gate}/nuis'; os.makedirs(od, exist_ok=True)
    CD.a.calib = config.model(config.INSITU_05)
    T = 6
    sim, rpsf, lut = CD.models(T)
    for sc in a.scen.split(','):
        for seed in [int(s_) for s_ in a.seeds.split(',')]:
            t0 = time.time()
            REF = dict(np.load(f'{a.gate}/{sc}__s{seed}{a.ref}.npz'))
            nf = int(REF['n_fov']); fov0 = 0; it = 0; L = []
            while fov0 < nf:
                it += 1
                o = GT.gen(sc, it, seed); nb = o['nb']
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k_: v[act] for k_, v in G.items()}
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]; Gy = G['y'] + o['shift_est'][G['fov']][:, 1]
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                vis = (on >= 1) & (ph >= 100)
                tg = torch.nonzero(inter & vis)[:, 0]
                if len(tg) == 0: continue
                fovs = torch.unique(G['fov'][tg]); ai = torch.nonzero(torch.isin(G['fov'], fovs))[:, 0]
                psS = torch.zeros(len(G['x']), 4, R, R, device=dev)
                psS[ai] = CD.psf_stack(rpsf, G['x'][ai], G['y'][ai], G['z'][ai], sim.focus)
                Cb = crlb_nuis(G, o, psS, tg, T, a.c, a.ca); del psS
                nn = len(G['x'])
                full = lambda v: torch.full((nn, 2), float('nan'), device=dev).index_copy_(0, tg, v.float())
                L.append({k_: v.detach().float().cpu().numpy() for k_, v in dict(
                    x=Gx, y=Gy, phot=ph, target=(inter & vis).float(), sf_m=full(Cb['sf']), nu_m=full(Cb['nu'])).items()})
                fov0 += nb; torch.cuda.empty_cache()
            D = {'gt_' + k_: np.concatenate([r[k_] for r in L]) for k_ in L[0]}
            assert np.allclose(D['gt_x'], REF['gt_x'], atol=1e-4), 'scene mismatch vs the reference gate npz'
            m = np.isfinite(REF['gt_sf_m']).all(1) & np.isfinite(D['gt_sf_m']).all(1)
            dev_sf = np.nanmax(np.abs(D['gt_sf_m'][m] / REF['gt_sf_m'][m] - 1))
            fn = f'{od}/{sc}__s{seed}_c{a.c:.3f}_ca{a.ca:.3f}.npz'
            np.savez_compressed(fn, **D, c=a.c, ca=a.ca, ref=a.ref)
            r = np.hypot(*D['gt_nu_m'][m].T) / np.hypot(*D['gt_sf_m'][m].T)
            print(f'{sc} s{seed} c {a.c:.3f} ca {a.ca:.3f}: {m.sum()} targets, sf_m vs ref max rel dev {dev_sf:.2e}, '
                  f'nu/sf median {np.median(r):.3f}, {time.time() - t0:.0f}s -> {fn}', flush=True)


if __name__ == '__main__':
    main()
