"""joint prior / intensity covariance (2026-11-17): calibration of an emitter network's joint (x, y, I_1..6) Gaussian on simulated
held-out batches (the pass-2 trainer's make_pattern_hold scenes of the model's PSF, seed 909; detector candidates matched to the
truth). Per GT-window-photon bin: the whitened joint residual z_J = L_J^-1 [true - prior; I_true - I] (L_J = [[diag(sx, sy), 0],
[C, L]], EmitterNet): the eigenvalues of its 8 x 8 sample covariance (all 1 for a calibrated covariance; robust: on rows with all
|z| < 6), the rms of the position components z_p and of the intensity components; the per-emitter canonical correlation between
the prior error and the intensity errors implied by the network (rho: the largest singular value of
Sigma_pp^-1/2 Sigma_pI Sigma_II^-1/2), p10 / p50 / p90 per photon bin.
usage: python net_eval.py <out json> <calib json> <detector> <emitter ckpt> [<emitter ckpt> ...] [--pass1] [--n 40]"""
import json, sys
import numpy as np, torch

PH = [(0, 300), (300, 500), (500, 800), (800, 1200), (1200, 2000), (2000, 3000), (3000, 1e9)]


@torch.no_grad()
def collect(det, head, col, H, il_on):
    from simcode.emitter.model import detect_candidates, apply_to_candidates
    from simcode.simulate.targets import match_gt
    acc = {k: [] for k in ('Ig', 'I', 'L', 'C', 'ep', 'sp')}
    for o in H:
        il = o['illum_input'] if il_on else torch.zeros_like(o['illum_input'])
        c, nbr, lgm = detect_candidates(det, o['frames'], il)
        c5 = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
        m = match_gt(o, c5)
        if m is None: continue
        p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
        out = apply_to_candidates(head, col, o['frames'], il, m['b'], m['iy'], m['ix'], m['x'], m['y'], nbr, p_own)
        G = o['gt']; g = m['g']
        acc['Ig'].append(G['phot'][g].float()); acc['I'].append(out['I'].float()); acc['L'].append(out['L'].float()); acc['C'].append(out['C'].float())
        acc['ep'].append(torch.stack([G['x'][g] - out['x'], G['y'][g] - out['y']], 1).float()); acc['sp'].append(out['lsxy'].exp().float())
    return {k: torch.cat(v) for k, v in acc.items()}


def metrics(A):
    Ig, I, L, Cj, ep, sp = (A[k] for k in ('Ig', 'I', 'L', 'C', 'ep', 'sp'))
    n, T = I.shape
    LJ = torch.zeros(n, T + 2, T + 2, device=I.device)
    LJ[:, 0, 0] = sp[:, 0]; LJ[:, 1, 1] = sp[:, 1]; LJ[:, 2:, :2] = Cj; LJ[:, 2:, 2:] = L
    e = torch.cat([ep, Ig - I], 1)[..., None]
    z = torch.linalg.solve_triangular(LJ, e, upper=False)[..., 0]
    S = LJ @ LJ.transpose(1, 2)
    Spp, SpI, SII = S[:, :2, :2], S[:, :2, 2:], S[:, 2:, 2:]
    def isqrt(M):
        w, V = torch.linalg.eigh(M); return V @ torch.diag_embed(w.clamp(min=1e-9).rsqrt()) @ V.transpose(1, 2)
    K = isqrt(Spp) @ SpI @ isqrt(SII)
    rho = torch.linalg.svdvals(K)[:, 0]
    ph = Ig.sum(1)
    res = []
    for lo, hi in PH:
        m = (ph >= lo) & (ph < hi)
        if m.sum() < 100: continue
        zz = z[m]; ok = (zz.abs() < 6).all(1)
        ev = torch.linalg.eigvalsh(torch.cov(zz[ok].T)).cpu().numpy()
        r = rho[m].cpu().numpy()
        res.append(dict(lo=lo, hi=hi, n=int(m.sum()), eig_min=float(ev.min()), eig_max=float(ev.max()), eig=[float(v) for v in ev],
                        zp_rms=float(zz[ok][:, :2].pow(2).mean().sqrt()), zI_rms=float(zz[ok][:, 2:].pow(2).mean().sqrt()),
                        rho_p10=float(np.percentile(r, 10)), rho_p50=float(np.percentile(r, 50)), rho_p90=float(np.percentile(r, 90)),
                        prior_rmse_nm=float(ep[m].pow(2).mean().sqrt()) * 108.33))
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    pass1 = '--pass1' in sys.argv
    nb = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 40
    if '--n' in sys.argv: args.remove(str(nb))
    outj, calib, detp, cks = args[0], args[1], args[2], args[3:]
    from simcode.simulate.batches import TrainingBatches
    from simcode.detector.model import load as load_det
    from simcode.emitter.model import load as load_net, PSFCol
    from simcode.emitter.train import make_pattern_hold
    gen = TrainingBatches(64, 64, 6, seed=909, calib=[calib], p_std=1.0)      # SIMCODE (DMD) pattern in every FOV
    H = make_pattern_hold(gen, n=nb, pert_model='real')
    det, _ = load_det(detp)
    col = PSFCol(gen.rs['0.5nM'].sim)
    res = dict(calib=calib, detector=detp, pass1=pass1, n_batches=nb, nets={})
    for ck in cks:
        net = load_net(ck)
        A = collect(det, net, col, H, il_on=not pass1)
        res['nets'][ck] = metrics(A)
        print('==', ck, len(A['I']), 'targets', flush=True)
        for b in res['nets'][ck]:
            print(f"  {b['lo']:>5g}-{b['hi']:<6g} n {b['n']:6d} eig {b['eig_min']:.2f}..{b['eig_max']:.2f} zp {b['zp_rms']:.3f} zI {b['zI_rms']:.3f} "
                  f"rho p10/50/90 {b['rho_p10']:.2f}/{b['rho_p50']:.2f}/{b['rho_p90']:.2f} prior rmse {b['prior_rmse_nm']:.1f} nm", flush=True)
    json.dump(res, open(outj, 'w'), indent=1)


if __name__ == '__main__':
    main()
