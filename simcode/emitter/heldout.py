"""
Held-out set of the pass-1 emitter-network trainer (critic 29 #6): gate-type scenes with their own seeds (iso / pair2 / rs25x,
seed 401; heldout5: + rs05p / rs25p, >= 150 photons, the pass-1 detector's candidates), multi-emitter cov_or (crlb_all) per
target -> projected efficiency (median of (w.e)^2 / w C_or w over chi2_1 median) and Sigma_net calibration along the
B-profiled phase-gradient directions, nll_I, slope, I_lin / I_gt and |I - I_lin| / I_gt, prior z, per nn bin; prepared once
(prep), evaluated during training from the stored patches (eval_hold).
usage: python -m simcode.emitter.heldout [--out <file>]           -> simcode_out/emitter_heldout/heldout4.pt
       python -m simcode.emitter.heldout --set 5 [--out <file>]   -> simcode_out/emitter_heldout/heldout5.pt (the set of the model recipe)
"""
import argparse, os, time
import numpy as np, torch
from simcode import config
from simcode.layers.blocks import extract_patches
from simcode.emitter.model import PSFCol, NBR_PMIN, nbr_map, nbr_patches, nll_joint, mod_slope

dev = 'cuda'
HOLD = os.path.join('simcode_out', 'emitter_heldout', 'heldout4.pt')      # default file of set 4 (stage 'prep' output; not shipped: 0.2-0.3 GB)
HOLD5 = os.path.join('simcode_out', 'emitter_heldout', 'heldout5.pt')     # set 5 (iteration 30b: pretrain_detector_pass1 candidates, + rs05p / rs25p, >= 150 ph)
HOLD5_NAME = os.path.basename(HOLD5)      # the trainer recognizes set 5 by this file name
SC5_SCEN = (('iso', 13000), ('pair2', 5000), ('rs25x', 5000), ('rs05p', 6000), ('rs25p', 6000))
PHB = ((200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 1e9))


def prep(scen=(('iso', 13000), ('pair2', 5000), ('rs25x', 5000)), seed=401, out=HOLD, minph=300):
    """gate-type scenes (own seed), det candidates (p >= 0.2 splat; score >= 0.5 matched to targets on x_det, 1 px),
    stored per matched target: frame patch, offset, PSF column, neighbour patch, GT I, cov_or I-block, the B-profiled
    phase-gradient projection vectors (cov_or metric) and their reference variances, nn, photons, kinetics."""
    from simcode.simulate import gate_layouts as ST
    from simcode.simulate.gate_layouts import CD, S
    from simcode.benchmark import gate_scenes as GA
    from simcode.benchmark.gate_scenes_report import match
    from simcode.detector.select import select_subpixel
    from simcode.detector.model import load as load_det
    from simcode.simulate.crlb import illum_at
    T = 6
    CD.a.calib = config.model(config.INSITU_05)               # the PSF of the scenes (gate.gen renders with it)
    sim, rpsf = CD.models(T)
    det, _ = load_det(config.DETECTOR)
    col = PSFCol(sim)
    H = {}
    for sc, ntarget in scen:
        rec = []; need = ntarget; it = 0; t0 = time.time()
        while need > 0 and it < 400:
            it += 1
            o = GA.gen(sc, it, seed); nb = o['nb']
            G = o['gt']; act = G['onf'].sum(1) > 0
            G = {k: v[act] for k, v in G.items()}
            ph = G['phot'].sum(1); on = G['onf'].sum(1)
            inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
            tg = torch.nonzero(inter & (on >= 1) & (ph >= minph))[:, 0]
            if len(tg) == 0: continue
            nnd = ST.nn_dist(G, tg)
            fovs = torch.unique(G['fov'][tg]); ai = torch.nonzero(torch.isin(G['fov'], fovs))[:, 0]
            psS = torch.zeros(len(G['x']), CD.psf_channels(rpsf), CD.R, CD.R, device=dev)
            psS[ai] = CD.psf_stack(rpsf, G['x'][ai], G['y'][ai], G['z'][ai])
            Cb = CD.crlb_all(G, o, psS, tg, T, sim.focus); del psS
            cov = Cb['cov_or'].double(); ok = torch.isfinite(cov).all(2).all(1) & torch.isfinite(Cb['sf_m']).all(1)
            tg, nnd, cov = tg[ok], nnd[ok], cov[ok]; sfm = Cb['sf_m'][ok]
            with torch.no_grad():
                fr = o['frames']
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dd = det(fr)
                c = select_subpixel(dd['logit'].float(), dd['dxy'].float(), 640, p_min=NBR_PMIN, radius=0.7)
                keep = torch.isfinite(c['logit'])
                b, iy, ix, x, y, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
                nbr = nbr_map(dd['logit'].float()[:, 0] if dd['logit'].dim() == 4 else dd['logit'].float(), b, iy, ix, x, y, lg)
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]; Gy = G['y'] + o['shift_est'][G['fov']][:, 1]
                s5 = torch.sigmoid(lg) >= 0.5
                ci = torch.nonzero(s5)[:, 0]
                og = match(G['fov'][tg].cpu().numpy(), Gx[tg].cpu().numpy(), Gy[tg].cpu().numpy(),
                           b[ci].cpu().numpy(), x[ci].cpu().numpy(), y[ci].cpu().numpy(), 1.0)
                sel = og >= 0
                if sel.sum() == 0: continue
                cj = ci[torch.tensor(np.nonzero(sel)[0], device=dev)]; tj = torch.tensor(og[sel], device=dev)
                gi = tg[tj]
                pt = extract_patches(fr, b[cj], iy[cj], ix[cj])
                off = torch.stack([x[cj] - ix[cj].float(), y[cj] - iy[cj].float()], 1)
                nbp = nbr_patches(nbr, b[cj], iy[cj], ix[cj], x[cj], y[cj], torch.sigmoid(lg[cj]))
                # projection vectors (as decomp3.int_diag): C_or^-1 (g - a il), B-profiled
                CI = cov[tj][:, 2:, 2:]
                CI = 0.5 * (CI + CI.transpose(1, 2))
                Ci = torch.linalg.inv(CI + 1e-9 * CI.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(T, device=dev, dtype=torch.float64))
                ptr = o['patt'][G['fov'][gi]].double()
                il, gdx, gdy = illum_at(ptr, G['x'][gi].double(), G['y'][gi].double())
                Bv = G['bright'][gi].double()[:, None]
                Wv = []
                for gx in (Bv * gdx, Bv * gdy):
                    a_ = (il[:, None] @ Ci @ gx[..., None])[:, 0, 0] / (il[:, None] @ Ci @ il[..., None])[:, 0, 0]
                    Wv.append((Ci @ (gx - a_[:, None] * il)[..., None])[..., 0])
                Wv = torch.stack(Wv, 1)                                                     # [n,2,T]
                vo = torch.einsum('nat,nts,nas->na', Wv, CI, Wv)
                rec.append(dict(pt=pt.half().cpu(), off=off.cpu(), psf=col(off).cpu(), nbr=nbp.half().cpu(), x=x[cj].cpu(), y=y[cj].cpu(),
                                Ig=G['phot'][gi].cpu(), tx=Gx[gi].cpu(), ty=Gy[gi].cpu(), W=Wv.float().cpu(), vo=vo.float().cpu(),
                                CI=CI.float().cpu(), nn=nnd[tj].cpu(), ph=ph[gi].cpu(),
                                on=((G['t0'][gi] < 0) & (G['t1'][gi] > T)).cpu(), sfm=sfm[tj].cpu(), fov=(G['fov'][gi] + 1000 * it).cpu()))
                need -= len(gi)
            torch.cuda.empty_cache()
        H[sc] = {k: torch.cat([r[k] for r in rec]) for k in rec[0]}
        print(f'heldout {sc}: {len(H[sc]["Ig"])} matched targets ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(H, out)


@torch.no_grad()
def eval_hold(head, H, chunk=4096, band=(500, 5000), extra=False):
    """metrics per scene x nn bin in the photon band; extra: + 's5|' nn bins at 500-5k and 'ph<lo>-<hi>' photon bins (all nn)"""
    was = head.training; head.eval()
    R = {}
    for sc, D in H.items():
        outs = {k: [] for k in ('I', 'Ilin', 'Sn', 'px', 'py', 'sp', 'nll')}
        n = len(D['Ig'])
        for c0 in range(0, n, chunk):
            sl = slice(c0, c0 + chunk)
            pt = D['pt'][sl].to(dev).float(); off = D['off'][sl].to(dev); psf = D['psf'][sl].to(dev)
            o = head(pt, None, off, psf, D['nbr'][sl].to(dev).float())
            zp = torch.stack([D['tx'][sl].to(dev) - D['x'][sl].to(dev) - o['dxy'][:, 0], D['ty'][sl].to(dev) - D['y'][sl].to(dev) - o['dxy'][:, 1]], 1) / o['lsxy'].exp()
            nll, _ = nll_joint(D['Ig'][sl].to(dev), o['I'].float(), o['L'].float(), o['C'].float(), zp)   # given the prior error
            Lf, Cf = o['L'].float(), o['C'].float()
            outs['I'].append(o['I'].float()); outs['Ilin'].append(o['I_lin'].float()); outs['Sn'].append(Lf @ Lf.transpose(1, 2) + Cf @ Cf.transpose(1, 2))
            outs['px'].append(D['x'][sl].to(dev) + o['dxy'][:, 0]); outs['py'].append(D['y'][sl].to(dev) + o['dxy'][:, 1])
            outs['sp'].append(o['lsxy'].exp()); outs['nll'].append(nll)
        A = {k: torch.cat(v) for k, v in outs.items()}
        Ig = D['Ig'].to(dev); W = D['W'].to(dev); vo = D['vo'].to(dev); nnd = D['nn'].to(dev); ph = D['ph'].to(dev); on = D['on'].to(dev)
        e = A['I'] - Ig
        pe = (W * e[:, None]).sum(-1) ** 2 / vo.clamp(min=1e-9)
        vn = torch.einsum('nat,nts,nas->na', W, A['Sn'], W)
        cal = (W * e[:, None]).sum(-1) ** 2 / vn.clamp(min=1e-9)
        ez = torch.stack([(A['px'] - D['tx'].to(dev)) / A['sp'][:, 0], (A['py'] - D['ty'].to(dev)) / A['sp'][:, 1]], 1)
        ep = torch.stack([A['px'] - D['tx'].to(dev), A['py'] - D['ty'].to(dev)], 1)
        R[sc] = {}
        bins = {'all': (0, 1e9)} if sc in ('iso', 'rs05p', 'rs25p') else {}
        bins.update({'nn<1.5': (0, 1.5), 'nn1.5-2.5': (1.5, 2.5), 'nn2.5-4': (2.5, 4), 'nn>4': (4, 1e9)})
        masks = {bn: (ph >= band[0]) & (ph < band[1]) & (nnd >= a_) & (nnd < b_) for bn, (a_, b_) in bins.items()}
        if extra:
            masks.update({'s5|' + bn: (ph >= 500) & (ph < 5000) & (nnd >= a_) & (nnd < b_) for bn, (a_, b_) in bins.items()})
            masks.update({f'ph{lo}-{hi:.0f}': (ph >= lo) & (ph < hi) for lo, hi in PHB})
        for bn, m in masks.items():
            if m.sum() < 30: continue
            mo = m & on
            r = dict(n=int(m.sum()), nll_I=float(A['nll'][m].mean()), slope=mod_slope(A['I'][m], Ig[m]),
                     ilin_ratio=float((A['Ilin'][m].sum(1) / Ig[m].sum(1)).median()),
                     dI_rel=float(((A['I'][m] - A['Ilin'][m]).abs().sum(1) / Ig[m].sum(1)).median()),
                     prior_rmse=float(ep[m].pow(2).mean().sqrt()), prior_zrms=float(ez[m].pow(2).mean().sqrt()),
                     prior_z3=float((ez[m].abs() > 3).any(1).float().mean()))
            if mo.sum() >= 30:
                b = mod_slope(A['I'][mo], Ig[mo])
                I = A['I'][mo]; Ic = I.mean(1, keepdim=True) + (I - I.mean(1, keepdim=True)) / max(b, 0.3)
                pec = (W[mo] * (Ic - Ig[mo])[:, None]).sum(-1) ** 2 / vo[mo].clamp(min=1e-9)
                r.update(n_on=int(mo.sum()), peff=float(pe[mo].median() / 0.4549), peff_sc=float(pec.median() / 0.4549),
                         calib=float(cal[mo].median() / 0.4549))
            R[sc][bn] = r
    if was: head.train()
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', type=int, default=4, choices=[4, 5])
    ap.add_argument('--out', default='', help=f'output file (default {HOLD} / {HOLD5}; set 5 must be named {HOLD5_NAME})')
    a = ap.parse_args()
    out = a.out or (HOLD if a.set == 4 else HOLD5)
    if a.set == 4: prep(out=out)
    else: prep(SC5_SCEN, out=out, minph=150)
