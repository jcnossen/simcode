"""Per-frame intensity and background vs their CRLBs (coordinator / user item, iteration 32). Evaluation only.
Scenes regenerated with gate.gen (same seeds -> the same scenes as the gate npz). Per visible interior target:
  I_t   network per-frame intensities (SCHead4 apply_head; no solve needed) vs GT per-frame photons; CRLB = the gate's
        multi-emitter free-model per-frame intensity CRLB (crlb_dense.crlb_all 'fr_m': positions, z, per-angle bg free)
  bg_t  network per-frame bg vs the head's training target (true bg o['bg'] at the rounded true pixel, sc3.targets
        bgt_px); CRLB = multi-emitter Fisher over the head's 21 x 21 patch with a FLAT per-frame bg level b_t and the
        per-frame intensities of all emitters within the patch (+ 7 px PSF reach) as parameters (positions known,
        exact 15 x 15 PSF, var = mu + rn^2) -> [J^-1]_bb. 'floor' = rms of (pixel bg - 21 x 21 patch mean of the true bg):
        the part of the pixel target a flat-bg model cannot see (0 in flat-bg scenes).
Two candidate sets: 'det' = the detector's candidates (p >= 0.5) matched one-to-one to targets within 1 px (greedy);
'forced' = the head evaluated at ALL active GT emitters' true positions (no detection selection).
usage: CUDA_VISIBLE_DEVICES=1 python icrlb.py <ckpt> <name> [scenes] [seeds] -> <out>/sc7/icrlb_<name>_<scene>.npz"""
import sys, os, math
import numpy as np, torch
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
if not ARGS or ARGS[0] in ('-h', '--help'): print(__doc__); sys.exit(0)
from simcode2.sim import simtools as ST
from simcode2.sim.simtools import CD, dev, S
from simcode2.hd import gate as GA, sc as SCM
from simcode2.det.model import select_subpixel
from simcode2 import config
OUTR = config.root('out')

OUT = os.path.join(OUTR, 'sc7')
T = 6; PP = 21; HP = PP // 2


def bg_crlb(G, o, psS, tg, Ie):
    """flat per-frame bg CRLB over the head's 21 x 21 patch (see module doc). Returns crlb [n,T], floor [n,T], patch mean [n,T]"""
    R, HR = CD.R, CD.HR
    pxi, pyi = torch.round(G['x']).long(), torch.round(G['y']).long()
    n = len(tg); nb = o['nb']
    tb, tcx, tcy = G['fov'][tg], pxi[tg], pyi[tg]
    bgp = SCM.patches(o['bg'], tb, tcy, tcx)                                   # [n,T,21,21] centred on the target pixel
    rn2 = SCM.patches((o['rn'] ** 2).expand(nb, 1, S, S), tb, tcy, tcx)[:, 0]
    crl = torch.full((n, T), float('nan'), device=dev); flo = crl.clone(); pmean = crl.clone()
    u = torch.arange(PP, device=dev) - HP
    for i in range(n):
        j = tg[i]
        cand = torch.nonzero((G['fov'] == G['fov'][j]) & ((pxi - pxi[j]).abs() <= HP + HR) & ((pyi - pyi[j]).abs() <= HP + HR))[:, 0]
        K = len(cand)
        dx = pxi[cand] - pxi[j]; dy = pyi[cand] - pyi[j]
        yy = u[None, :, None] - dy[:, None, None] + HR; xx = u[None, None, :] - dx[:, None, None] + HR
        ok = (yy >= 0) & (yy < R) & (xx >= 0) & (xx < R)
        g = psS[cand, 0][torch.arange(K, device=dev)[:, None, None], yy.clamp(0, R - 1), xx.clamp(0, R - 1)] * ok   # [K,21,21]
        mu = (Ie[cand][:, :, None, None] * g[:, None]).sum(0) + bgp[i] + rn2[i][None]      # [T,21,21]
        w = 1.0 / mu.clamp(min=1e-3)
        Jc = torch.cat([g, torch.ones(1, PP, PP, device=dev)], 0).reshape(K + 1, -1)          # [K+1, P2]
        FI = torch.einsum('ap,tp,bp->tab', Jc, w.reshape(T, -1), Jc).double()
        FI = FI + 1e-9 * torch.eye(K + 1, device=dev, dtype=torch.float64)
        try:
            crl[i] = torch.linalg.inv(FI)[:, K, K].float().clamp(min=0).sqrt()
        except Exception:
            pass
        pm = bgp[i].mean((1, 2)); pmean[i] = pm
        flo[i] = bgp[i][:, HP, HP] - pm
    return crl, flo, pmean


def run(ck, name, scenes, seeds, ntarget=3500):
    sim, rpsf, lut = CD.models(T)
    det, _ = SCM.load_det_front(config.DET); det.eval()
    head = SCM.load(ck); col = SCM.PSFCol(sim)
    for sc in scenes:
        rows = []
        for seed in seeds:
            need = ntarget; it = 0
            while need > 0 and it < 400:
                it += 1
                o = GA.gen(sc, it, seed)
                Gf = o['gt']; act = Gf['onf'].sum(1) > 0
                G = {k: v[act] for k, v in Gf.items()}
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                tg = torch.nonzero(inter & (on >= 1) & (ph >= 100))[:, 0]
                if len(tg) == 0: continue
                with torch.no_grad():
                    nn = ST.nn_dist(G, tg)
                    psS = CD.psf_stack(rpsf, G['x'], G['y'], G['z'], sim.focus)
                    Cb = CD.crlb_all(G, o, psS, tg, T, sim.focus)
                    bc, bf, bpm = bg_crlb(G, o, psS, tg, G['phot'])
                    fr = o['frames']
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        dd = det(fr)
                    lgm = dd['logit'].float()
                    # --- detector candidates
                    c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=GA.THR0, radius=0.7)
                    keep = torch.isfinite(c['logit'])
                    b, iy, ix, x, y, lg = (c[k][keep] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit'))
                    nbr = SCM.nbr_for(head, lgm, b, iy, ix, x, y, lg)
                    sel = torch.sigmoid(lg) >= 0.5
                    oh = SCM.apply_head(head, col, fr, None, b[sel], iy[sel], ix[sel], x[sel], y[sel], nbr=nbr, p=torch.sigmoid(lg[sel]))
                    bd, xd, yd = b[sel], x[sel], y[sel]
                    # one-to-one greedy match targets <-> detections within 1 px (same FOV)
                    D = torch.cdist(torch.stack([G['x'][tg], G['y'][tg]], 1), torch.stack([xd, yd], 1))
                    D = torch.where(G['fov'][tg][:, None] == bd[None], D, torch.full_like(D, 1e9)).cpu().numpy()
                    mi = np.full(len(tg), -1)
                    if D.size:
                        ii, jj = np.nonzero(D < 1.0); o_ = np.argsort(D[ii, jj]); ua, ub = set(), set()
                        for k in o_:
                            if ii[k] in ua or jj[k] in ub: continue
                            ua.add(ii[k]); ub.add(jj[k]); mi[ii[k]] = jj[k]
                    # --- forced candidates at all active GT positions inside the image (photons >= 50)
                    fa = torch.nonzero((G['x'] > -0.5) & (G['x'] < S - 0.5) & (G['y'] > -0.5) & (G['y'] < S - 0.5) & (ph >= 50))[:, 0]
                    fb, fx, fy = G['fov'][fa], G['x'][fa], G['y'][fa]
                    fix, fiy = torch.round(fx).long(), torch.round(fy).long(); flg = torch.full_like(fx, 3.0)
                    fnbr = SCM.nbr_for(head, lgm, fb, fiy, fix, fx, fy, flg)
                    of = SCM.apply_head(head, col, fr, None, fb, fiy, fix, fx, fy, nbr=fnbr, p=torch.sigmoid(flg))
                    pos = {int(v): k for k, v in enumerate(fa.tolist())}
                    fi = torch.tensor([pos[int(t)] for t in tg.tolist()], device=dev)
                bgt = o['bg'][G['fov'][tg], :, torch.round(G['y'][tg]).long(), torch.round(G['x'][tg]).long()]
                mt = torch.tensor(mi, device=dev)
                nanT = torch.full((len(tg), T), float('nan'), device=dev)
                Idet = torch.where((mt >= 0)[:, None], oh['I'].float()[mt.clamp(min=0)], nanT) if len(xd) else nanT
                bdet = torch.where((mt >= 0)[:, None], oh['bg'].float()[mt.clamp(min=0)], nanT) if len(xd) else nanT
                r = dict(ph=ph[tg], I_gt=G['phot'][tg], fr_m=Cb['fr_m'], sf_m=Cb['sf_m'], nn=nn, I_det=Idet, bg_det=bdet,
                         I_f=of['I'].float()[fi], bg_f=of['bg'].float()[fi], bg_t=bgt, bg_crlb=bc, bg_floor=bf, bg_pmean=bpm,
                         matched=(mt >= 0).float(), seed=torch.full_like(ph[tg], seed))
                rows.append({k: v.detach().float().cpu().numpy() for k, v in r.items()})
                need -= len(tg); torch.cuda.empty_cache()
            print(f'{name} {sc} seed {seed}: done ({it} it)', flush=True)
        D = {k: np.concatenate([r_[k] for r_ in rows]) for k in rows[0]}
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(f'{OUT}/icrlb_{name}_{sc}.npz', **D)


if __name__ == '__main__':
    ck, name = ARGS[0], ARGS[1]
    scenes = (ARGS[2] if len(ARGS) > 2 else 'iso,rs05q,rs05qm').split(',')
    seeds = [int(s) for s in (ARGS[3] if len(ARGS) > 3 else '301').split(',')]
    run(config.resolve(ck), name, scenes, seeds)      # e.g. hd_sc5nb.pt (models dir; development runs/hd_sc5nb/final.pt)
