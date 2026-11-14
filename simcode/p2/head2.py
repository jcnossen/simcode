"""
SCHeadP2: pattern-aware pass-2 head = SCHead4 (sc5nb: neighbour splat + detector-probability channels, signed per-frame
readout, full T x T intensity covariance, bg_t, position prior +- sigma) + the 2T illumination patch channels
(illum.illum_maps of the estimated pattern, cropped with the frame patch; zero-initialised input weights, so it starts
exactly at sc5nb). Outputs as SCHead4.
NOTE the position prior of this head is pattern-INFORMED (it sees the illumination); see front.py for how
the explicit solve uses it (variants a / b).
Training (train): init sc5nb, candidates from a given detector (the pass-2 detector DetP2 with the same estimated pattern,
or det_n1 / a control detector), realistic batches (p2/data.Gen2 'paper' mix, sc5 pair curriculum), the sc4 losses
(full-cov NLL of the GT per-frame photons, prior NLL vs the TRUE position, anti-shrinkage mr + slope, bg) with nn-bin weights.
Held-out: fixed seeded batches, matched candidates: nll_I, |I - I_gt| / I_gt, prior RMSE / z per nn bin and photon bin.
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode.p2.head2 train --out runs/p2_head --det runs/p2_det/final.pt
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from simcode.util.ckpt import save as ckpt_save     # atomic checkpoint write with a disk guard
from simcode.hd.sc4 import SCHead4, nbr_map, nbr_patches, nn_weights, nll_full, NBR_PMIN
from simcode.hd.sc import PSFCol, patches, mod_slope, PP, norm_photons

dev = 'cuda'
from simcode import config as _cfg
SC5NB = _cfg.model(_cfg.HEAD)


class SCHeadP2(SCHead4):
    nbr = True
    p2 = True

    def __init__(self, T=6, ch=96, nblk=4, P=21, bf16=True):
        super().__init__(T, ch, nblk, P, bf16)
        self.cin_il = nn.Conv2d(2 * T, ch, 3, padding=1, bias=False)
        nn.init.zeros_(self.cin_il.weight)

    def forward(self, pt, rn2p, off, psf=None, nbrp=None, illp=None, trunk_fn=None):
        n, T, P, _ = pt.shape
        flat = pt.reshape(n, T, -1)
        border = torch.cat([pt[:, :, 0], pt[:, :, -1], pt[:, :, 1:-1, 0], pt[:, :, 1:-1, -1]], -1)
        b0 = border.median(-1).values
        lvl = b0.mean(1)[:, None, None, None]
        o = torch.arange(P, device=pt.device, dtype=torch.float32) - P // 2
        dx = o[None, None, :] - off[:, 0, None, None]; dy = o[None, :, None] - off[:, 1, None, None]
        cx = dx.expand(n, P, P) / (P // 2); cy = dy.expand(n, P, P) / (P // 2)
        lin = (pt - lvl) / 100.0
        x = torch.cat([norm_photons(pt), norm_photons(pt.mean(1, keepdim=True)), lin, lin.mean(1, keepdim=True),
                       cx[:, None], cy[:, None], nbrp], 1)
        m1, d1, hm = (trunk_fn or self.trunk)(x, b0, off, illp)
        m = self.wmap[2](m1.float())[:, 0]; d = self.dmap[2](d1.float())
        q = self.out(hm.float())
        if psf is not None:
            g = psf
        else:
            g = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / 1.3 ** 2).expand(n, P, P)
            g = 0.7 * g / g.sum((1, 2), keepdim=True)
        gg = (g * g).sum((1, 2), keepdim=True)
        w0 = g / gg
        w = w0 * torch.exp(m.clamp(-5, 5))
        w = w / (w * g).sum((1, 2), keepdim=True)
        D = d * w0.amax((1, 2), keepdim=True)[:, None]
        D = D - ((D * g[:, None]).sum((2, 3), keepdim=True) / gg[:, None]) * g[:, None]
        wt = w[:, None] + D
        nt = T * (T + 1) // 2
        bg = b0 + q[:, T + nt + 4:] * (b0.abs() + 1).sqrt()
        Ilin = (wt.reshape(n, T, -1) * (flat - bg[..., None])).sum(-1)
        c = (Ilin.clamp(min=0).sum(1, keepdim=True) / T + bg.clamp(min=0).mean(1, keepdim=True) * 9).sqrt().clamp(min=3.0)
        dI = q[:, :T]
        ld = q[:, T:2 * T]; od = q[:, 2 * T:T + nt]
        L = torch.zeros(n, T, T, device=pt.device)
        ar, t0, t1 = self._diag_idx(pt.device)
        L[:, ar, ar] = c * torch.exp(ld.clamp(-4, 4)) * 1.5
        L[:, t0, t1] = c * od
        I = Ilin + c * dI
        dxy = 0.5 * torch.tanh(q[:, T + nt:T + nt + 2])
        lsxy = -4.5 + 5.0 * torch.sigmoid(q[:, T + nt + 2:T + nt + 4])
        return dict(I=I, L=L, dxy=dxy, lsxy=lsxy, bg=bg, I_lin=Ilin, w=w, wt=wt, dI=dI)


def is_p2_head(ck):
    """(2026-11-10) a pass-2 head checkpoint (head2.train args carry 'no_illum'), independent of the path name"""
    from simcode.config import resolve
    a = torch.load(resolve(ck), map_location='cpu', weights_only=False).get('args', {})
    return isinstance(a, dict) and 'no_illum' in a


def load(ck):
    from simcode.config import resolve
    c = torch.load(resolve(ck), map_location=dev, weights_only=False)
    h = SCHeadP2(6, 96, P=PP, bf16=True).to(dev)
    h.load_state_dict(c['model']); h.eval()
    return h


def load_init(ck=SC5NB):
    """init from sc5nb (illumination weights zeroed) or, if the checkpoint already has them (a pass-2 head), continue from it"""
    h = SCHeadP2(6, 96, P=PP, bf16=True).to(dev)
    from simcode.config import resolve
    sd = torch.load(resolve(ck), map_location=dev, weights_only=False)['model']
    own = h.state_dict(); own.update(sd); h.load_state_dict(own)
    if 'cin_il.weight' not in sd:
        nn.init.zeros_(h.cin_il.weight)
    return h


def load_det(path):
    """DetP2 checkpoint (needs the pattern) or a pattern-free detector -> (net, needs_illum)"""
    from simcode.config import resolve
    path = resolve(path)
    c = torch.load(path, map_location='cpu', weights_only=False)
    a = c.get('args', {})
    if isinstance(a, dict) and 'no_illum' in a or 'p2_det' in path or 'cin_il.weight' in c.get('model', {}):
        from simcode.p2.det2 import load as L2
        net, _ = L2(path); return net, True
    from simcode.hd.sc import load_det_front
    net, _ = load_det_front(path); net.eval(); return net, False


@torch.no_grad()
def candidates(det, needs_il, fr, il, k=640):
    """detector -> kept candidates (p >= NBR_PMIN) + neighbour maps (pass-2: the same estimated pattern for the detector)"""
    from simcode.det.model import select_subpixel
    with torch.autocast('cuda', dtype=torch.bfloat16):
        dd = det(fr, il) if needs_il else det(fr)
    lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
    c = select_subpixel(lgm, dd['dxy'].float(), k, p_min=NBR_PMIN, radius=0.7)
    kp = torch.isfinite(c['logit'])
    cc = {kk: c[kk][kp] for kk in ('b', 'iy', 'ix', 'x', 'y', 'logit')}
    nbr = nbr_map(lgm, cc['b'], cc['iy'], cc['ix'], cc['x'], cc['y'], cc['logit'])
    return cc, nbr, lgm


def head_apply(head, col, fr, il, b, iy, ix, x, y, nbr, p):
    off = torch.stack([x - ix.float(), y - iy.float()], 1)
    pt = patches(fr, b, iy, ix)
    nbp = nbr_patches(nbr, b, iy, ix, x, y, p)
    if getattr(head, 'p2', False):
        o = head(pt, None, off, col(off), nbp, patches(il, b, iy, ix))
    else:
        o = head(pt, None, off, col(off), nbp)
    o['x'] = x + o['dxy'][:, 0]; o['y'] = y + o['dxy'][:, 1]; o['x_det'] = x; o['y_det'] = y
    return o


# ------------------------------------------------------------------------------------------------ held-out
def make_hold(gen, n=30, nb=16, seed=909, pert_model='v1'):
    H = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for i in range(n):
            o = gen.batch2(nb, mix='paper', pair_curr=True, pert_model=pert_model)
            H.append(o)
    return H


@torch.no_grad()
def eval_hold(det, needs_il, head, col, H, il_on=True):
    from simcode.hd.sc4 import match_gt
    was = head.training; head.eval()
    acc = {}
    for o in H:
        il = o['illum_p2'] if il_on else torch.zeros_like(o['illum_p2'])
        c, nbr, lgm = candidates(det, needs_il, o['frames'], il)
        c5 = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
        m = match_gt(o, c5)
        if m is None: continue
        p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
        out = head_apply(head, col[o['calib_i']] if isinstance(col, list) else col, o['frames'], il, m['b'], m['iy'], m['ix'], m['x'], m['y'], nbr, p_own)
        G = o['gt']; g = m['g']; Ig = G['phot'][g]
        nll, _ = nll_full(Ig, out['I'].float(), out['L'].float())
        ep = torch.stack([out['x'] - G['x'][g], out['y'] - G['y'][g]], 1)
        z = ep / out['lsxy'].exp()
        ph = Ig.sum(1); nnd = m['nn']
        eI = ((out['I'] - Ig).abs().sum(1) / Ig.sum(1).clamp(min=1))
        for nm, (a, b) in (('nn<1.5', (0, 1.5)), ('nn1.5-2.5', (1.5, 2.5)), ('nn>2.5', (2.5, 1e9))):
            for pl, phh in ((200, 700), (700, 2000), (2000, 1e9)):
                mm = (nnd >= a) & (nnd < b) & (ph >= pl) & (ph < phh)
                if not mm.any(): continue
                k = f'{nm}|{pl}-{phh:.0f}'
                r = acc.setdefault(k, dict(n=0, nll=0.0, eI=[], e2=0.0, z2=0.0))
                r['n'] += int(mm.sum()); r['nll'] += float(nll[mm].sum()); r['eI'].append(eI[mm].cpu())
                r['e2'] += float((ep[mm] ** 2).sum()); r['z2'] += float((z[mm] ** 2).sum())
    if was: head.train()
    return {k: dict(n=r['n'], nll_I=r['nll'] / r['n'], eI_med=float(torch.cat(r['eI']).median()),
                    prior_rmse_nm=float(np.sqrt(r['e2'] / (2 * r['n']))) * 108.33, prior_zrms=float(np.sqrt(r['z2'] / (2 * r['n']))))
            for k, r in sorted(acc.items())}


# ------------------------------------------------------------------------------------------------ training
def _load_pgen(v):
    """(2026-11-13 generic p2, opt-in) --pgen: '' -> None (unchanged), a json file path, or inline json"""
    import json as _j, os as _o
    if not v: return None
    return _j.load(open(v)) if _o.path.exists(v) else _j.loads(v)


def train(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--det', required=True)
    ap.add_argument('--init', default=SC5NB)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--pct', type=float, default=0.1)
    ap.add_argument('--w_mr', type=float, default=2.0)
    ap.add_argument('--w_slope', type=float, default=50.0)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--eval_every', type=int, default=4000)
    ap.add_argument('--ckpt_every', type=int, default=10000)
    ap.add_argument('--no_illum', type=int, default=0)       # control: same data / schedule, illumination inputs zeroed
    ap.add_argument('--pert_model', default='real')         # pattern-error model of the input estimate (illum.MODELS; p2headL: real,
                                                            # the p2head stage before it: v1; default 'v1' until 2026-11-14)
    ap.add_argument('--hold_v1', type=int, default=0)       # also a held-out set with the v1 error model (n batches)
    # the PSF of the simulator (in-situ calibration json; a comma list = a mixture, hd/data.Gen calib; the PSF-template column
    # follows the batch's calib)
    ap.add_argument('--calib', required=True)
    # (2026-11-12 g13 retrain, opt-in; '' = unchanged) training ranges 'lo,hi': ON-emitter density (/px^2) and mean on-time (frames)
    ap.add_argument('--dens', default='')
    ap.add_argument('--tau', default='')
    ap.add_argument('--pgen', default='')   # (2026-11-13 generic p2, opt-in; '' = unchanged) json file or inline json: realsim pgen dict
    a = ap.parse_args(argv)
    from simcode.util.proctitle import auto; auto('train-p2head')     # recognizable name in nvidia-smi / ps
    from simcode.p2.data import Gen2
    from simcode.hd.sc4 import match_gt, targets
    from simcode.hd import sc as SCM
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    gen = Gen2(64, 64, 6, seed=a.seed, calib=[_cfg.resolve(c) for c in a.calib.split(',')],
               pgen=_load_pgen(a.pgen))   # (2026-11-13 generic p2, opt-in)
    psub = {}
    if a.dens: psub['dens'] = tuple(float(v) for v in a.dens.split(','))
    if a.tau: psub['tau_r'] = tuple(float(v) for v in a.tau.split(','))
    det, needs_il = load_det(a.det)
    head = load_init(a.init)
    col = PSFCol(gen.rs['0.5nM'].sim)
    if getattr(gen, 'calibs', None): col = [PSFCol(r['0.5nM'].sim) for r in gen.rsm]    # per-calib template columns
    cols = col
    Hd = make_hold(gen, pert_model=a.pert_model)
    Hv1 = make_hold(gen, n=a.hold_v1, seed=919) if a.hold_v1 else None
    ilon = not a.no_illum
    # references: the default set's pass-1 head on its pass-1 detector's candidates (config.HEAD / config.DET; evaluation
    # only), and the initial head on this detector
    from simcode.hd.sc import load_det_front
    n1, _ = load_det_front(_cfg.model(_cfg.DET)); n1.eval()
    ref = dict(p1_ref=eval_hold(n1, False, SCM.load(SC5NB), col, Hd), init=eval_hold(det, needs_il, head, col, Hd, ilon))
    if Hv1: ref['init_v1'] = eval_hold(det, needs_il, head, col, Hv1, ilon)
    del n1
    json.dump(ref, open(a.out + '/eval_ref.json', 'w'), indent=1)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); elog = open(a.out + '/eval.jsonl', 'a')
    t0 = time.time(); acc = {}; IS, GS = [], []
    for step in range(a.steps + 1):
        with torch.no_grad():
            o = gen.batch2(a.batch, mix='paper', pair_curr=True, pert_model=a.pert_model, **psub)
            col = cols[o['calib_i']] if isinstance(cols, list) else cols
            fr = o['frames']; il = o['illum_p2'] if ilon else torch.zeros_like(o['illum_p2'])
            c, nbr, lgm = candidates(det, needs_il, fr, il)
            c5 = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
            m = match_gt(o, c5)
        if m is None:
            sched.step(); continue
        with torch.no_grad():
            t = targets(o, m, col)
            iso_w = (m['nn'] >= 2.0).float()
            wt = nn_weights(o['gt']['fov'][m['g']], m['nn'])
            p_own = torch.sigmoid(lgm[m['b'], m['iy'], m['ix']])
        out = head_apply(head, col, fr, il, m['b'], m['iy'], m['ix'], m['x'], m['y'], nbr, p_own)
        Ig, tx, ty = t['Ig'], t['tx'], t['ty']
        nll_i, z = nll_full(Ig, out['I'], out['L'])
        nll_I = (wt * nll_i).mean()
        ex = tx - out['x']; ey = ty - out['y']; ls = out['lsxy']
        nll_p = (wt * (0.5 * (ex / ls[:, 0].exp()) ** 2 + ls[:, 0] + 0.5 * (ey / ls[:, 1].exp()) ** 2 + ls[:, 1])).mean()
        cn = (Ig.clamp(min=0).sum(1, keepdim=True) / 6 + 30).sqrt()
        mr = ((out['I'] - out['I'].mean(1, keepdim=True)) - (Ig - Ig.mean(1, keepdim=True))) / cn
        l_mr = (wt[:, None] * F.huber_loss(mr, torch.zeros_like(mr), delta=3.0, reduction='none')).mean()
        am = out['I'] - out['I'].mean(1, keepdim=True); gm = Ig - Ig.mean(1, keepdim=True)
        slope_b = (am * gm * iso_w[:, None]).sum() / (gm * gm * iso_w[:, None]).sum().clamp(min=1e-6)
        l_sl = (1 - slope_b) ** 2
        bgt = t['bgt_px']
        l_bg = (wt[:, None] * ((out['bg'] - bgt) ** 2) / (bgt + 1.0)).mean()
        loss = nll_I / 6 + nll_p + 0.1 * l_bg + a.w_mr * l_mr + a.w_slope * l_sl
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        else:
            acc['nonfinite'] = acc.get('nonfinite', 0) + 1
        sched.step()
        with torch.no_grad():
            IS.append(out['I'].detach()); GS.append(Ig)
            for k, v in dict(slope_iso=slope_b.item(), mr=l_mr.item(), nll_I=nll_I.item(), nll_p=nll_p.item(), bg=l_bg.item(), n=len(Ig),
                             zI=float((z ** 2).mean().sqrt()), pos_rmse=float(torch.sqrt((ex ** 2 + ey ** 2).mean() / 2))).items():
                acc[k] = acc.get(k, 0) + v
            acc['_n'] = acc.get('_n', 0) + 1
        if step % 200 == 0:
            nn_ = acc.pop('_n', 1)
            Ia, Ga = torch.cat(IS), torch.cat(GS); IS, GS = [], []
            r = dict(step=step, t=round(time.time() - t0), lr=sched.get_last_lr()[0], slope=mod_slope(Ia, Ga),
                     w_il=float(head.cin_il.weight.abs().mean()), **{k: v / nn_ for k, v in acc.items()})
            acc = {}
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.eval_every == 0 and step > 0:
            ev = eval_hold(det, needs_il, head, cols, Hd, ilon)
            if Hv1: ev['v1'] = eval_hold(det, needs_il, head, cols, Hv1, ilon)
            elog.write(json.dumps(dict(step=step, t=round(time.time() - t0), **ev)) + '\n'); elog.flush()
            print('EVAL', step, json.dumps(ev), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            ckpt_save(dict(model=head.state_dict(), args=vars(a), step=step), a.out + f'/ckpt_{step}.pt')
    ckpt_save(dict(model=head.state_dict(), args=vars(a), step=a.steps), a.out + '/final.pt')


def check():
    """zero-init check: SCHeadP2 from sc5nb == sc5nb on det_n1 candidates"""
    from simcode.p2.data import Gen2
    from simcode.hd import sc as SCM
    torch.manual_seed(0)
    gen = Gen2(64, 64, 6, seed=0)
    o = gen.batch2(4, pair_curr=True)
    det, _ = load_det(_cfg.model(_cfg.DET))
    col = PSFCol(gen.rs['0.5nM'].sim)
    c, nbr, lgm = candidates(det, False, o['frames'], o['illum_p2'])
    p = torch.sigmoid(c['logit'])
    h0 = SCM.load(SC5NB); h1 = load_init().eval()
    with torch.no_grad():
        a = head_apply(h0, col, o['frames'], o['illum_p2'], c['b'], c['iy'], c['ix'], c['x'], c['y'], nbr, p)
        b = head_apply(h1, col, o['frames'], o['illum_p2'], c['b'], c['iy'], c['ix'], c['x'], c['y'], nbr, p)
    print(len(p), 'cands; max |dI|', float((a['I'] - b['I']).abs().max()), 'max |dx|', float((a['x'] - b['x']).abs().max()))


if __name__ == '__main__':
    {'train': train, 'check': lambda a: check()}[sys.argv[1]](sys.argv[2:])
