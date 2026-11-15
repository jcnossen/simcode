"""Solve-gap diagnostic (2026-11-14), part A: classical single-emitter SIMFLUX MLE pixel fit on the iso gate scene of the
m1809a set (seeds 301-303), on the SAME simulated frames (simcode/benchmark/gate.gen regenerated with the same seeds / iteration
loop as simcode/processing/gate_dump.py; GT checked identical to out/psfmatch/fe_m1809a) and the TRUE pattern.
Model = the simulator's own forward model (RealSim LUT PSF, 41 x 41 render cropped to the 15 x 15 ROI of the CRLB, per-angle
LUT if the calib has one): mu_t = bg_angle + B il_t(x, y) psf_t(x, y, z); parameters (x, y, z, B, bg_0, bg_1) as the CRLB_SF of
simcode/simulate/crlb.py (kinetics are constant in iso); likelihood Poisson(d + rn^2 | mu + rn^2) (the CRLB's weights).
ROI centred on the matched detector pixel (pass-2 cache, h3 rows). Two-step classical fit: SMLM (flat illumination) from the
detector point, then SIMFLUX from the SMLM result (LM / Fisher scoring, 60 iterations). Also: SIMFLUX fit started at the truth,
and the Fisher bound of this model at the truth (vs the stored gt_sf_i: is the CRLB reference the simulation's model?).
Output out/solvegap/sffit_s<seed>.npz (rows = h3-matched iso targets of the pass-2 cache).
usage: SIMCODE_GATE_CALIB=models/insitu/calib_1809_2_psf3d_avg.json CUDA_VISIBLE_DEVICES=0 python analysis/solvegap/sffit.py 301,302,303"""
import os, sys, time
import numpy as np, torch
from simcode.simulate.gate_layouts import CD, S
from simcode.benchmark import gate as GA
from simcode.solve import pattern as C
from simcode.simulate import simulator as _SIMM

_render0 = _SIMM.render
def _render_small(psf, p, **kw):   # GPU 0 is shared: build the PSF LUT in chunks of 128 (same values)
    return torch.cat([_render0(psf, q, **kw) for q in p.split(128)])
_SIMM.render = _render_small

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, 'out', 'crlbcmp', 'cache')
OUT = os.path.join(REPO, 'out', 'solvegap')
dev = 'cuda'; R = 15; HR = R // 2; T = 6


class Model:
    def __init__(self, rs):
        self.sim = rs.sim
        L = rs.luts[0]
        self.luts = list(L) if isinstance(L, tuple) else [L]
        self.RR = self.sim.R; self.c = self.RR // 2

    def psf(self, x, y, z, cx, cy, lut):
        """[n,R,R] PSF on the ROI centred at pixel (cx, cy) for an emitter at (x, y, z) (absolute px)"""
        base = self.sim.lut; self.sim.lut = lut
        im = self.sim.render_psf(x, y, z)                       # centred at round(x), round(y)
        self.sim.lut = base
        dx = (cx - torch.round(x)).long(); dy = (cy - torch.round(y)).long()
        o = torch.arange(R, device=dev) - HR
        iy = (self.c + dy[:, None] + o[None]).clamp(0, self.RR - 1); ix = (self.c + dx[:, None] + o[None]).clamp(0, self.RR - 1)
        n = len(x)
        return im[torch.arange(n, device=dev)[:, None, None], iy[:, :, None], ix[:, None, :]]

    def psf_t(self, x, y, z, cx, cy, ang):
        """[n,T,R,R]: per-frame PSF (angle LUT per frame if available)"""
        ps = [self.psf(x, y, z, cx, cy, l) for l in self.luts]
        if len(ps) == 1: return ps[0][:, None].expand(-1, T, -1, -1)
        P2 = torch.stack(ps, 1)                                  # [n,2,R,R]
        return P2.gather(1, ang[:, :, None, None].expand(-1, -1, R, R))

    def mu(self, p, pt, cx, cy, ang, sf=True):
        """p [n,6] = x, y, z, B, bg0, bg1 -> mu [n,T,R,R]"""
        il = C.illum_at(pt, p[:, 0], p[:, 1])[0] if sf else pt[..., 4]
        ps = self.psf_t(p[:, 0], p[:, 1], p[:, 2], cx, cy, ang)
        bg = torch.where(ang == 0, p[:, 4:5], p[:, 5:6])
        return bg[..., None, None] + (p[:, 3:4] * il)[..., None, None] * ps

    def jac(self, p, pt, cx, cy, ang, sf=True):
        m0 = self.mu(p, pt, cx, cy, ang, sf)
        J = []
        for i, h in ((0, 1e-3), (1, 1e-3), (2, 2e-3)):
            dp = torch.zeros_like(p); dp[:, i] = h
            pp, pm = p + dp, p - dp
            pp[:, 2] = pp[:, 2].clamp(0, self.sim.zmax); pm[:, 2] = pm[:, 2].clamp(0, self.sim.zmax)
            J.append((self.mu(pp, pt, cx, cy, ang, sf) - self.mu(pm, pt, cx, cy, ang, sf)) / (pp[:, i] - pm[:, i]).clamp(min=1e-6)[:, None, None, None])
        il = C.illum_at(pt, p[:, 0], p[:, 1])[0] if sf else pt[..., 4]
        ps = self.psf_t(p[:, 0], p[:, 1], p[:, 2], cx, cy, ang)
        J.append(il[..., None, None] * ps)
        J.append((ang == 0).float()[..., None, None].expand_as(m0)); J.append((ang == 1).float()[..., None, None].expand_as(m0))
        return m0, torch.stack(J, -1)                            # [n,T,R,R,6]


def fit(M, d, rn2, p, pt, cx, cy, ang, sf=True, iters=60):
    zmax = M.sim.zmax
    def nll(m): mm = (m + rn2).clamp(min=1e-3); return (mm - (d + rn2) * torch.log(mm)).sum((1, 2, 3))
    lam = torch.full((len(p),), 1e-2, device=dev)
    m, J = M.jac(p, pt, cx, cy, ang, sf); cost = nll(m)
    for _ in range(iters):
        w = 1.0 / (m + rn2).clamp(min=1e-3)
        g = torch.einsum('ntyxk,ntyx->nk', J, (d + rn2) * w - 1)            # gradient of log L
        H = torch.einsum('ntyxk,ntyxl,ntyx->nkl', J, J, w)
        Hd = H + torch.diag_embed(lam[:, None] * H.diagonal(dim1=1, dim2=2))
        dp = torch.linalg.solve_ex(Hd, g[..., None])[0][..., 0]
        pn = p + torch.nan_to_num(dp)
        pn[:, 2] = pn[:, 2].clamp(0, zmax); pn[:, 3] = pn[:, 3].clamp(min=1.0); pn[:, 4:] = pn[:, 4:].clamp(min=0.01)
        pn[:, :2] = torch.maximum(torch.minimum(pn[:, :2], torch.stack([cx, cy], 1) + 3), torch.stack([cx, cy], 1) - 3)
        mn, Jn = M.jac(pn, pt, cx, cy, ang, sf); cn = nll(mn)
        acc = cn < cost
        p = torch.where(acc[:, None], pn, p); m = torch.where(acc[:, None, None, None], mn, m)
        J = torch.where(acc[:, None, None, None, None], Jn, J); cost = torch.where(acc, cn, cost)
        lam = torch.where(acc, lam / 3, lam * 4).clamp(1e-7, 1e7)
    return p


def fisher_xy(M, p, pt, cx, cy, ang, rn2):
    m, J = M.jac(p, pt, cx, cy, ang, True)
    w = 1.0 / (m + rn2)
    F = torch.einsum('ntyxk,ntyxl,ntyx->nkl', J.double(), J.double(), w.double())
    Ci = torch.linalg.inv(F + 1e-9 * torch.diag_embed(F.diagonal(dim1=1, dim2=2)))
    return Ci.diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt().float()


@torch.no_grad()
def run(seed):
    fo = f'{OUT}/sffit_s{seed}.npz'
    if os.path.exists(fo): print('exists', fo); return
    D = dict(np.load(f'{CACHE}/iso__s{seed}_p2.npz'))
    CD.a.calib = GA.gate_calib()
    rs = GA.rsim('0.5nM')
    M = Model(rs)
    tg_all = np.nonzero(D['gt_target'] > 0)[0]
    j3 = D['m_h3'][tg_all]; ok = j3 >= 0
    rows, cand = tg_all[ok], j3[ok]
    t0 = time.time(); need = 3500; it = 0; fov0 = 0; goff = 0
    res = {k: [] for k in ('row', 'smlm', 'sf', 'sf_true', 'crlb_model', 'roi_sum', 'bg_true')}
    gx = []
    while need > 0 and it < 400:
        it += 1
        o = GA.gen('iso', it, seed)
        nb = o['nb']; G = o['gt']; act = G['onf'].sum(1) > 0
        G = {k: v[act] for k, v in G.items()}
        n = len(G['x'])
        gx.append(G['x'].cpu().numpy())
        ph = G['phot'].sum(1); on = G['onf'].sum(1)
        inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
        vis = (on >= 1) & (ph >= 100)
        ntg = int((inter & vis).sum())
        # rows of the cache that belong to this iteration
        sel = (rows >= goff) & (rows < goff + n)
        r_, c_ = rows[sel], cand[sel]
        if len(r_):
            li = torch.tensor(r_ - goff, device=dev)
            fov = G['fov'][li]
            assert np.allclose(G['x'][li].cpu().numpy(), D['gt_x'][r_], atol=1e-4) and np.allclose(D['gt_fov'][r_], fov.cpu().numpy() + fov0)
            cx = torch.round(torch.tensor(D['c_x_det'][c_], device=dev)); cy = torch.round(torch.tensor(D['c_y_det'][c_], device=dev))
            dat = GA_patch(o['frames'], fov, cy, cx)
            rn2 = GA_patch((o['rn'] ** 2).expand(nb, T, S, S), fov, cy, cx)
            bgt = GA_patch(o['bg'], fov, cy, cx)
            pt = o['patt'][fov]
            ang = (o['steps'] // 3)[fov]
            # check: the frame angle of the steps = the pattern's angle index
            assert (ang == C.angle_index(pt)).all() or len(M.luts) == 1
            # SMLM start: detector point, z = median, bg = ROI border median per angle, B = ROI sum - bg
            brd = torch.cat([dat[..., 0, :], dat[..., -1, :], dat[..., 1:-1, 0], dat[..., 1:-1, -1]], -1).median(-1).values   # [n,T]
            b0 = torch.stack([(brd * (ang == a)).sum(1) / (ang == a).sum(1) for a in (0, 1)], 1)
            Bs = ((dat - torch.where(ang == 0, b0[:, :1], b0[:, 1:])[..., None, None]).sum((2, 3)) / pt[..., 4]).mean(1).clamp(min=50) / 0.85
            p0 = torch.stack([torch.tensor(D['c_x_det'][c_], device=dev), torch.tensor(D['c_y_det'][c_], device=dev),
                              torch.full_like(cx, 0.05), Bs, b0[:, 0], b0[:, 1]], 1)
            ps = fit(M, dat, rn2, p0, pt, cx, cy, ang, sf=False)
            pf = fit(M, dat, rn2, ps.clone(), pt, cx, cy, ang, sf=True)
            # truth start
            bga = torch.stack([(bgt.mean((2, 3)) * (ang == a)).sum(1) / (ang == a).sum(1) for a in (0, 1)], 1)
            ptru = torch.stack([G['x'][li], G['y'][li], G['z'][li], G['bright'][li], bga[:, 0], bga[:, 1]], 1)
            ptf = fit(M, dat, rn2, ptru.clone(), pt, cx, cy, ang, sf=True)
            cr = fisher_xy(M, ptru, pt, cx, cy, ang, rn2)
            res['row'].append(r_); res['smlm'].append(ps.cpu().numpy()); res['sf'].append(pf.cpu().numpy())
            res['sf_true'].append(ptf.cpu().numpy()); res['crlb_model'].append(cr.cpu().numpy())
            res['roi_sum'].append(M.psf_t(ptru[:, 0], ptru[:, 1], ptru[:, 2], cx, cy, ang).sum((2, 3)).mean(1).cpu().numpy())
            res['bg_true'].append(bga.cpu().numpy())
        goff += n; fov0 += nb
        if ntg: need -= ntg
    assert goff == len(D['gt_x']), (goff, len(D['gt_x']))
    out = {k: np.concatenate(v) for k, v in res.items()}
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(fo, **out)
    print(f'iso s{seed}: {len(out["row"])} rows, {it} its, {time.time() - t0:.0f}s', flush=True)


def GA_patch(x, fov, cy, cx):
    from simcode.layers.blocks import extract_patches
    return extract_patches(x, fov, cy.long(), cx.long(), R)


if __name__ == '__main__':
    for sd in [int(s) for s in sys.argv[1].split(',')]:
        run(sd)
