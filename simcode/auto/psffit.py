"""In-situ PSF fit of the automatic pipeline (stage d; docs/AUTO_PIPELINE.md sections 2 and 4).

fit(tracks, out, ...) = simcode/insitu/fit_insitu2.py as a function (same algorithm, defaults, random draws and operation
order, so with the same inputs it gives the same result), but with the optics and the camera constants as arguments
(fit_insitu2 hard-codes wavelength 0.70 um, pixel 0.10833 um, offset 99.3 ADU, read noise 1.9 e). Supported terms: Zernike
5-21 + focus, blur 'none' | 'line', photoselection 0 | 3, --init-fit / --freeze-base, checkerboard split. Writes
<out>_psf.pt / _psf.json in the fit_insitu2 format.

lut=True: the alternating fit of section 4. Per iteration the knots of a Catmull-Rom spline table (simcode.psf.conv_spline.ConvSplinePSF
geometry) of the CURRENT vectorial model are built with autograd to the PSF parameters; all per-sample PSF evaluations
(x, y, z gradients, varpro I / bg) run on the spline; the PSF parameters get the exact gradient of the spline loss over all
samples by one backward pass through the table construction. Same optimizer, learning rates and iteration count as the
direct fit; the final residuals / intensities use the vectorial model.

heldout(...): PSF frozen, x, y, z refit per sample (on the spline table, multi-start in z), NLL / chi2 per sample with the
vectorial model; the same samples for every PSF (paired comparison).
pattern_from_intensities(...): k per orientation (FFT + DFT refinement of the 3-step phasor over the pitch range), per-step
phase / modulation / relative intensity.
"""
import json, math, time
import numpy as np, torch
from simcode.psf.vectorial import VectorialPSF
from simcode.psf.conv_spline import ConvSplinePSF
from simcode.insitu.varpro import solve_linear


class Fit:
    def __init__(self, R, optics, dev='cuda', blur='none', photosel=0, kdir=None, init_fit=None, freeze_base=False,
                 init_zcoef=None, readnoise_e=1.9, offset_adu=99.3, gamma_init=0.05, zmax=0.2):
        self.dev, self.R, self.c, self.zmax = dev, R, (R - 1) / 2, zmax
        o = optics
        self.model = VectorialPSF(R, pixelsize=o['pixel_um'], oversample=2, NA=o['NA'], n_imm=o['n_imm'], wavelength=o['wavelength_um'],
                                  n_med=o['n_med'], device=dev)
        m = self.model
        if init_zcoef is not None:
            with torch.no_grad(): m.zcoef.copy_(torch.as_tensor(init_zcoef).to(dev))
        m.zcoef.requires_grad_(True); m.apod.requires_grad_(False); m.NA.requires_grad_(False)
        self.G = G = dict(focus=torch.tensor(0.0, device=dev), logrn=torch.tensor(math.log(readnoise_e), device=dev),
                          offset=torch.tensor(float(offset_adu), device=dev))
        G['focus'].requires_grad_(True)
        self.BL0 = None
        if init_fit:
            S0 = torch.load(init_fit, weights_only=False) if isinstance(init_fit, str) else init_fit
            with torch.no_grad():
                m.zcoef.copy_(S0['zcoef'].to(dev)); G['focus'].fill_(float(S0['focus']))
            if 'blur_px2' in S0.get('extras', {}): self.BL0 = torch.tensor(S0['extras']['blur_px2'][0], device=dev)
            if freeze_base:
                m.zcoef.requires_grad_(False); G['focus'].requires_grad_(False)
        self.BL = None
        if blur == 'line':
            self.BL = torch.zeros(1, 2, device=dev); self.BL[:, 0] = 0.05; self.BL.requires_grad_(True)
        self.GAM = self.ZETA = None
        if photosel:
            assert photosel == 3 and kdir is not None
            kc = torch.tensor(kdir, device=dev, dtype=torch.float32)
            KD = kc / kc.norm(dim=1, keepdim=True)
            self.EDIR = torch.stack([-KD[:, 1], KD[:, 0]], 1)
            self.GAM = torch.tensor(gamma_init, device=dev, requires_grad=True)
            self.ZETA = torch.tensor(1 / 3, device=dev, requires_grad=True)
        self.EXTRA = [v for v in (self.BL, self.GAM, self.ZETA) if v is not None]
        self.blur_mode, self.photosel = blur, photosel

    # ---- per-angle PSF terms (= fit_insitu2.dip_tensor / blur_cov / psf_of)
    def dip_tensor(self, k):
        e = torch.zeros(3, device=self.dev); e[:2] = self.EDIR[k]
        Z = self.ZETA
        base = torch.diag(torch.stack([(1 - Z) / 2, (1 - Z) / 2, Z]))
        return (1 - self.GAM) * base + self.GAM * e[:, None] * e[None, :]

    def blur_cov(self, k):
        if self.BL is None: return self.BL0
        v = self.BL[k]
        return torch.stack([v[0] ** 2, v[1] ** 2, v[0] * v[1]])

    def n_angles(self):
        return 2 if self.GAM is not None else 1

    def kwargs(self, k):
        kw = {}
        if self.BL is not None: kw['blur'] = self.blur_cov(min(k, self.BL.shape[0] - 1))
        elif self.BL0 is not None: kw['blur'] = self.blur_cov(k)
        if self.GAM is not None: kw['dipM'] = self.dip_tensor(k)
        return kw

    def psf_of(self, xyz, focus, model=None):
        model = model or self.model
        if self.BL is None and self.GAM is None and self.BL0 is None:
            return model(xyz, focus)
        na = self.n_angles(); out = [model(xyz, focus, **self.kwargs(k)) for k in range(na)]
        return out[0] if na == 1 else torch.stack(out, 1)

    def extras_json(self):
        r = {}
        if self.BL is not None: r['blur_px2'] = [self.blur_cov(k).detach().cpu().tolist() for k in range(self.BL.shape[0])]
        elif self.BL0 is not None: r['blur_px2'] = [self.BL0.cpu().tolist()]
        if self.GAM is not None:
            r['gamma'] = self.GAM.item(); r['dipole_M'] = [self.dip_tensor(k).detach().cpu().tolist() for k in range(2)]
            r['e_dir'] = self.EDIR.cpu().tolist(); r['zeta'] = self.ZETA.item()
        return r

    # ---- data
    def load(self, D, idx):
        dev = self.dev
        return dict(adu=torch.tensor(D['rois'][idx].astype(np.float32), device=dev), mask=torch.tensor(D['mask'][idx], device=dev),
                    frames=torch.tensor(D['frames'][idx], device=dev), corner=torch.tensor(D['corner'][idx], device=dev, dtype=torch.float32),
                    xy=(torch.tensor(D['xy_init'][idx], device=dev, dtype=torch.float32) - self.c).requires_grad_(True),
                    z=torch.zeros(len(idx), device=dev, requires_grad=True))

    def loss_and_linear(self, S, sl, gain, psf_fn=None):
        adu, mask, frames = S['adu'][sl], S['mask'][sl], S['frames'][sl]
        b = adu.shape[0]; G = self.G
        xyz = torch.cat([S['xy'][sl], self.zmax * torch.sigmoid(S['z'][sl])[:, None]], 1)
        psf = (psf_fn or self.psf_of)(xyz, G['focus'].expand(b))
        data = (adu - G['offset']) * gain
        rn2 = torch.exp(2 * G['logrn'])
        group = torch.where(mask, (frames % 6) // 3, torch.zeros_like(frames))
        if psf.dim() == 4:
            psf = torch.gather(psf, 1, group[..., None, None].expand(-1, -1, psf.shape[-2], psf.shape[-1]))
            I, bg = solve_linear(psf.reshape(b, adu.shape[1], -1).detach(), data.reshape(b, adu.shape[1], -1).detach(), mask, group, rn2.detach())
            mu = (I[..., None, None] * psf + torch.gather(bg, 1, group)[..., None, None]).clamp(min=1e-3)
        else:
            I, bg = solve_linear(psf.reshape(b, -1).detach(), data.reshape(b, adu.shape[1], -1).detach(), mask, group, rn2.detach())
            mu = (I[..., None, None] * psf[:, None] + torch.gather(bg, 1, group)[..., None, None]).clamp(min=1e-3)
        var = mu + rn2
        nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
        nllm = torch.where(mask[..., None, None], nll, torch.zeros_like(nll))
        return nllm.sum(), I, bg, mu, var, data, nllm.sum((1, 2, 3))

    # ---- spline table of the current model (section 4), differentiable in the PSF parameters
    def lut_init(self, sub=4, hz=0.01, xmax=2.0):
        m = self.model
        self.luts = []
        for k in range(self.n_angles()):
            src = VectorialPSF(self.R, pixelsize=m.pixelsize, NA=float(m.NA), n_imm=m.n_imm, n_med=float(m.n_med), wavelength=m.wavelength,
                               zernike_indices=m.zidx, npupil=m.npupil, oversample=m.os, device=self.dev)
            with torch.no_grad(): src.zcoef.copy_(m.zcoef)
            self.luts.append(ConvSplinePSF(src, float(self.G['focus'].detach()), self.zmax, sub=sub, hz=hz, xmax=xmax))
        L = self.luts[0]
        Pl = L.zstack.shape[1] // sub
        self.big = VectorialPSF(Pl, pixelsize=m.pixelsize, NA=float(m.NA), n_imm=m.n_imm, n_med=float(m.n_med), wavelength=m.wavelength,
                                zernike_indices=m.zidx, npupil=m.npupil, oversample=m.os, device=self.dev)
        self.big.zcoef = m.zcoef; self.big.NA = m.NA; self.big.apod = m.apod; self.big.saf = m.saf   # shared parameters
        zs = L.z0 + L.hz * torch.arange(L.nz, device=self.dev, dtype=torch.float64)
        a = torch.arange(sub, device=self.dev, dtype=torch.float64) / sub
        Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')
        self.lut_pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
        self.lut_geom = (L.nz, sub, Pl)

    def lut_knots(self, k):
        """knots [nz, Pl S, Pl S] of angle k's PSF (graph to the PSF parameters), = ConvSplinePSF.__init__"""
        nz, S, Pl = self.lut_geom
        f = self.G['focus'].expand(len(self.lut_pos))
        imgs = self.big(self.lut_pos, f, **self.kwargs(k))
        imgs = imgs.reshape(nz, S, S, Pl, Pl)
        return imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(nz, Pl * S, Pl * S).contiguous()

    def lut_psf_fn(self, knots):
        def fn(xyz, focus):
            out = [spline_eval(L, K, xyz) for L, K in zip(self.luts, knots)]
            return out[0] if len(out) == 1 else torch.stack(out, 1)
        return fn


def spline_eval(L, K, pos):
    """value of ConvSplinePSF L's spline with knots K [nz, nv, nu] at pos [N, 3] -> [N, P, P]; = ConvSplinePSF.evaluate (same knot
    geometry, clamping and Catmull-Rom weights), but separable: only the 4 x 4P x 4P knots that the P x P output pixels use are
    gathered, then contracted along z, y, x (instead of a grouped stride-S conv2d over the whole (P-1)S+4 block).
    Differentiable in pos and in K."""
    N, P, S = pos.shape[0], L.P, L.S
    x = pos[:, 0].clamp(-L.xmax, L.xmax); y = pos[:, 1].clamp(-L.xmax, L.xmax)
    tz = ((pos[:, 2] - L.z0) / L.hz).clamp(1, L.nz - 2 - 1e-4)
    zyx = torch.stack([tz, L.oS0 - y * S, L.oS0 - x * S], 1)
    ix = torch.floor(zyx); offs = zyx - ix
    i0 = ix.long() - 1
    if not hasattr(L, '_sep'):
        e4 = torch.arange(4, device=pos.device)
        j = (torch.arange(P, device=pos.device)[:, None] * S + e4[None]).reshape(-1)     # [4P]: knot offsets used by the pixels
        L._sep = (e4, j)
    e4, j = L._sep
    base = i0[:, 0] * L.HW + i0[:, 1] * L.W + i0[:, 2]
    idx = base[:, None, None, None] + (e4 * L.HW)[None, :, None, None] + (j * L.W)[None, None, :, None] + j[None, None, None, :]
    blk = K.reshape(-1)[idx]                                                   # [N, 4, 4P, 4P]
    w = (offs[..., None] ** e4) @ L.G                                          # [N, 3, 4] (z, y, x)
    v = torch.einsum('nz,nzab->nab', w[:, 0], blk).reshape(N, P, 4, 4 * P)
    v = torch.einsum('ny,nayb->nab', w[:, 1], v).reshape(N, P, P, 4)
    return torch.einsum('nx,nabx->nab', w[:, 2], v)


def _opt(F, S):
    m = F.model
    return torch.optim.Adam([{'params': [S['xy'], S['z']], 'lr': 0.03},
                             {'params': [F.G['focus']], 'lr': 0.005},
                             {'params': [m.zcoef], 'lr': 0.002}, {'params': [m.apod, m.NA], 'lr': 0.003}]
                            + ([{'params': [F.BL], 'lr': 0.003}] if F.BL is not None else [])
                            + ([{'params': [F.GAM], 'lr': 0.003}] if F.GAM is not None else [])
                            + ([{'params': [F.ZETA], 'lr': 0.003}] if F.ZETA is not None else []))


def select(D, nsamples, split='', tile=16, seed=0):
    """= fit_insitu2's sample selection (np.random.seed(0) state)"""
    np.random.seed(seed)
    Ball = len(D['rois'])
    if split:
        pos_ = D['corner'] + D['xy_init']
        par = ((pos_[:, 0] // tile).astype(int) + (pos_[:, 1] // tile).astype(int)) % 2
        cand = np.flatnonzero(par == int(split[2]))
        return np.sort(cand[np.random.permutation(len(cand))[:nsamples]])
    return np.sort(np.random.permutation(Ball)[:nsamples])


def fit(tracks, out, optics, gain, nsamples=8000, iters=600, zmax=0.2, blur='none', photosel=0, kdir=None, init_fit=None,
        freeze_base=False, init_zcoef=None, readnoise_e=1.9, offset_adu=99.3, split='', tile=16, lut=False, lut_sub=8,
        lut_inner=1, lut_polish=50, dev='cuda', log=print, return_state=False):
    torch.manual_seed(0); np.random.seed(0)
    D = np.load(tracks) if isinstance(tracks, str) else tracks
    R = D['rois'].shape[-1]
    F = Fit(R, optics, dev, blur, photosel, kdir, init_fit, freeze_base, init_zcoef, readnoise_e, offset_adu, zmax=zmax)
    sel = select(D, nsamples, split, tile)
    S = F.load(D, sel)
    npx = S['mask'].sum().item() * R * R
    opt = _opt(F, S)
    chunks = torch.arange(len(sel), device=dev).split(2000)
    m, G = F.model, F.G
    if lut: F.lut_init(lut_sub)
    psf_params = [v for v in [G['focus'], m.zcoef] + F.EXTRA if v.requires_grad]
    torch.cuda.synchronize(); t0 = time.time()
    for it in range(iters):
        opt.zero_grad(); tot = 0
        warm = it < 100
        if not lut or it >= iters - lut_polish:      # direct (vectorial) step; with lut: the last lut_polish iterations
            for ch in chunks:
                l = F.loss_and_linear(S, ch, gain)[0] / npx; l.backward(); tot += l.item()
        else:
            # knots: once for the position-only warm-up, then rebuilt (with the graph to the PSF parameters) every
            # lut_inner iterations; the PSF parameters are updated on the rebuild iterations
            rebuild = it == 0 or (not warm and (it - 100) % lut_inner == 0)
            if rebuild:
                with torch.set_grad_enabled(not warm):
                    K = [F.lut_knots(k) for k in range(F.n_angles())]
                KL = [k_.detach().requires_grad_(not warm) for k_ in K]
            fn = F.lut_psf_fn(KL)
            for ch in chunks:
                l = F.loss_and_linear(S, ch, gain, psf_fn=fn)[0] / npx; l.backward(); tot += l.item()
            if rebuild and not warm and psf_params:
                torch.autograd.backward(K, [k_.grad for k_ in KL])
            for k_ in KL:
                k_.grad = None
        if warm:
            for v in list(G.values()) + [m.zcoef, m.apod, m.NA] + F.EXTRA: v.grad = None
        opt.step()
        with torch.no_grad(): m.NA.clamp_(1.0, 1.515)
        if it % 100 == 0 or it == iters - 1:
            log(f'{it:5d} nll/px {tot:.4f}  focus {G["focus"].item():.3f}  z med {(zmax*torch.sigmoid(S["z"])).median().item():.3f}  '
                f'|Z| {m.zcoef.norm().item():.3f}' + (f'  extras {json.dumps(F.extras_json())}' if F.EXTRA else ''))
    torch.cuda.synchronize(); t_fit = time.time() - t0
    with torch.no_grad():
        rs, nl, tot_v = [], [], 0.0
        for ch in chunks:
            nll, I, bg, mu, var, data, _ = F.loss_and_linear(S, ch, gain)
            rs.append(((data - mu) / var.sqrt())[S['mask'][ch]]); tot_v += nll.item() / npx
        rs = torch.cat(rs)
    res = dict(gain=gain, readnoise_e=torch.exp(G['logrn']).item(), offset=G['offset'].item(), focus=G['focus'].item(),
               NA=m.NA.item(), apod=m.apod.tolist(), zernike=m.zcoef.tolist(), resid_var=rs.var().item(), zmax=zmax,
               blur_mode=blur, photosel=photosel, split=split, tile=tile, nll_per_px=tot_v, nll_per_px_last_iter=tot,
               n_train=int(len(sel)), lut=bool(lut), lut_sub=lut_sub, lut_polish=lut_polish if lut else 0, fit_s=t_fit, z_median=float((zmax * torch.sigmoid(S['z'])).median()),
               optics=optics, **F.extras_json())
    log(f'normalized residual var {res["resid_var"]:.4f}, nll/px (vectorial) {tot_v:.5f}, fit {t_fit:.0f} s')
    if out:
        json.dump(res, open(out + '_psf.json', 'w'), indent=1)
        ex = F.extras_json()
        torch.save(dict(zcoef=m.zcoef.detach().cpu(), NA=m.NA.detach().cpu(), apod=m.apod.detach().cpu(),
                        **{k: v.detach().cpu() for k, v in G.items()}, gain=gain, zmax=zmax, optics=optics,
                        **({'extras': ex} if ex else {})), out + '_psf.pt')
    if return_state:
        with torch.no_grad():
            Is = torch.cat([F.loss_and_linear(S, ch, gain)[1] for ch in chunks]).cpu()
        st = dict(I=Is, r=(S['xy'] + F.c + S['corner']).detach().cpu(), z=(zmax * torch.sigmoid(S['z'])).detach().cpu(),
                  frames=S['frames'].cpu(), mask=S['mask'].cpu(), sel=sel)
        return res, st
    return res


def model_from_pt(pt, R, dev='cuda'):
    """frozen Fit object from a _psf.pt (this module's or fit_insitu2's; optics default = the DMD setup)"""
    S = torch.load(pt, weights_only=False) if isinstance(pt, str) else pt
    o = S.get('optics') or dict(pixel_um=0.10833, NA=float(S['NA']), wavelength_um=0.70, n_imm=1.518, n_med=1.33)
    o = dict(o); o['NA'] = float(S['NA'])
    F = Fit(R, o, dev, zmax=float(S.get('zmax', 0.2)), readnoise_e=float(torch.exp(torch.as_tensor(S['logrn']))), offset_adu=float(S['offset']))
    with torch.no_grad():
        F.model.zcoef.copy_(S['zcoef'].to(dev)); F.G['focus'].fill_(float(S['focus']))
    ex = S.get('extras', {})
    if 'blur_px2' in ex: F.BL0 = torch.tensor(ex['blur_px2'][0], device=dev)
    if 'dipole_M' in ex:
        M = torch.tensor(ex['dipole_M'], device=dev)
        F.GAM = torch.tensor(0.0, device=dev)       # marks per-angle; dip_tensor overridden below
        F.dip_tensor = lambda k, M=M: M[k]
    for p in [F.model.zcoef, F.G['focus']]: p.requires_grad_(False)
    return F, float(S['gain'])


def heldout(tracks, pts, test='cb1', nsamples=6000, iters=300, zstarts=(0.02, 0.08, 0.15), tile=16, seed=1, dev='cuda', log=print):
    """per-sample NLL / chi2 of several frozen PSFs on the same held-out samples. Returns {name: dict(nll [n], chi2 [n], ...)}"""
    D = np.load(tracks) if isinstance(tracks, str) else tracks
    R = D['rois'].shape[-1]
    rng = np.random.RandomState(seed)
    pos_ = D['corner'] + D['xy_init']
    par = ((pos_[:, 0] // tile).astype(int) + (pos_[:, 1] // tile).astype(int)) % 2
    cand = np.flatnonzero(par == int(test[2])) if test else np.arange(len(pos_))
    sel = np.sort(cand[rng.permutation(len(cand))[:nsamples]])
    out = {}
    for name, pt in pts.items():
        F, gain = model_from_pt(pt, R, dev)
        F.lut_init(8)
        with torch.no_grad():
            K = [F.lut_knots(k) for k in range(F.n_angles())]
        fn = F.lut_psf_fn(K)
        best = None
        for z0 in zstarts:
            S = F.load(D, sel)
            with torch.no_grad(): S['z'].fill_(math.log(z0 / (F.zmax - z0)))
            o = torch.optim.Adam([S['xy'], S['z']], lr=0.03)
            chunks = torch.arange(len(sel), device=dev).split(3000)
            for it in range(iters):
                o.zero_grad()
                for ch in chunks: F.loss_and_linear(S, ch, gain, psf_fn=fn)[0].backward()
                o.step()
            with torch.no_grad():
                nl, c2 = [], []
                for ch in chunks:
                    _, I, bg, mu, var, data, per = F.loss_and_linear(S, ch, gain)
                    msk = S['mask'][ch][..., None, None].float()
                    c2.append((((data - mu) ** 2 / var) * msk).sum((1, 2, 3)) / (msk.sum((1, 2, 3)) * R * R)); nl.append(per)
                nl = torch.cat(nl).cpu().numpy(); c2 = torch.cat(c2).cpu().numpy()
            if best is None: best = dict(nll=nl, chi2=c2)
            else:
                b = nl < best['nll']; best['nll'] = np.where(b, nl, best['nll']); best['chi2'] = np.where(b, c2, best['chi2'])
        npx = D['mask'][sel].sum(1) * R * R
        best.update(nll_per_px=float(best['nll'].sum() / npx.sum()), chi2_per_px=float(np.mean(best['chi2'])), n=int(len(sel)))
        out[name] = best
        log(f'heldout {name}: nll/px {best["nll_per_px"]:.5f}, chi2/px {best["chi2_per_px"]:.4f} ({len(sel)} samples, {test})')
    return out


def paired(H, a, b):
    """paired NLL difference b - a per sample: mean, se, z"""
    d = H[b]['nll'] - H[a]['nll']
    se = d.std() / math.sqrt(len(d))
    return dict(mean=float(d.mean()), se=float(se), z=float(d.mean() / se) if se > 0 else 0.0)


def pattern_from_intensities(st, pitch_px=(1.6, 2.8), dev='cuda', log=print):
    """k [2,2] rad/px, per-step phase [6], modulation [6], relint [6] from per-frame track intensities (blocked order:
    slot = frame % 6, orientation = slot // 3). Model I_s = A relint_s (1 + m_s cos(k_a . r - phi_s)); positions = pixel-centre
    coordinates of the movie."""
    I, fr, mk, r = st['I'].numpy(), st['frames'].numpy(), st['mask'].numpy(), st['r'].numpy()
    med = np.array([np.median(I[b][mk[b]]) if mk[b].any() else 0 for b in range(len(I))])
    on = mk & (I > 0.25 * med[:, None])
    res = dict(k=[], phase=[0.0] * 6, mod=[0.0] * 6, relint=[0.0] * 6, n=[], pitch_px=[])
    for a in range(2):
        rows, vs = [], []
        for b in range(len(I)):
            f = fr[b]
            for t in range(I.shape[1] - 2):
                if not mk[b, t] or f[t] % 6 != 3 * a: continue
                if not (on[b, t:t + 3].all() and f[t + 2] == f[t] + 2): continue
                v = I[b, t:t + 3]; rows.append(b); vs.append(v / v.mean())
        rows = np.array(rows); vs = np.array(vs)
        if len(rows) < 200: raise RuntimeError(f'pattern: only {len(rows)} complete fully-on triples for orientation {a}')
        z = (vs * np.exp(-2j * np.pi * np.arange(3) / 3)).sum(1)
        p = r[rows]
        # FFT of the phasor binned at 0.25 px, peak inside the pitch annulus (both signs of k), then DFT refinement
        h = 0.25; L = int(np.ceil((p.max(0) - p.min(0)).max() / h)) + 1; Nf = 1 << int(np.ceil(np.log2(2 * L)))
        ij = np.round((p - p.min(0)) / h).astype(int)
        g = np.zeros((Nf, Nf), complex); np.add.at(g, (ij[:, 1], ij[:, 0]), z)
        Fg = np.abs(np.fft.fft2(g))
        kf = 2 * np.pi * np.fft.fftfreq(Nf, d=h)
        KX, KY = np.meshgrid(kf, kf)
        kk = np.hypot(KX, KY)
        ok = (kk > 2 * np.pi / pitch_px[1]) & (kk < 2 * np.pi / pitch_px[0])
        j = np.argmax(np.where(ok, Fg, 0))
        k0 = np.array([KX.flat[j], KY.flat[j]])        # fft2 kernel exp(-i k.r): peak at k where z ~ exp(+i k.r)
        P = torch.tensor(p, device=dev, dtype=torch.float64); Z = torch.tensor(z, device=dev)
        kt = torch.tensor(k0, device=dev, dtype=torch.float64, requires_grad=True)
        o = torch.optim.LBFGS([kt], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')
        def clo():
            o.zero_grad(); v = -(Z * torch.exp(-1j * (P @ kt))).sum().abs() / len(Z); v.backward(); return v
        o.step(clo)
        k = kt.detach().cpu().numpy()
        ph = p @ k
        for s in range(3):
            X = np.stack([np.ones_like(ph), np.cos(ph), np.sin(ph)], 1)
            c = np.linalg.lstsq(X, vs[:, s], rcond=None)[0]
            res['phase'][3 * a + s] = float(np.arctan2(c[2], c[1])); res['mod'][3 * a + s] = float(np.hypot(c[1], c[2]) / c[0])
            res['relint'][3 * a + s] = float(c[0])
        res['k'].append(k.tolist()); res['n'].append(int(len(rows))); res['pitch_px'].append(float(2 * np.pi / np.hypot(*k)))
        log(f'pattern orientation {a}: k ({k[0]:.5f}, {k[1]:.5f}) rad/px, pitch {res["pitch_px"][-1]:.4f} px, {len(rows)} triples, '
            f'phases {np.round(res["phase"][3*a:3*a+3], 3).tolist()}, mod {np.round(res["mod"][3*a:3*a+3], 3).tolist()}')
    # relint: per-orientation mean of the triple-normalised values is 1 by construction; the angle ratio from the frame means
    sec = [0 if abs(k[0]) > abs(k[1]) else 1 for k in res['k']]
    res['sectors'] = sec; res['sector_ok'] = sec[0] != sec[1]
    return res


def centre_vs_z(st, D, px_nm, zbins=(0.0, 0.03, 0.06, 0.1, 0.15, 0.21)):
    """z-dependent lateral offset of the crop centre (docs/AUTO_PIPELINE.md section 5 note): d = vectorial-fit position minus
    the track's Gaussian centre (corner + xy_init), per sample, against the fitted z. Returns the slope (nm per 100 nm z) and
    the z-binned medians (nm) of dx, dy."""
    sel = st['sel']
    g = D['corner'][sel] + D['xy_init'][sel]
    d = (st['r'].numpy() - g) * px_nm; z = st['z'].numpy()
    out = dict(n=int(len(z)))
    for i, ax in enumerate('xy'):
        p = np.polyfit(z, d[:, i], 1); out[f'slope_{ax}_nm_per_100nm'] = float(p[0] * 0.1)
        out[f'median_{ax}_nm_by_z'] = [float(np.median(d[(z >= lo) & (z < hi), i])) if ((z >= lo) & (z < hi)).sum() > 50 else None
                                       for lo, hi in zip(zbins[:-1], zbins[1:])]
    out['z_bins_um'] = list(zbins)
    return out
