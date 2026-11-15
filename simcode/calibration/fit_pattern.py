"""
Stage B: per-frame intensities of all track samples (PSF + camera frozen from stage A), then fit the illumination
pattern:  I[b,t] = A[b,angle] * relint[s] * (1 + m[s] * cos(k[angle] . r_b - phi[s] - dphi[angle](t)))
s = t % 6, angle = s // 3. dphi: piecewise-linear phase drift per angle.
Only frames where the emitter is on for the whole frame are used (it is also bright in the previous and next frame).
"""
import argparse, json, math, pickle
import numpy as np, torch
if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from simcode.psf import VectorialPSF
    from simcode.calibration import SFCFG_05, data, out

    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', default=out('tracks_0.5nM.npz'))
    ap.add_argument('--stageA', default=out('insitu_A_apod_stageA.pt'))
    ap.add_argument('--oldmod', default=SFCFG_05)
    ap.add_argument('--zmax', type=float, default=0.2)
    ap.add_argument('--knot_frames', type=int, default=1500)
    ap.add_argument('--intensities', default='')   # precomputed (fit_psf_tracks); skips the refit
    ap.add_argument('--out', default=out('pattern_0.5nM'))
    a = ap.parse_args()
    a.oldmod = data(a.oldmod)
    dev = 'cuda'

    if a.intensities:
        _d = torch.load(a.intensities)
        I, r, frames, mask = _d['I'].to(dev), _d['r'].to(dev), _d['frames'].to(dev), _d['mask'].to(dev)
        B, T = I.shape
    else:
        S = torch.load(a.stageA)
        D = np.load(a.tracks)
        adu = torch.tensor(D['rois'].astype(np.float32), device=dev)
        mask = torch.tensor(D['mask'], device=dev)
        frames = torch.tensor(D['frames'], device=dev)
        corner = torch.tensor(D['corner'], device=dev, dtype=torch.float32)
        B, T, R, _ = adu.shape
        c = (R - 1) / 2

        model = VectorialPSF(R, oversample=2, NA=float(S['NA']), wavelength_um=0.70, n_med=1.33, device=dev)
        with torch.no_grad():
            model.zcoef.copy_(S['zcoef'].to(dev)); model.apod.copy_(S['apod'].to(dev))
        for prm in model.parameters(): prm.requires_grad_(False)
        g, rn2, off, focus = 0.56, float(torch.exp(2 * S['logrn'])), float(S['offset']), float(S['focus'])
        data = (adu - off) * g

        # ---- per-sample refit (x, y, z constant; intensity and bg per frame)
        sp = torch.nn.functional.softplus
        bg0 = torch.quantile(data.reshape(B, T, -1), 0.3, dim=2).clamp(min=0.3)
        I0 = (data - bg0[..., None, None]).clamp(min=0).sum((2, 3)).clamp(min=1) * 1.4
        P = dict(xy=torch.tensor(D['xy_init'], device=dev, dtype=torch.float32) - c, z=torch.zeros(B, device=dev),
                 Iraw=I0 + torch.log(-torch.expm1(-I0)), bg=bg0.clone())
        for v in P.values(): v.requires_grad_(True)
        opt = torch.optim.Adam(P.values(), lr=0.03)

        def nll(idx):
            psf = model(torch.cat([P['xy'][idx], a.zmax * torch.sigmoid(P['z'][idx])[:, None]], 1),
                        torch.full((len(idx),), focus, device=dev))
            mu = (sp(P['Iraw'][idx])[..., None, None] * psf[:, None] + P['bg'][idx][..., None, None]).clamp(min=1e-3)
            var = mu + rn2
            l = 0.5 * (data[idx] - mu) ** 2 / var + 0.5 * torch.log(var)
            return torch.where(mask[idx][..., None, None], l, torch.zeros_like(l)).sum()

        chunks = torch.arange(B, device=dev).split(4000)
        for it in range(400):
            opt.zero_grad()
            for ch in chunks: nll(ch).backward()
            opt.step()
        I = sp(P['Iraw']).detach()
        r = (P['xy'].detach() + c + corner)  # global (x=col, y=row) px
        print(f'refit {B} samples; median photons/frame {I[mask].median().item():.0f}')


    # ---- select fully-on frames
    on = mask.clone()
    Im = torch.where(mask, I, torch.zeros_like(I))
    med = (Im.sum(1) / mask.sum(1)).clamp(min=1)
    bright = Im > 0.25 * med[:, None]
    full = bright.clone()
    full[:, 1:] &= bright[:, :-1]; full[:, :-1] &= bright[:, 1:]
    full[:, 0] = False; full[:, -1] = False
    full &= mask
    bi, ti = torch.nonzero(full, as_tuple=True)
    fr = frames[bi, ti]; Iv = I[bi, ti]; rv = r[bi]
    s = fr % 6; ang = s // 3
    print(f'{len(bi)} fully-on frames used for pattern fit')
    if not a.intensities: torch.save(dict(I=I.cpu(), r=r.cpu(), frames=frames.cpu(), mask=mask.cpu(), full=full.cpu()), a.out + '_intensities.pt')

    # ---- pattern model, init from existing SIMFLUX estimate
    old = pickle.load(open(a.oldmod, 'rb'))['mod'][0]
    k_init = torch.tensor(np.stack([old['k'][0][:2], old['k'][3][:2]]), device=dev, dtype=torch.float32)
    nk = int(frames.max().item()) // a.knot_frames + 2
    Q = dict(k=k_init.clone(), phi=torch.tensor(np.array(old['phase']), device=dev, dtype=torch.float32),
             m=torch.full((6,), 0.7, device=dev), logrel=torch.zeros(6, device=dev),
             dphi=torch.zeros(2, nk, device=dev))
    A = torch.zeros(B, 2, device=dev)
    # init A as mean intensity per sample/angle
    for an in range(2):
        sel = ang == an
        A[:, an].index_add_(0, bi[sel], Iv[sel])
        cnt = torch.zeros(B, device=dev).index_add_(0, bi[sel], torch.ones_like(Iv[sel]))
        A[:, an] = torch.log((A[:, an] / cnt.clamp(min=1)).clamp(min=10))
    Q['logA'] = A
    for v in Q.values(): v.requires_grad_(True)

    def phase_drift(fr, an):
        t = fr.float() / a.knot_frames
        i0 = t.floor().long().clamp(max=nk - 2); w = t - i0
        return Q['dphi'][an, i0] * (1 - w) + Q['dphi'][an, i0 + 1] * w

    def predict():
        kr = (Q['k'][ang] * rv).sum(1)
        rel = torch.exp(Q['logrel'] - Q['logrel'].mean())
        mod = rel[s] * (1 + Q['m'][s] * torch.cos(kr - Q['phi'][s] - phase_drift(fr, ang)))
        return torch.exp(Q['logA'][bi, ang]) * mod

    w = 1.0 / (Iv + 100.0)
    opt = torch.optim.Adam([{'params': [Q['logA'], Q['logrel'], Q['m'], Q['phi'], Q['dphi']], 'lr': 0.02},
                            {'params': [Q['k']], 'lr': 2e-4}])
    for it in range(3000):
        opt.zero_grad()
        res = predict() - Iv
        loss = (torch.nn.functional.huber_loss(res * w.sqrt(), torch.zeros_like(res), delta=3.0, reduction='sum')
                + 10.0 * (Q['dphi'][:, 1:] - Q['dphi'][:, :-1]).pow(2).sum())
        loss.backward(); opt.step()
        with torch.no_grad():
            Q['m'].clamp_(0, 1); Q['dphi'] -= Q['dphi'].mean(1, keepdim=True)
        if it % 500 == 0 or it == 2999:
            print(f'{it:5d} loss {loss.item():.1f}  depth {np.round(Q["m"].tolist(), 3)}')

    with torch.no_grad():
        res = (predict() - Iv) / Iv.mean()
        kk = Q['k'].cpu().numpy()
        out = dict(k=kk.tolist(), pitch_nm=(2 * np.pi / np.linalg.norm(kk, axis=1) * 108.33).tolist(),
                   depth=Q['m'].tolist(), phase=Q['phi'].tolist(), relint=torch.exp(Q['logrel'] - Q['logrel'].mean()).tolist(),
                   rel_residual_rms=res.std().item(), n_frames=len(bi), old_depth=[float(v) for v in old['depth']])
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out + '.json', 'w'), indent=1)
    torch.save(dict(**{k: v.detach().cpu() for k, v in Q.items() if k != 'logA'}, knot_frames=a.knot_frames), a.out + '.pt')

    # diagnostic: modulation curve, intensity normalized by A vs phase, per step
    with torch.no_grad():
        kr = (Q['k'][ang] * rv).sum(1)
        phase = torch.remainder(kr - Q['phi'][s] - phase_drift(fr, ang), 2 * math.pi).cpu().numpy()
        In = (Iv / torch.exp(Q['logA'][bi, ang])).cpu().numpy(); sn = s.cpu().numpy()
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for k in range(6):
        sel = sn == k
        h = ax.flat[k].hexbin(phase[sel], In[sel], gridsize=60, extent=(0, 2 * np.pi, 0, 3), mincnt=1, bins='log')
        xx = np.linspace(0, 2 * np.pi, 100)
        ax.flat[k].plot(xx, out['relint'][k] * (1 + out['depth'][k] * np.cos(xx)), 'r')
        ax.flat[k].set_title(f'step {k}: depth {out["depth"][k]:.2f}')
    plt.tight_layout(); plt.savefig(a.out + '.png', dpi=70)
