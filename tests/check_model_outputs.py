"""
Outputs of every network checkpoint on a fixed batch (GPU): every detector and emitter network under the models directory
(models.json models, the model folders, models/archive_*, the other shipped weights) and the 4Pi detector / heads of the 4Pi
work directory, loaded with simcode.detector.model.load / simcode.emitter.model.load / simcode.fourpi.loadhead.load_head.
The fixed batch (4 windows of 6 x 64 x 64 photons, a fixed pattern, 4Pi patches) is drawn from a seeded CPU generator; the
emitter networks run on the candidates of the m1809 pass-1 detector with the PSF template of the 1809_2 calibration and with
the Gaussian template. Used to check that a code change leaves every checkpoint's outputs bit-identical (2026-11-15
reorganisation: the reference outputs were made with the code before it).

  python tests/check_model_outputs.py --save <outputs.pt>            write the outputs
  python tests/check_model_outputs.py --ref <outputs.pt>             compare with a saved set (exit code 1 on any difference)
"""
import argparse, glob, os, sys
import torch

dev = 'cuda'


def fixed_batch():
    g = torch.Generator().manual_seed(1234)
    B, T, S = 4, 6, 64
    yy, xx = torch.meshgrid(torch.arange(S).float(), torch.arange(S).float(), indexing='ij')
    lam = torch.full((B, T, S, S), 8.0)
    for b in range(B):
        n = 40
        x = torch.rand(n, generator=g) * (S - 8) + 4; y = torch.rand(n, generator=g) * (S - 8) + 4
        ph = 50 + 2000 * torch.rand(n, T, generator=g)
        for i in range(n):
            gs = torch.exp(-((xx - x[i]) ** 2 + (yy - y[i]) ** 2) / (2 * 1.3 ** 2)) / (2 * 3.14159 * 1.3 ** 2)
            lam[b] += ph[i][:, None, None] * gs
    frames = torch.poisson(lam, generator=g)
    k = 2 * 3.14159 / 2.07
    patt = torch.zeros(B, T, 5)
    for t in range(T):
        a = 0.02 if t < 3 else 1.59
        patt[:, t, 0] = k * torch.cos(torch.tensor(a)); patt[:, t, 1] = k * torch.sin(torch.tensor(a))
        patt[:, t, 2] = (t % 3) * 2.094 + 0.3; patt[:, t, 3] = 0.9; patt[:, t, 4] = 1.0
    img4 = torch.poisson(torch.full((64, 6, 4, 15, 15), 20.0) + 200 * torch.rand(64, 6, 4, 15, 15, generator=g), generator=g)
    doff = torch.rand(64, 4, 2, generator=g) - 0.5
    return dict(frames=frames, patt=patt, img4=img4, doff=doff)


def checkpoints(md):
    """(detector checkpoints, emitter-network checkpoints) under the models directory (by state dict content)"""
    dets, nets = [], []
    for f in sorted(glob.glob(f'{md}/**/*.pt', recursive=True)):
        c = torch.load(f, map_location='cpu', weights_only=False)
        if not isinstance(c, dict) or 'model' not in c: continue
        sd = c['model']; a = c.get('args') or {}
        if 'cin.weight' in sd and 'e0.0.c1.weight' in sd: dets.append(f)
        elif 'wmap.0.weight' in sd and (a.get('arch') in ('sc4n', 'emitter_net') or 'no_illum' in a): nets.append(f)
    return dets, nets


def outputs():
    from simcode import config
    from simcode.pattern.illumination import illum_maps
    from simcode.detector.model import load as load_det
    from simcode.detector.select import select_subpixel
    from simcode.emitter import model as EM
    from simcode.layers.blocks import extract_patches
    from simcode.simulate.simulator import Simulator
    md = config.models_dir(); root = os.path.dirname(os.path.abspath(md))
    p4 = os.path.join(config.root('fourpi'), 'models')
    rel = lambda f: os.path.relpath(f, root)
    D = fixed_batch()
    fr = D['frames'].to(dev); il = illum_maps(D['patt'].to(dev), 64, 64)
    dets, nets = checkpoints(md)
    dets.append(f'{p4}/det4pi/final.pt')
    R = {}
    for f in dets:
        net, _ = load_det(f)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(fr, il) if net.illum else net(fr)
        for k, v in o.items(): R[f'det|{rel(f)}|{k}'] = v.float().cpu()
    det, _ = load_det(f'{md}/m1809/det_m1809.pt')
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dd = det(fr)
        lgm = dd['logit'].float()
        c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=0.2, radius=0.7)
        kp = torch.isfinite(c['logit'])
        cc = {k: c[k][kp] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit')}
        nbr = EM.nbr_map(lgm, cc['b'], cc['iy'], cc['ix'], cc['x'], cc['y'], cc['logit'])
        R['cand|b'] = cc['b'].cpu(); R['cand|x'] = cc['x'].cpu(); R['cand|nbr'] = nbr.cpu()
        col = EM.PSFCol(Simulator(f'{md}/insitu/calib_1809_2_psf3d.json', H=32, W=32, T=6, device=dev))
        b, iy, ix, x, y = cc['b'], cc['iy'], cc['ix'], cc['x'], cc['y']
        p = torch.sigmoid(cc['logit'])
        off = torch.stack([x - ix.float(), y - iy.float()], 1)
        pt = extract_patches(fr, b, iy, ix); nbp = EM.nbr_patches(nbr, b, iy, ix, x, y, p); ilp = extract_patches(il, b, iy, ix)
        for f in nets:
            h = EM.load(f)
            for cn, cl in (('psf', col(off)), ('gauss', None)):
                o = h(pt, None, off, cl, nbp, ilp) if h.illum else h(pt, None, off, cl, nbp)
                for k, v in o.items(): R[f'head|{rel(f)}|{cn}|{k}'] = v.float().cpu()
            R[f'head|{rel(f)}|z3d'] = torch.tensor(float(getattr(h, 'z3d', -1)))
        from simcode.fourpi.loadhead import load_head
        for tag in ('v10', 'v3L'):
            net, _ = load_head(tag)
            o = net(D['img4'].to(dev), D['doff'].to(dev))
            for k, v in o.items():
                if torch.is_tensor(v): R[f'head4pi|{tag}|{k}'] = v.float().cpu()
    print(f'{len(R)} tensors; {len(dets)} detectors, {len(nets)} emitter networks, 2 4Pi heads')
    return R


def compare(A, B):
    same = lambda a, b: a.shape == b.shape and torch.equal(torch.nan_to_num(a, nan=1e30), torch.nan_to_num(b, nan=1e30))
    bad = sorted(k for k in A if k not in B or not same(A[k], B[k]))
    extra = sorted(k for k in B if k not in A)
    for k in bad[:30]: print('DIFF' if k in B else 'MISSING', k)
    for k in extra[:30]: print('EXTRA', k)
    print(f'{len(A)} reference tensors: {len(A) - len(bad)} identical, {len(bad)} differ / missing, {len(extra)} extra')
    return 1 if bad or extra else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default=''); ap.add_argument('--ref', default='')
    a = ap.parse_args()
    R = outputs()
    if a.save: torch.save(R, a.save)
    sys.exit(compare(torch.load(a.ref), R) if a.ref else 0)
