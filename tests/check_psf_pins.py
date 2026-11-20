"""The pixel checks of a torch-psf pin move (docs/TORCH_PSF.md): SIMCODE's PSF renders with one code + torch-psf version
(--save), then with another (--ref); exit code 1 on any difference beyond --tol of the peak (default 0 = bit-identical).
Only SIMCODE names that exist on both sides of the psf-unify migration (2026-11-20) are used.

  2D (--calib, calibration json / psf.zip files): per file and angle None / 0 / 1 (0 / 1 only with per-angle terms):
     vectorial_psf(f, 15, angle).render(pos, grad=True), calib_psf(f, P, angle).render(pos, grad=True) for P = the SPLINE
     ROI (41) and 15 (a json or a vectorial zip: the spline built from the vectorial model; an integrated_spline zip: its
     coefficients), PSF.load(<integrated_spline zip>, angle=angle).render(pos, grad=True), and the coefficient bank of
     calib_psf (old: coefs_angle / coefs, new: knots) of the spline.
  4Pi (--work4pi, optional --plain): the states of tests/check_fourpi_spline.py (psf4pi/{NPC_f_dxy15, MT_f_dxy15,
     NPC_asym_d0.05_2.0}.json, ck['psf'] of models/head4pi_{v10, v3L, v11}.pt, the plain-4Pi in-situ states): fourpi_psf
     forward and its autograd gradients (pos, phase, intensity, bg, every parameter; asymmetric states in both zmodes),
     fourpi_spline knots / evaluate (value and 4 derivatives) / fourpi_channel / the per-component renders (old:
     luts[k][c].render, new: fourpi_component) on SIMCODE's grids (15 px x 4, 15 px x 1, 17 px x 2), also at 2500
     positions (chunked evaluation), and the perturbed member of a refined spline state.

  PYTHONPATH=<old code>:<old torch-psf> python tests/check_psf_pins.py --calib <files> --work4pi <dir> --save pins.npz
  PYTHONPATH=<new code>:<new torch-psf> python tests/check_psf_pins.py --calib <files> --work4pi <dir> --ref pins.npz
Options: --device cpu (default) | cuda, --threads.
"""
import argparse, inspect, json, math, os, sys, time, zipfile
import numpy as np
import torch

from simcode import psf as spsf          # the code on PYTHONPATH (not necessarily this checkout's)

DERIV = ('v', 'dx', 'dy', 'dz', 'dphi')


def _put(R, k, v):
    if isinstance(v, (tuple, list)):
        for i, x in enumerate(v):
            R[f'{k}|{DERIV[i]}'] = x.detach().cpu().numpy()
    else:
        R[k] = v.detach().cpu().numpy()


def _bank(s):
    if hasattr(s, 'coefs_angle'):        # torch-psf before psf-unify: IntegratedSplinePSF
        return s.coefs_angle if s.coefs_angle is not None else s.coefs[None]
    return s.knots


def pins_2d(R, files, dev):
    g = torch.Generator().manual_seed(7)
    N = 64
    pos = torch.cat([3.8 * torch.rand(N, 2, generator=g) - 1.9, 0.5 * torch.rand(N, 1, generator=g) - 0.05], 1)
    pos[:4] = torch.tensor([[0., 0., 0.], [0.3, -0.2, 0.1], [1.99, -1.99, 0.4], [-0.7, 0.4, 0.25]])
    pos = pos.to(dev)
    for f in files:
        t0 = time.time(); nm = os.path.basename(f)
        c = spsf.read_calib(f)
        gauss = c.get('psf_gauss_sigma_um') is not None
        per_angle = spsf.has_angle_psf(c)
        ztype = None
        if spsf.is_psf_zip(f):
            with zipfile.ZipFile(f) as z:
                ztype = json.loads(z.read('psf.json'))['channels'][0]['type']
        with torch.no_grad():
            for a in ([None, 0, 1] if per_angle else [None, 0]):
                if not gauss and (a is None or per_angle):
                    _put(R, f'{nm}|vectorial15|a{a}', spsf.vectorial_psf(f, 15, a, dev).render(pos, True))
                for P in (None, 15):
                    s = spsf.calib_psf(f, P, a, dev)
                    _put(R, f'{nm}|calib{P}|a{a}', s.render(pos) if gauss else s.render(pos, True))
                if ztype == 'integrated_spline':
                    _put(R, f'{nm}|load|a{a}', spsf.PSF.load(f, device=dev, angle=a if per_angle else None).render(pos, True))
            if not gauss:
                _put(R, f'{nm}|bank', _bank(spsf.calib_psf(f, None, None, dev)))
        print(f'{nm}: {ztype or "json"}, per-angle {per_angle}, {time.time() - t0:.0f} s', flush=True)


def fourpi_states(P4, plain):
    out = {}
    for nm in ('NPC_f_dxy15', 'MT_f_dxy15', 'NPC_asym_d0.05_2.0'):
        f = os.path.join(P4, 'psf4pi', nm + '.json')
        if os.path.isfile(f):
            out[nm] = (json.load(open(f))['psf'], (-0.6, 0.6))
    for tag in ('v10', 'v3L', 'v11'):
        f = os.path.join(P4, 'models', f'head4pi_{tag}.pt')
        if os.path.isfile(f):
            ck = torch.load(f, weights_only=False, map_location='cpu')
            out[f'head4pi_{tag}'] = (ck['psf'], tuple(ck['args'].get('zlut') or (-0.6, 0.6)))
    if plain:
        for nm, f in (('plain_astig', 'psf/psf4pi_Cell11_astig.json'), ('insitu_v2', 'p4psf/psf/psf4pi_Cell11_insitu_v2.json'),
                      ('spline_r3', 'p4psf/psf/psf4pi_Cell11_spline_r3.json')):
            f = os.path.join(plain, f)
            if os.path.isfile(f):
                s = json.load(open(f)); out[nm] = (s.get('psf', s), (-0.6, 0.6))
    return out


def _component(L, k, c, pos, grad=False):
    if hasattr(spsf, 'fourpi_component'):
        return spsf.fourpi_component(L, k, c, pos, grad)
    return L.luts[k][c].render(pos, grad)          # before psf-unify


def pins_4pi(R, P4, plain, dev):
    g = torch.Generator().manual_seed(11)
    pos = torch.cat([3 * torch.rand(200, 2, generator=g) - 1.5, 1.1 * torch.rand(200, 1, generator=g) - 0.55], 1).to(dev)
    Phi = (2 * math.pi * torch.rand(200, generator=g) - math.pi).to(dev)
    big = torch.cat([3 * torch.rand(2500, 2, generator=g) - 1.5, 1.1 * torch.rand(2500, 1, generator=g) - 0.55], 1).to(dev)
    seen = {}
    for name, (st, zl) in fourpi_states(P4, plain).items():
        t0 = time.time()
        key = json.dumps(st, sort_keys=True); same = seen.get(key)
        if same and zl == (-0.6, 0.6):
            print(f'{name}: the state of {same}', flush=True); continue
        seen.setdefault(key, name)
        sm = {k: v for k, v in st.items() if k not in spsf.FOURPI_TABLE_KEYS}
        if not same:
            for zm in (('emitter', 'stage') if st['meta'].get('asym') else (None,)):
                tg = f'{name}|{zm}'
                p = spsf.fourpi_psf(sm, device=dev, **({'zmode': zm} if zm else {}))
                phk = 'phi' if 'phi' in inspect.signature(type(p).forward).parameters else 'phi_extra'
                n = 48; ps, ph = pos[:n].clone().requires_grad_(), Phi[:n].clone().requires_grad_()
                inten = torch.linspace(500, 3000, n, device=dev).requires_grad_()
                bg = torch.linspace(0.5, 3, 4 * n, device=dev).reshape(n, 4).requires_grad_()
                w = torch.rand(n, 4, p.roisize, p.roisize, generator=g).to(dev)
                I = p(ps, **{phk: ph}, intensity=inten, bg=bg)
                (I * w).sum().backward()
                for k, v in (('I', I), ('g_pos', ps.grad), ('g_phase', ph.grad), ('g_intensity', inten.grad), ('g_bg', bg.grad)):
                    _put(R, f'{tg}|{k}', v)
                for k, v in p.named_parameters():
                    if v.grad is not None: _put(R, f'{tg}|g_{k}', v.grad)
                with torch.no_grad():
                    _put(R, f'{tg}|forward', p(pos, **{phk: Phi}))
        grids = [dict(P=15, sub=4, zr=zl)] + ([] if st.get('spline') else [dict(P=15, sub=1, zr=zl), dict(P=17, sub=2, zr=(-0.6, 0.6))])
        for gr in grids:
            tag = f'{name}|P{gr["P"]}s{gr["sub"]}'
            L = spsf.fourpi_spline(st, gr['P'], gr['zr'], oversample=gr['sub'], device=dev)
            C = L.n_channels
            _put(R, f'{tag}|knots', L.knots.reshape(3, C, *L.knots.shape[-3:]))
            pp = pos.clone(); pp[:, 2] = pp[:, 2].clamp(*gr['zr'])
            with torch.no_grad():
                _put(R, f'{tag}|evaluate', L.evaluate(pp, Phi, grad=True))
                _put(R, f'{tag}|evaluate_value', L.evaluate(pp, Phi))
                for c in range(C):
                    _put(R, f'{tag}|channel{c}|value', spsf.fourpi_channel(L, pp, Phi, c))
                    _put(R, f'{tag}|channel{c}', spsf.fourpi_channel(L, pp, Phi, c, grad=True))
                    for k in range(3):
                        _put(R, f'{tag}|component{k}{c}', _component(L, k, c, pp, True))
                        _put(R, f'{tag}|component{k}{c}|2500', _component(L, k, c, big))
        if st.get('spline'):
            s2 = json.loads(json.dumps(st)); s2['gamma'] *= 0.9; s2['blur_px2'] += 0.03
            L = spsf.fourpi_spline(s2, 15, device=dev)
            _put(R, f'{name}|perturbed|knots', L.knots.reshape(3, L.n_channels, *L.knots.shape[-3:]))
        print(f'{name}: {time.time() - t0:.0f} s', flush=True)


def compare(R, ref, tol):
    S = dict(np.load(ref))
    if set(S) != set(R):
        print('keys only in the reference:', sorted(set(S) - set(R))[:20], '| only here:', sorted(set(R) - set(S))[:20])
    group = lambda k: k.rsplit('|', 1)[0] if k.rsplit('|', 1)[-1] in DERIV else k
    peak = {}
    for k in set(S) & set(R):
        peak[group(k)] = max(peak.get(group(k), 0.0), float(np.abs(S[k].astype(np.float64)).max()) if S[k].size else 0.0)
    n_id, worst, bad = 0, (0.0, ''), []
    for k in sorted(set(S) & set(R)):
        x, y = R[k], S[k]
        if x.shape != y.shape:
            bad.append(k); print('  SHAPE', k, x.shape, y.shape); continue
        if x.dtype == y.dtype and x.tobytes() == y.tobytes():
            n_id += 1; continue
        d = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) / max(peak[group(k)], 1e-30)
        print(f'  diff {k}: {d:.3e} of the peak', flush=True)
        worst = max(worst, (d, k))
        if d > tol: bad.append(k)
    print(f'{len(set(S) & set(R))} arrays, {n_id} bit-identical; largest |d| / peak {worst[0]:.3e} {worst[1]}; '
          f'{len(bad)} beyond {tol:g}', flush=True)
    return not bad and set(S) == set(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--calib', nargs='*', default=[], help='calibration json / psf.zip files (2D pins)')
    ap.add_argument('--work4pi', default=None, help='the 4Pi work folder (4Pi pins)')
    ap.add_argument('--plain', default=None, help='the plain-4Pi work folder (its in-situ states)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--save', default=None); ap.add_argument('--ref', default=None)
    ap.add_argument('--tol', type=float, default=0.0)
    a = ap.parse_args()
    if bool(a.save) == bool(a.ref): ap.error('one of --save / --ref')
    torch.set_num_threads(a.threads)
    import torch_psf
    print('torch_psf', torch_psf.__file__, '| simcode', os.path.dirname(spsf.__file__), '| device', a.device, flush=True)
    R = {}
    pins_2d(R, a.calib, a.device)
    if a.work4pi: pins_4pi(R, a.work4pi, a.plain, a.device)
    if a.save:
        np.savez(a.save, **R); print(len(R), 'arrays ->', a.save); return
    sys.exit(0 if compare(R, a.ref, a.tol) else 1)


if __name__ == '__main__':
    main()
