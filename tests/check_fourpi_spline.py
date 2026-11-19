"""The switch of SIMCODE's 4Pi PSF to torch-psf (2026-11-20): simcode.psf.fourpi_psf / fourpi_spline / fourpi_channel
(torch_psf FourPiPSF / FourPiSplinePSF) vs the removed SIMCODE 4Pi code (simcode/fourpi/coherent_psf.py PSF4Pi /
PSF4PiA / psf_from_state, simcode/fourpi/spline_table.py LUT4Pi), read from git at OLD (31197be8) into a temporary
package. Needs the torch-psf pin of that commit (4b0ffc3: the old code uses torch_psf.vectorial.gauss_blur_os / BLUR_PAD).
Per state, max |difference| / peak (0 = bit-identical) of
  - the model: forward(pos, phi_extra=Phi) (4 channels, z -0.55 .. 0.55 um, lateral offsets +-1.5 px), with the
    intensity / bg arguments, and the autograd gradients of a fixed weighted sum of the images with respect to every
    parameter and to pos / phi_extra / intensity / bg (what the bead fit, analysis/fourpi/fit_psf4pi.py, uses);
    asymmetric states in both zmodes (PSF4PiA vs FourPiPSF with depth_um)
  - the spline table on every grid SIMCODE builds (heads: 15 px, 4 x oversampled, z of the checkpoint's zlut or +-0.6;
    prior template: sub 1; scene simulator: 17 px, sub 2): knots, evaluate (value and the 4 derivatives) and
    fourpi_channel vs LUT4Pi.evaluate_ch / evaluate_ch_grad
  - the in-situ states (arms / reg / beta, the refined spline file and a perturbed member of it) when present
States: <4Pi work folder>/psf4pi/{NPC_f_dxy15, MT_f_dxy15, NPC_asym_d0.05_2.0}.json, ck['psf'] of head4pi_{v10, v3L, v11}.pt,
and with --plain <plain-4Pi work folder>: psf/psf4pi_Cell11_astig.json, p4psf/psf/psf4pi_Cell11_{insitu_v2, spline_r3}.json.

  PYTHONPATH=<this checkout>:<torch-psf 4b0ffc3> python tests/check_fourpi_spline.py --work <4Pi work folder> [--plain <dir>]
      [--device cpu|cuda] [--quick]
Exit code 1 on any difference beyond --tol of the peak (default 0 = bit-identical). The tables of states with 'reg' are
reported, not counted: FourPiPSF channel_offsets bake the offsets in by a pupil phase ramp, LUT4Pi by a shifted model
position (the same image up to rounding).
"""
import argparse, json, math, os, subprocess, sys, tempfile
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OLD = '31197be8'


def old_modules():
    """the removed SIMCODE 4Pi code from git (OLD) as the package old4pi in a temporary folder"""
    d = tempfile.mkdtemp(prefix='old4pi_')
    os.makedirs(os.path.join(d, 'old4pi'))
    rep = {'from simcode.psf import VectorialPSF, gauss_blur_os, dipole_tensor': 'from torch_psf.vectorial import VectorialPSF, gauss_blur_os, dipole_tensor',
           'from simcode.psf import CubicSplinePSF': 'from torch_psf.spline import CubicSplinePSF',
           'from simcode.fourpi.coherent_psf import': 'from old4pi.coherent_psf import'}
    for f in ('coherent_psf.py', 'spline_table.py'):
        s = subprocess.run(['git', '-C', REPO, 'show', f'{OLD}:simcode/fourpi/{f}'], check=True, capture_output=True, text=True).stdout
        for a, b in rep.items():
            s = s.replace(a, b)
        open(os.path.join(d, 'old4pi', f), 'w').write(s)
    open(os.path.join(d, 'old4pi', '__init__.py'), 'w').close()
    sys.path.insert(0, d)
    import old4pi.coherent_psf as C, old4pi.spline_table as T
    return C, T


def states(P4, plain):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True, help='the 4Pi work folder')
    ap.add_argument('--plain', default=None, help='the plain-4Pi work folder (in-situ states)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--quick', action='store_true', help='head grid only (no sub 1 / P 17 grids)')
    ap.add_argument('--tol', type=float, default=0.0, help='largest accepted |difference| / peak (default 0: bit-identical)')
    a = ap.parse_args()
    dev = a.device
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    import torch_psf
    from simcode.psf import fourpi_psf, fourpi_spline, fourpi_channel
    C, T = old_modules()
    print('torch_psf', torch_psf.__file__, '| old code', OLD, '| device', dev, flush=True)
    bad, rows = [], []

    def chk(name, x, y, count=True):
        x, y = x.detach(), y.detach()
        if x.shape != y.shape:
            bad.append(name); print('  SHAPE', name, tuple(x.shape), tuple(y.shape), flush=True); return False
        d = (x.double() - y.double()).abs().max().item() / max(y.double().abs().max().item(), 1e-30)
        rows.append((name, d))
        if d > a.tol and count:
            bad.append(name)
        if d > 0:
            print(f'  {"DIFF" if count else "diff (reported)"} {name}: max |d| / peak {d:.3e}', flush=True)
        return d == 0

    g = torch.Generator().manual_seed(11)
    pos = torch.cat([3 * torch.rand(200, 2, generator=g) - 1.5, 1.1 * torch.rand(200, 1, generator=g) - 0.55], 1).to(dev)
    Phi = (2 * math.pi * torch.rand(200, generator=g) - math.pi).to(dev)
    S = states(a.work, a.plain)
    seen = {}
    for name, (st, zl) in S.items():
        key = json.dumps(st, sort_keys=True)
        same = seen.get(key)
        print(f'{name}: zlut {zl}, asym {bool(st["meta"].get("asym"))}, keys {sorted(st)}' + (f' (same state as {same})' if same else ''), flush=True)
        if same and zl == (-0.6, 0.6): continue
        seen.setdefault(key, name)
        asym = bool(st['meta'].get('asym'))
        reg = st.get('reg') is not None
        # ---- the model (and its gradients); in-situ states: the model of the state without its table keys
        sm = {k: v for k, v in st.items() if k not in ('reg', 'beta', 'spline', 'spline_base')}
        if not same:
            for zm in (('emitter', 'stage') if asym else (None,)):
                kw = {'zmode': zm} if zm else {}
                tg = f'{name}{"|" + zm if zm else ""}'
                q = C.psf_from_state(sm, device=dev, **kw); p = fourpi_psf(sm, device=dev, **kw)
                n = 48; ps, ph = pos[:n].clone().requires_grad_(), Phi[:n].clone().requires_grad_()
                inten = torch.linspace(500, 3000, n, device=dev).requires_grad_()
                bg = torch.linspace(0.5, 3, 4 * n, device=dev).reshape(n, 4).requires_grad_()
                w = torch.rand(n, 4, q.R, q.R, generator=g).to(dev)
                outs = {}
                for t, m in (('old', q), ('new', p)):
                    for v in [ps, ph, inten, bg] + list(m.parameters()): v.grad = None
                    I = m(ps, phi_extra=ph, intensity=inten, bg=bg)
                    (I * w).sum().backward()
                    outs[t] = dict(I=I.detach(), pos=ps.grad.clone(), phi_extra=ph.grad.clone(), intensity=inten.grad.clone(), bg=bg.grad.clone(),
                                   **{k: v.grad.clone() for k, v in m.named_parameters() if v.grad is not None})
                with torch.no_grad():
                    chk(f'{tg}|forward', p(pos, phi_extra=Phi), q(pos, phi_extra=Phi))
                ko, kn = set(outs['old']), set(outs['new'])
                if ko != kn:
                    bad.append(f'{tg}|grad keys'); print('  grad keys differ', sorted(ko ^ kn), flush=True)
                for k in sorted(ko & kn):
                    chk(f'{tg}|' + ('forward(intensity, bg)' if k == 'I' else f'grad {k}'), outs['new'][k], outs['old'][k])
                print(f'  {tg}: model and gradients of {sorted(ko & kn - {"I"})} compared', flush=True)
        # ---- the spline table
        grids = [dict(P=15, sub=4, zmin=zl[0], zmax=zl[1])]
        if not a.quick and not st.get('spline'):
            grids += [dict(P=15, sub=1, zmin=zl[0], zmax=zl[1]), dict(P=17, sub=2, zmin=-0.6, zmax=0.6)]
        for gr in grids:
            tag = f'{name}|P{gr["P"]}s{gr["sub"]}'
            L = T.LUT4Pi(st, dev=dev, **gr)
            N = fourpi_spline(st, gr['P'], (gr['zmin'], gr['zmax']), oversample=gr['sub'], device=dev)
            K0 = torch.stack([torch.stack([L.luts[k][c].knots for c in range(4)]) for k in range(3)])
            cnt = not (reg and not st.get('spline'))
            chk(f'{tag}|knots', N.knots, K0, count=cnt)
            pp = pos.clone(); pp[:, 2] = pp[:, 2].clamp(L.zmin, L.zmax)
            with torch.no_grad():
                for i, (x, y) in enumerate(zip(N.evaluate(pp, Phi, grad=True), L.evaluate(pp, Phi, grad=True))):
                    chk(f'{tag}|evaluate{["", "_dx", "_dy", "_dz", "_dphi"][i]}', x, y, count=cnt)
                for c in range(4):
                    chk(f'{tag}|fourpi_channel {c}', fourpi_channel(N, pp, Phi, c), L.evaluate_ch(pp, Phi, c), count=cnt)
                    for i, (x, y) in enumerate(zip(fourpi_channel(N, pp, Phi, c, grad=True), L.evaluate_ch_grad(pp, Phi, c))):
                        chk(f'{tag}|fourpi_channel {c} grad{i}', x, y, count=cnt)
            print(f'  {tag}: knots {tuple(N.knots.shape)}', flush=True)
        if st.get('spline'):          # a perturbed member of the refined table (the simulator's PSF bank)
            s2 = json.loads(json.dumps(st)); s2['gamma'] *= 0.9; s2['blur_px2'] += 0.03
            L = T.LUT4Pi(s2, P=15, dev=dev); N = fourpi_spline(s2, 15, device=dev)
            K0 = torch.stack([torch.stack([L.luts[k][c].knots for c in range(4)]) for k in range(3)])
            chk(f'{name}|perturbed|knots', N.knots, K0, count=not reg)
            print(f'  {name}|perturbed: knots {tuple(N.knots.shape)}', flush=True)
    worst = max(rows, key=lambda r: r[1]) if rows else ('', 0.0)
    print(f'{len(rows)} comparisons, {sum(d == 0 for _, d in rows)} bit-identical; largest |d| / peak {worst[1]:.3e} ({worst[0]})', flush=True)
    print('ALL IDENTICAL' if not bad and worst[1] == 0 else (f'{len(bad)} DIFFERENCES: {bad}' if bad else 'OK (differences only where reported)'), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
