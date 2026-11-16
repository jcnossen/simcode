"""SIMCODE's own 4Pi PSF (simcode/fourpi/coherent_psf.py) and spline table (simcode/fourpi/spline_table.LUT4Pi) vs torch-psf's
FourPiPSF / FourPiSplinePSF (MultiChannelPSF subclasses) for the PSF states of the 4Pi paper heads: the pre-check of the
switch to torch-psf (out/COMPAT_AUDIT.md, plan A1). Bit-identity (torch.equal) of
  - FourPiPSF.from_state(state).forward(pos, phi_extra=Phi) vs coherent_psf.psf_from_state(state).forward
  - the knots of FourPiPSF.to_spline vs LUT4Pi for every grid SIMCODE builds: the heads (P 15, 4 x oversampled, z of the
    checkpoint's zlut or -0.6 .. 0.6), the prior template (P 15, sub 1), the scene simulator (P 17, sub 2)
  - FourPiSplinePSF.evaluate (value and the 4 derivatives), .components and luts[k][c].render(pos, grad) vs LUT4Pi
  - FourPiSplinePSF.render_channels (MultiChannelPSF interface, phi as the extra column) vs LUT4Pi.evaluate
States: <4Pi work dir>/psf4pi/NPC_f_dxy15.json and ck['psf'] of head4pi_{v10, v3L, v11}.pt. LUT4Pi.tau (the stored tau)
and FourPiSplinePSF.tau (softmax of the source's tau_logit) are compared too but only reported (no SIMCODE code reads
lut.tau).

  python tests/check_fourpi_spline.py [--device cpu|cuda] [--quick]     (exit code 1 on any difference)
"""
import argparse, json, math, os, sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def states(P4):
    out = {'NPC_f_dxy15': json.load(open(os.path.join(P4, 'psf4pi', 'NPC_f_dxy15.json')))['psf']}
    for tag in ('v10', 'v3L', 'v11'):
        f = os.path.join(P4, 'models', f'head4pi_{tag}.pt')
        if os.path.isfile(f):
            ck = torch.load(f, weights_only=False, map_location='cpu')
            out[f'head4pi_{tag}'] = (ck['psf'], tuple(ck['args'].get('zlut') or (-0.6, 0.6)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--quick', action='store_true', help='head grid only (no sub 1 / P 17 grids)')
    a = ap.parse_args()
    dev = a.device
    import torch_psf
    from simcode import config
    from simcode.psf import FourPiPSF
    from simcode.fourpi.coherent_psf import psf_from_state
    from simcode.fourpi.spline_table import LUT4Pi
    print('torch_psf', torch_psf.__file__, '| device', dev, flush=True)
    bad = []

    def chk(name, x, y):
        ok = torch.equal(x, y)
        if not ok:
            bad.append(name)
            d = (x.double() - y.double()).abs().max().item() if x.shape == y.shape else float('nan')
            print('  DIFF', name, tuple(x.shape), tuple(y.shape), 'max abs', d, flush=True)
        return ok

    g = torch.Generator().manual_seed(11)
    pos = torch.cat([3 * torch.rand(200, 2, generator=g) - 1.5, 1.1 * torch.rand(200, 1, generator=g) - 0.55], 1).to(dev)
    Phi = (2 * math.pi * torch.rand(200, generator=g) - math.pi).to(dev)
    S = states(config.root('fourpi'))
    seen = {}
    for name, v in S.items():
        st, zl = (v, (-0.6, 0.6)) if isinstance(v, dict) else v
        key = json.dumps(st, sort_keys=True)
        same = seen.get(key)
        print(f'{name}: zlut {zl}, asym {bool(st["meta"].get("asym"))}, keys {sorted(st)}'
              + (f' (same state as {same})' if same else ''), flush=True)
        if same and zl == (-0.6, 0.6): continue
        seen.setdefault(key, name)
        kw = {'zmode': 'emitter'} if st['meta'].get('asym') else {}
        q = psf_from_state(st, device=dev, **kw).eval(); p = FourPiPSF.from_state(st, device=dev, **kw).eval()
        with torch.no_grad():
            chk(f'{name}|forward', p(pos[:64], phi_extra=Phi[:64]), q(pos[:64], phi_extra=Phi[:64]))
        grids = [dict(P=15, sub=4, zmin=zl[0], zmax=zl[1])]
        if not a.quick:
            grids += [dict(P=15, sub=1, zmin=zl[0], zmax=zl[1]), dict(P=17, sub=2, zmin=-0.6, zmax=0.6)]
        for gr in grids:
            tag = f'{name}|P{gr["P"]}s{gr["sub"]}'
            L = LUT4Pi(st, dev=dev, **gr)
            with torch.no_grad():
                T = p.to_spline(gr['P'], (gr['zmin'], gr['zmax']), oversample=gr['sub'], dz_um=0.01, zpad=3)
            K0 = torch.stack([torch.stack([L.luts[k][c].knots for c in range(4)]) for k in range(3)])
            if not chk(f'{tag}|knots', T.knots, K0): continue
            pp = pos.clone(); pp[:, 2] = pp[:, 2].clamp(gr['zmin'], gr['zmax'])
            with torch.no_grad():
                chk(f'{tag}|evaluate', T.evaluate(pp, Phi), L.evaluate(pp, Phi))
                for i, (x, y) in enumerate(zip(T.evaluate(pp, Phi, grad=True), L.evaluate(pp, Phi, grad=True))):
                    chk(f'{tag}|evaluate_grad{i}', x, y)
                chk(f'{tag}|components', T.components(pp), L.components(pp))
                for k in range(3):
                    for c in range(4):
                        for i, (x, y) in enumerate(zip(T.luts[k][c].render(pp, True), L.luts[k][c].render(pp, True))):
                            chk(f'{tag}|luts[{k}][{c}].render{i}', x, y)
                pc = pp[:, None].expand(-1, 4, -1).contiguous()
                chk(f'{tag}|render_channels', T.render_channels(pc, Phi[:, None]), L.evaluate(pp, Phi))
            print(f'  {tag}: knots {tuple(T.knots.shape)} identical; tau equal {torch.equal(T.tau, L.tau)} '
                  f'(max diff {(T.tau - L.tau).abs().max().item():.2e})', flush=True)
    print('ALL IDENTICAL' if not bad else f'{len(bad)} DIFFERENCES: {bad}', flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
