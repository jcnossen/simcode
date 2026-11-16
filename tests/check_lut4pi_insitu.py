"""SIMCODE's LUT4Pi from an in-situ state (arms / reg / beta) and from a refined spline file vs the torch-psf models of
analysis/fourpi_plain/insitu4pi.py (two processes: torch-psf master for A, the pinned torch_psf of SIMCODE for B).

  python tests/check_lut4pi_insitu.py A <Model4Pi state json> <out.npz> [<refined knots .pt>]
  python tests/check_lut4pi_insitu.py B <out.npz>
"""
import json, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mode = sys.argv[1]
g = torch.Generator().manual_seed(0)
pos = torch.cat([torch.rand(300, 2, generator=g) * 2 - 1, torch.rand(300, 1, generator=g) * 1.0 - 0.9], 1)
Phi = torch.rand(300, generator=g) * 6.28
if mode == 'A':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis', 'fourpi_plain'))
    from insitu4pi import Model4Pi, simcode_state, baked_spline, SplineModel
    m = Model4Pi.from_state(json.load(open(sys.argv[2])), roisize=15)
    dev = 'cuda'; p, f = pos.to(dev), Phi.to(dev)
    xy = p[:, None, :2].expand(-1, 4, -1)
    with torch.no_grad():
        vec = m.images(xy, p[:, 2], f)
    out = dict(vec=vec.cpu().numpy(), state=json.dumps(simcode_state(m)))
    if len(sys.argv) > 4:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis', 'fourpi_plain'))
        from p4spline_refine import load_refined
        S = load_refined(sys.argv[4]); S.spl = S.spl.resized(15)
        with torch.no_grad():
            out['ref'] = S.images(xy, p[:, 2], f).cpu().numpy()
        out['ref_file'] = sys.argv[4]
    np.savez(sys.argv[3], **out)
else:
    from simcode.fourpi.lut4pi import LUT4Pi
    d = np.load(sys.argv[2])
    st = json.loads(str(d['state']))
    L = LUT4Pi(st, P=15, zmin=-1.0, zmax=1.0, dev='cuda')
    with torch.no_grad():
        v = L.evaluate(pos.cuda(), Phi.cuda()).cpu().numpy()
    print('LUT4Pi(in-situ state) vs torch-psf vectorial: rel L2', np.linalg.norm(v - d['vec']) / np.linalg.norm(d['vec']))
    if 'ref' in d.files:
        st2 = dict(st, spline=str(d['ref_file']))
        L2 = LUT4Pi(st2, P=15, dev='cuda')
        with torch.no_grad():
            v2 = L2.evaluate(pos.cuda(), Phi.cuda()).cpu().numpy()
        print('LUT4Pi(spline file) vs torch-psf refined spline: rel L2', np.linalg.norm(v2 - d['ref']) / np.linalg.norm(d['ref']),
              '; vs vectorial', np.linalg.norm(v2 - d['vec']) / np.linalg.norm(d['vec']))
        st3 = json.loads(json.dumps(st)); st3['gamma'] = st3['gamma'] * 0.9; st3['spline'] = str(d['ref_file']); st3['spline_base'] = st
        L3 = LUT4Pi(st3, P=15, dev='cuda')
        with torch.no_grad():
            v3 = L3.evaluate(pos.cuda(), Phi.cuda()).cpu().numpy()
        print('perturbed (gamma x 0.9) spline member differs from the refined table by rel', np.linalg.norm(v3 - v2) / np.linalg.norm(v2))
