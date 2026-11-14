"""Run a simcode entry point under the process title 'band-ab <tag>', with the pass-1 head's position-prior range
optionally widened (2026-11-13 A/B; no package edit): env DETBAND_HRANGE = R replaces dxy = 0.5 tanh(q) in
SCHead4.forward by R tanh(q 0.5 / R) (same slope at 0 -> warm start from a 0.5-range checkpoint; R = 0.5 is identical
to the package code). The wrapper also stamps 'hrange' into nothing else: evaluation must use the same R as training.
usage: DETBAND_HRANGE=1.0 python analysis/detband/wrap.py <tag> <module> [args...]
  module 'sc4' = simcode.hd.sc4 train (args after it), 'gate' = simcode.hd.gate main, 'up2' = simcode.det.up2 train"""
import os, sys, inspect, textwrap
from simcode.util import proctitle
tag, mod = sys.argv[1], sys.argv[2]
proctitle.set_title('band-ab', tag)
proctitle.auto = lambda name: name
R = float(os.environ.get('DETBAND_HRANGE', '0.5'))
if R != 0.5:
    from simcode.hd import sc4
    src = textwrap.dedent(inspect.getsource(sc4.SCHead4.forward))
    old = 'dxy = 0.5 * torch.tanh(q[:, T + nt:T + nt + 2])'
    assert src.count(old) == 1
    src = src.replace(old, f'dxy = {R!r} * torch.tanh(q[:, T + nt:T + nt + 2] * {0.5 / R!r})')
    ns = {}; exec(compile(src, sc4.__file__, 'exec'), sc4.__dict__, ns)
    sc4.SCHead4.forward = ns['forward']
    print('DETBAND_HRANGE', R, flush=True)
if mod == 'sc4':
    from simcode.hd import sc4; sc4.train(sys.argv[3:])
elif mod == 'up2':
    from simcode.det import up2; up2.train(sys.argv[3:])
elif mod == 'gate':
    # also store the candidate pixel (ix, iy) of every row (sc_ix / sc_iy: the head patch centre; x_det - ix = the head's
    # 'off' input, outside [-0.5, 0.5] when the detector's winning pixel is not the nearest pixel)
    import torch
    from simcode.det import model as DM
    from simcode.hd import gate
    ST_ = {}
    _sel = DM.select_subpixel
    def _sel2(*a_, **k_):
        c = _sel(*a_, **k_); ST_['c'] = c; return c
    DM.select_subpixel = _sel2
    _out = gate.out_sc
    def _out2(SC, o, zmax):
        r = _out(SC, o, zmax)
        if r is not None:
            c = ST_['c']; keep = torch.isfinite(c['logit'])
            r['ix'] = c['ix'][keep].float(); r['iy'] = c['iy'][keep].float()
        return r
    gate.out_sc = _out2
    sys.argv = ['gate'] + sys.argv[3:]; gate.main()
