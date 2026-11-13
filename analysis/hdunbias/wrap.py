"""Run a simcode entry point under the process title 'hd-unbias <tag>' (2026-11-13 head-prior pixel-bias experiment).
usage: python analysis/hdunbias/wrap.py <tag> sc4 [simcode.hd.sc4 train args]
       python analysis/hdunbias/wrap.py <tag> gate [simcode.hd.gate args]   (also stores the candidate pixel sc_ix / sc_iy)"""
import sys
from simcode.util import proctitle
tag, mod = sys.argv[1], sys.argv[2]
proctitle.set_title('hd-unbias', tag)
proctitle.auto = lambda name: name
if mod == 'sc4':
    from simcode.hd import sc4; sc4.train(sys.argv[3:])
elif mod == 'gate':
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
