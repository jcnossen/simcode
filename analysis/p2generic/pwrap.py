"""Run a simcode module (gate, dump, trainer, ...) with the AcquisitionSim presets patched in memory from a json (no package file is
edited by this wrapper): env P2G_PRESET = {"common": {...}, "0.5nM": {...}, "2.5nM": {...}} (keys as acquisition.PRESETS, incl. the
opt-in keys of 2026-11-13 generic p2). Process title env P2G_TITLE (default 'p2gen').
    python analysis/p2generic/pwrap.py simcode.benchmark.gate --scen iso ... (the module's own arguments)
Without P2G_PRESET the module runs unchanged."""
import json, math, os, runpy, sys
sys.path.insert(0, '/home/jc/simcode2')
import simcode.simulate.acquisition as RS
from simcode.util.proctitle import set_title
f = os.environ.get('P2G_PRESET', '')
if f:
    NR = json.load(open(f))
    for name, p in RS.PRESETS.items():
        p.update({k: (tuple(v) if isinstance(v, list) and k not in ('k', 'step_offs') else v)
                  for k, v in {**NR.get('common', {}), **NR.get(name, {})}.items()})
import torch
_pattern0 = RS.AcquisitionSim.pattern


def _pattern(self, nb, exact_est=False):
    """key step_offs = [[0, a, b], [0, c, d]] rad: fixed per-angle phases of the 3 steps relative to step 0 (e.g. a movie's measured,
    unequal spacings). Implemented without editing the package: the original pattern() runs with spacing (0, 0) (same random
    draws, so the same emitters / everything else), then the offsets are added to the true and the estimated phases."""
    so = self.p.get('step_offs')
    if so is None:
        P = _pattern0(self, nb, exact_est)
    else:
        sp0 = self.p['spacing']; self.p['spacing'] = (0.0, 0.0)
        try:
            P = _pattern0(self, nb, exact_est)
        finally:
            self.p['spacing'] = sp0
        off = torch.tensor(so, device=P['patt'].device, dtype=P['patt'].dtype)[P['steps'] // 3, P['steps'] % 3]   # [nb,T]
        for k in ('patt', 'patt_est'): P[k][..., 2] = P[k][..., 2] + off
    if self.p.get('patt_spec'): P = _spec(self.p['patt_spec'], P)
    return P


def _spec(sp, P):
    """key patt_spec = {"slots": [[kx, ky, phase offset, depth, rel], ...] (an L-frame acquisition cycle), "random_phase": bool}:
    replaces the true pattern after the original draw (emitters unchanged). Per FOV the window starts at a random slot of the cycle
    (U{0..L-1}); per FOV and orientation a random global phase U(0, 2 pi) + the slot's offset (random_phase: every frame U(0, 2 pi)).
    No within-window drift. The estimate keeps the original draw's estimate errors (patt_est - patt added; zero for exact scenes);
    shift_est recomputed for the new k (p2.dump.gauge_shift). steps = angle * 3 + slot % 3 (angle by |kx| > |ky|), so every
    steps // 3 user in the acquisition simulator / CRLB sees the right orientation."""
    from simcode.processing.gate_dump import gauge_shift
    pt = P['patt']; dv = pt.device; nb, T, _ = pt.shape
    Sl = torch.tensor(sp['slots'], device=dv, dtype=pt.dtype); L = len(Sl)
    slot = (torch.randint(0, L, (nb, 1), device=dv) + torch.arange(T, device=dv)[None]) % L
    q = Sl[slot]                                                                                    # [nb,T,5]
    ang = (q[..., 0].abs() < q[..., 1].abs()).long()
    ph = (torch.rand(nb, 2, device=dv) * 2 * math.pi).gather(1, ang) + q[..., 2]
    if sp.get('random_phase'): ph = torch.rand(nb, T, device=dv) * 2 * math.pi
    new = torch.stack([q[..., 0], q[..., 1], ph, q[..., 3], q[..., 4]], -1)
    pe = new + (P['patt_est'] - pt); pe[..., 3] = pe[..., 3].clamp(0.05, 1.0)
    P = dict(P, patt=new, patt_est=pe, steps=ang * 3 + slot % 3, angle=ang)
    P['shift_est'] = gauge_shift(pe, new) if not torch.equal(pe, new) else torch.zeros(nb, 2, device=dv)
    return P


RS.AcquisitionSim.pattern = _pattern
_title = os.environ.get('P2G_TITLE', 'p2gen'); set_title(_title)
import simcode.util.proctitle as _PT
_PT.auto = lambda name: set_title(_title)      # keep the p2gen title when the module sets its own (trainers call auto())
mod = sys.argv[1]; sys.argv = [mod] + sys.argv[2:]
if mod.endswith('.py'): runpy.run_path(mod, run_name='__main__')
else: runpy.run_module(mod, run_name='__main__', alter_sys=True)
