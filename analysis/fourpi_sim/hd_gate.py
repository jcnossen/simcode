"""hd head experiment: the isolated-emitter CRLB gate (set A) and the 1-3 neighbour set (set B) of analysis/fourpi/gatek4pi.py,
run unchanged on a checkpoint path; its json (normally $P4/sim/gatek_<head>.json) is redirected to --out.
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi_sim/hd_gate.py --head $P4/models/head4pi_hd1/head4pi_hd1.pt --out $P4/models/head4pi_hd1/gate
"""
import argparse, builtins, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f'{REPO}/analysis/fourpi')
if REPO not in sys.path: sys.path.insert(0, REPO)
p = argparse.ArgumentParser(); p.add_argument('--head', required=True); p.add_argument('--out', required=True)
a, rest = p.parse_known_args()
os.makedirs(a.out, exist_ok=True)
import gatek4pi as g
_open = builtins.open
g.open = lambda f, m='r', *x, **k: _open(os.path.join(a.out, os.path.basename(f)) if ('w' in m and f.startswith(f'{g.P4}/sim/')) else f, m, *x, **k)
sys.argv = [sys.argv[0], '--head', a.head] + rest
g.main()
