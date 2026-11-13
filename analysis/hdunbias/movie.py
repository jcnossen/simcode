"""papersim movie through SIMCODE v2 (analysis/papersim/run_simcode.py, m1809 set, sim calibration) with only the pass-1
head swapped (pass-1 solve calibration of hd_m1809 kept): python analysis/hdunbias/movie.py <D> <tag> <head path>
-> out/hdunbias/movie_<D>_<tag>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'papersim'))
import run_simcode as RS
D, tag, head = sys.argv[1:4]
_ns = RS.netset
RS.netset = lambda name: dict(_ns(name), head=os.path.abspath(head))
from simcode.util import proctitle
proctitle.set_title('hd-unbias', 'mov ' + tag); proctitle.auto = lambda n: n
sys.argv = ['run_simcode', os.path.join(REPO, 'out', 'papersim', D), '--netset', 'm1809', '--cal_kind', 'sim', '--gt_diag', '0', '--out', os.path.join(REPO, 'out', 'hdunbias', f'movie_{D}_{tag}')]
RS.main()
