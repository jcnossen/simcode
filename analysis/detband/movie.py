"""papersim movie through SIMCODE v2 (analysis/papersim/run_simcode.py, m1809 set, sim calibration) with only the pass-1
detector swapped: python analysis/detband/movie.py <D> <tag> <det path>  -> out/detband/movie_<D>_<tag>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'papersim'))
import run_simcode as RS
D, tag, det = sys.argv[1:4]
_ns = RS.netset
RS.netset = lambda name: dict(_ns(name), det=os.path.abspath(det))
from simcode.util import proctitle
proctitle.set_title('band-ab', 'mov ' + tag); proctitle.auto = lambda n: n
sys.argv = ['run_simcode', os.path.join(REPO, 'out', 'papersim', D), '--netset', 'm1809', '--cal_kind', 'sim', '--gt_diag', '0', '--out', os.path.join(REPO, 'out', 'detband', f'movie_{D}_{tag}')]
RS.main()
