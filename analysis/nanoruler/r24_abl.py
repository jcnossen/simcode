"""R24: run an nr80 ablation script (abl_autoc / abl_gate / abl_solve / abl_capture / abl_pattern) on the nr40 / nr20 data WITHOUT editing it:
abl_lib is imported first and its module constants are replaced in memory, then the script runs as __main__ with the given arguments.
  abl_lib.OUT    -> out/nanoruler/<mv>/abl          (autoc_*.json, calibs/, runs/ -> big/abl_runs, head/, patterns/)
  abl_lib.SETS   -> p1 / p2 = the real movie (run out/nanoruler/<mv>/runs/real), sima_p1|p2, simc0_p1|p2 = the simulated LINEAR-ruler movies s<d>_a / s<d>_c0,
                    simga_p1|p2, simgc0_p1|p2 = the simulated GRID movies s<d>g_a / s<d>g_c0
  abl_lib.MOVIES -> real / the two simulated movies of <mv>
Unchanged on purpose: the per-variant solve calibrations are those of the nr80 NETWORK SET, fitted on the nr80 gate scenes
(out/nanoruler/eval/fe_nr80); the existing fits out/nanoruler/ablation/calibs/calib_<front>_<variant>.json are copied into <OUT>/calibs so that
abl_gate.fit finds them (a variant whose calibration depends on this movie's theta, auto / autoS, is fitted here and written to <OUT>/calibs).
The process title is 'nr-r24'.
usage: python analysis/nanoruler/r24_abl.py <nr40|nr20> <autoc|gate|solve|capture|pattern> <arguments of that script>"""
import glob, os, runpy, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abl_lib as A
mv, mod = sys.argv[1], sys.argv[2]; D = dict(nr40=('s40', 6666), nr20=('s20', 16666))[mv]
NR = A.NR; O = f'{NR}/{mv}'; A.OUT = f'{O}/abl'; MC = f'{NR}/models/calib'
os.makedirs(f'{O}/big/abl_runs', exist_ok=True); os.makedirs(f'{A.OUT}/calibs', exist_ok=True)
if not os.path.exists(f'{A.OUT}/runs'): os.symlink(f'{O}/big/abl_runs', f'{A.OUT}/runs')
if not os.path.exists(f'{A.OUT}/head'): os.makedirs(f'{O}/big/abl_head', exist_ok=True); os.symlink(f'{O}/big/abl_head', f'{A.OUT}/head')
for f in glob.glob(f'{NR}/ablation/calibs/calib_s[12]_*.json'):
    b = os.path.basename(f)
    if '_auto' not in b and not os.path.exists(f'{A.OUT}/calibs/{b}'): shutil.copy(f, f'{A.OUT}/calibs/{b}')
A.SETS = {'p1': dict(kind='real', run=f'{O}/runs/real', front='s1', tag='est', ncyc=D[1], prod=f'{MC}/calib_sc_hd_nr80.json'),
          'p2': dict(kind='real', run=f'{O}/runs/real', front='s2', tag='est2', ncyc=D[1], prod=f'{MC}/calib_p2L_nr80.json', head=f'{A.OUT}/head/real')}
A.MOVIES = {'real': dict(movie=f'/mnt/data/dmdsf/simcode2_out/nanoruler/raw_{mv}/movie_reord.tif', calib=f'{NR}/calib_{mv}.json', crop='5,5,390', chunk=1500)}
for g_ in ('', 'g'):                                   # '' = linear 2 / 3-site rulers (s40_a ...), 'g' = grid structures (s40g_a ...): sets sim[g]a_p1 ...
    for v in ('a', 'c0'):
        s = f'{D[0]}{g_}_{v}'; sd = f'{O}/sim/{s}'
        if not os.path.exists(f'{sd}/sim_info.json'): continue
        import json; nc = json.load(open(f'{sd}/sim_info.json'))['args']['n_frames'] // 6
        A.SETS[f'sim{g_}{v}_p1'] = dict(kind='sim', run=f'{O}/runs/{s}', front='s1', tag='est', ncyc=nc, prod=f'{MC}/calib_sc_hd_nr80.json', sim=sd)
        A.SETS[f'sim{g_}{v}_p2'] = dict(kind='sim', run=f'{O}/runs/{s}', front='s2', tag='est2', ncyc=nc, prod=f'{MC}/calib_p2L_nr80.json', sim=sd, head=f'{A.OUT}/head/{s}')
        A.MOVIES[s] = dict(movie=f'{sd}/sim_movie.tif', calib=f'{sd}/calib_sim.json', crop='5,5,246', chunk=1500)
if mod == 'pattern': A.NR = O      # abl_pattern.PST (the nr80 per-step pattern file) must not be found: only the k<N> re-estimates apply here
_t = A.title; A.title = lambda t='nr-r24': _t('nr-r24')
sys.argv = [f'abl_{mod}.py'] + sys.argv[3:]
runpy.run_path(os.path.join(HERE, f'abl_{mod}.py'), run_name='__main__')
