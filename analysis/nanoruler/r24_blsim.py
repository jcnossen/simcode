"""R24: the re-implemented SMLM / SIMFLUX baseline (bl_run.py) on a SIMULATED nanoruler movie, and its tables with the modulation error.
bl_run.py is not edited: its source is read, patched in memory and executed (the method of r24b_run.py):
  "set_title('nr-base')" -> 'nr-r24'; "'nr80'" -> "'<sim name>'" (3 x, label only); the frame-count / shape assertion
  "n_tf == n_rd == 30000 and len(files) == 1 and tuple(shape) == (400, 400)" -> "n_tf == n_rd and len(files) == 1";
  globals MOVIE = <sim>/sim_movie.tif, CAM = <sim>/camera_maps_sim.npz, CALIB = <sim>/calib_sim.json, OUTROOT = out/nanoruler/<mv>/big/blsim.
Then (CPU, as bl_tables.py main() up to the tables; no published tables exist for a simulation): cycle-aligned rows only, frame = cycle,
  <OUTROOT>/<name>_smlm.npz   frame, x_nm, y_nm, sx_nm, sy_nm (CRLB), photons, bg, chisq, moderr (modulation error of the SMLM row against the
                              SIMFLUX pattern, smlmtorch ModulationPattern.mod_error), in_sf (row passed the run's filter 0.1 and the SIMFLUX fit)
  <OUTROOT>/<name>_simflux.npz  the SIMFLUX table of the run (modulation-error filter 0.1, as bl_run) + moderr of its SMLM row
The published processing used a modulation-error filter of 0.02 (20 nm movie) / 0.012 (80 nm): rows with moderr < 0.02 of these tables.
usage: bash analysis/nanoruler/r24_gpu.sh 5 python analysis/nanoruler/r24_blsim.py <nr40|nr20> <sim name> [--det_sigma 1.83] [bl_run options]"""
import json, os, sys
import numpy as np
REPO = '/home/jc/simcode2'; SRC = f'{REPO}/analysis/nanoruler/bl_run.py'
mv, name = sys.argv[1], sys.argv[2]; argv = [SRC] + sys.argv[3:]
if not any(x.startswith('--det_sigma') for x in argv): argv += ['--det_sigma', dict(nr40='1.82', nr20='1.83')[mv]]
if '--tag' not in argv: argv += ['--tag', name]
sd = f'{REPO}/out/nanoruler/{mv}/sim/{name}'; outroot = f'{REPO}/out/nanoruler/{mv}/big/blsim'; os.makedirs(outroot, exist_ok=True)
d = f'{outroot}/{name}/'
if not os.path.exists(d + 'sf/sf.hdf5'):
    src = open(SRC).read()
    for old, new, n in (("set_title('nr-base')", "set_title('nr-r24')", 1), ("'nr80'", f"'{name}'", 3),
                        ('n_tf == n_rd == 30000 and len(files) == 1 and tuple(shape) == (400, 400)', 'n_tf == n_rd and len(files) == 1', 1)):
        assert src.count(old) == n, (old, src.count(old)); src = src.replace(old, new)
    ns = dict(__name__='bl_run_patched', __file__=SRC); exec(compile(src, SRC, 'exec'), ns)
    ns.update(MOVIE=f'{sd}/sim_movie.tif', CAM=f'{sd}/camera_maps_sim.npz', CALIB=f'{sd}/calib_sim.json', OUTROOT=outroot)
    sys.argv = argv; ns['main']()
import h5py
from scipy.spatial import cKDTree
from smlmtorch.simflux.dataset import SFDataset
from smlmtorch.simflux.pattern_estimator import ModulationPattern
PX = 65.0
ds = SFDataset.load(d + 'smlm/smlm.npy'); f = h5py.File(d + 'sf/sf.hdf5', 'r')['locs'][:]; mp_sf = ModulationPattern.load(d + 'smlm/mp_sf.pickle'); me = np.asarray(mp_sf.mod_error(ds))
g = ds.frame % 6 == 0; S = dict(frame=(ds.frame[g] // 6).astype(np.int64), x=ds.pos[g, 0].astype(float), y=ds.pos[g, 1].astype(float))
gf = f['frame'] % 6 == 0; f = f[gf]; F = dict(frame=(f['frame'] // 6).astype(np.int64), x=f['x'].astype(float), y=f['y'].astype(float))
T = cKDTree(np.c_[S['x'], S['y'], S['frame'] * 1e3]); dd, j = T.query(np.c_[F['x'], F['y'], F['frame'] * 1e3], distance_upper_bound=0.5); ok = np.isfinite(dd)
in_sf = np.zeros(len(S['x']), bool); in_sf[j[ok]] = True; mef = np.full(len(F['x']), np.nan); mef[ok] = me[g][j[ok]]
np.savez_compressed(f'{outroot}/{name}_smlm.npz', frame=S['frame'], x_nm=S['x'] * PX, y_nm=S['y'] * PX, sx_nm=ds.crlb.pos[g, 0] * PX, sy_nm=ds.crlb.pos[g, 1] * PX,
                    photons=ds.photons[g], bg=ds.background[g], chisq=ds.chisq[g], moderr=me[g], in_sf=in_sf)
np.savez_compressed(f'{outroot}/{name}_simflux.npz', frame=F['frame'], x_nm=F['x'] * PX, y_nm=F['y'] * PX, sx_nm=f['lpx'] * PX, sy_nm=f['lpy'] * PX, photons=f['photons'], bg=f['bg'] * 6, moderr=mef)
print(f'{name}: SMLM rows {len(S["x"])}, SIMFLUX rows {len(F["x"])} ({int(ok.sum())} linked to their SMLM row); moderr percentiles 5/25/50/75/95 {np.round(np.percentile(me[g], [5, 25, 50, 75, 95]), 4).tolist()}; '
      f'fraction of SMLM rows with moderr < 0.012 / 0.02 / 0.05 / 0.1: {[round(float(np.mean(me[g] < v)), 4) for v in (0.012, 0.02, 0.05, 0.1)]}')
