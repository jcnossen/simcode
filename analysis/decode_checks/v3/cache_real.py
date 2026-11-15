"""One-time cache of a real-data chunk for the DECODE progress evaluation (eval_watch.py --real): 1000 non-overlapping 6-frame
sums (frames 60 000-65 999) of each paper movie, in ADU sums, + the camera-v2 map (gain, 6 x offset) and the SIMCODE crop.
-> out/decode_checks/v3/cache/real_<ds>.npz   (read once from /mnt/data; the evaluation never touches the raw movies)
usage: python analysis/decode_checks/v3/cache_real.py [1809_2 2.5nM_2 20nM_2]"""
import os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(os.path.expanduser('~/simcode2'), 'out', 'decode_checks', 'v3', 'cache')
D = '/mnt/data/dmdsf/bn_dnapaint'
DS = {'1809_2': ('1809/cover1/2/simflux_5.5ms_500mW.tif', 'camera_maps_1809_2_v2.npz'),
      '2.5nM_2': ('dnapaint/2.5nM/2/cover1_600mW_5.5ms_cf660.tif', 'camera_maps_2p5nM_m2_v2.npz'),
      '20nM_2': ('dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif', 'camera_maps_20nM_m2_v2.npz')}
F0, NS = 60000, 1000
os.makedirs(OUT, exist_ok=True)
from smlmtorch.util.multipart_tiff import tiff_read_file
for ds in (sys.argv[1:] or DS):
    tif, cam = DS[ds]
    fr = np.asarray(list(tiff_read_file(f'{D}/{tif}', startframe=F0, maxframes=6 * NS, use_progbar=False)), np.float32)
    sums = fr.reshape(NS, 6, *fr.shape[1:]).sum(1)
    c = np.load(os.path.join(REPO, 'models', 'insitu', cam))
    np.savez(os.path.join(OUT, f'real_{ds}.npz'), sums=sums, gain=float(c['gain']), offset6=c['offset_adu'].astype(np.float32) * 6,
             frame0=F0, crop=np.array([5, 5, 110]), camera=cam)
    print(ds, sums.shape, float(np.median(sums)), flush=True)
