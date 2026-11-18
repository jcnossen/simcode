"""
Per-pixel sCMOS maps from a sparse movie (no dark frames available): the command line of the pipeline's cammaps stage over
torch_psf maps_from_movie (via simcode.psf):
 - offset deviation: per-pixel temporal median away from the localizations minus its spatially smoothed version (the
   smooth part is background, the per-pixel residual is offset heterogeneity); absolute offset = global offset + deviation.
 - read noise: robust variance (MAD) of differences between frames 6 apart (same pattern step, removes static
   structure; MAD rejects blinking), minus the shot-noise contribution of the local mean level.
Outputs --out (default simcode_out/insitu/camera_maps_0.5nM.npz) (copy to models/insitu/ and reference it as camera.maps in the calib json) with offset_adu [H,W], readnoise_e [H,W], plus diagnostics.
"""
import argparse, json, os
import numpy as np
from simcode import config
from simcode.psf import maps_from_movie

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--movie', required=True, help='the movie (e.g. the 0.5 nM movie, simcode/calibration/__init__.py)')
    ap.add_argument('--nframes', type=int, default=20000)
    ap.add_argument('--gain', type=float, default=0.56)
    ap.add_argument('--offset', type=float, default=99.3)
    ap.add_argument('--locs', nargs='+', required=True,
                    help='Picasso hdf5 (frame, x, y; masked frames f-1..f+6) and/or localize_real .npz (x_np, y_np, wstart; masked wstart-1..wstart+T)')
    ap.add_argument('--npz_T', type=int, default=12)
    ap.add_argument('--readnoise_from', default='', help='take the read-noise map from this npz (camera property) instead of re-estimating')
    ap.add_argument('--readnoise_shift', default='auto', help="sensor-ROI shift dy,dx of the reference map: this[y,x] = ref[y+dy, x+dx]; "
                    "'auto' = cross-correlation of the high-pass log temporal-variance maps (the movies were recorded at different "
                    "sensor ROI positions: 2.5/20 nM vs 0.5 nM = 22,2). Uncovered pixels: high-passed own estimate.")
    ap.add_argument('--mask_r', type=float, default=3.0)
    ap.add_argument('--out', default='simcode_out/insitu/camera_maps_0.5nM.npz')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.readnoise_from: a.readnoise_from = config.resolve(a.readnoise_from)
    # mask of the localizations: frames f-1 .. f+6 of each 6-frame-window loc (hdf5) / wstart-1 .. wstart+T (npz)
    import h5py
    _lx, _ly, _lf, _lw = [], [], [], []
    for _p in a.locs:
        if _p.endswith('.npz'):
            _L = np.load(_p); _lx.append(_L['x_np']); _ly.append(_L['y_np']); _lf.append(_L['wstart']); _lw.append(np.full(len(_L['x_np']), a.npz_T))
        else:
            _L = h5py.File(_p)['locs'][:]; _lx.append(_L['x']); _ly.append(_L['y']); _lf.append(_L['frame']); _lw.append(np.full(len(_L), 6))
    spots = dict(x=np.concatenate(_lx), y=np.concatenate(_ly), frame=np.concatenate(_lf), win=np.concatenate(_lw))
    ref = dict(np.load(a.readnoise_from)) if a.readnoise_from else None
    if ref is not None: print(f'read-noise reference map: {a.readnoise_from}')
    shift = a.readnoise_shift if a.readnoise_shift == 'auto' else tuple(int(v) for v in a.readnoise_shift.split(','))
    cam, stats = maps_from_movie(a.movie, spots, a.gain, a.offset, nframes=a.nframes, lag=6, mask_radius=a.mask_r,
                                 readnoise_ref=ref, readnoise_shift=shift, log=print)
    stats['readnoise_shift'] = tuple(stats['readnoise_shift'])
    X = cam.map_extra
    np.savez_compressed(a.out, offset_adu=cam.offset_adu, readnoise_e=cam.readnoise_e, low5=X['low5'], var_adu=X['var_adu'],
                        gain=a.gain, readnoise_shift=X['readnoise_shift'], readnoise_covered=X['readnoise_covered'])
    print(json.dumps({k: v for k, v in stats.items() if k != 'corr_offset_density' and k != 'shift_peak_corr'}, indent=1))
