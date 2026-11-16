"""
Per-pixel sCMOS maps from a sparse movie (no dark frames available):
 - offset deviation: per-pixel low percentile of ADU minus its spatially smoothed version (the smooth part is
   background, the per-pixel residual is offset heterogeneity); absolute offset = global offset + deviation.
 - read noise: robust variance (MAD) of differences between frames 6 apart (same pattern step, removes static
   structure; MAD rejects blinking), minus the shot-noise contribution of the local mean level.
Outputs --out (default simcode_out/insitu/camera_maps_0.5nM.npz) (copy to models/insitu/ and reference it as camera.maps in the calib json) with offset_adu [H,W], readnoise_e [H,W], plus diagnostics.
"""
import argparse, json, os
import numpy as np, tifffile
from scipy import ndimage as ndi
from simcode import config

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

    t = tifffile.TiffFile(a.movie)
    n = min(a.nframes, len(t.pages))
    mov = np.stack([t.pages[i].asarray() for i in range(n)]).astype(np.float32)   # [n,H,W]
    low = np.percentile(mov, 5, axis=0)
    # offset deviation from the per-pixel temporal MEDIAN minus its smooth version (critic review 12: the p5 sits ~1.645 sigma_px
    # below the offset, so noisy pixels got offsets that were too low -> static excess, x T in multi-frame sums)
    # exclude pixels near localizations (frames f-1 .. f+6 of each 6-frame-window loc) so filament signal does not enter the
    # offset (critic review 13): median over the remaining frames
    import h5py
    _lx, _ly, _lf, _lw = [], [], [], []
    for _p in a.locs:
        if _p.endswith('.npz'):
            _L = np.load(_p); _lx.append(_L['x_np']); _ly.append(_L['y_np']); _lf.append(_L['wstart']); _lw.append(np.full(len(_L['x_np']), a.npz_T))
        else:
            _L = h5py.File(_p)['locs'][:]; _lx.append(_L['x']); _ly.append(_L['y']); _lf.append(_L['frame']); _lw.append(np.full(len(_L), 6))
    Lc = dict(x=np.concatenate(_lx), y=np.concatenate(_ly), frame=np.concatenate(_lf), win=np.concatenate(_lw))
    _k = (Lc['frame'] < n) & np.isfinite(Lc['x']) & np.isfinite(Lc['y'])
    Lc = {k_: v_[_k] for k_, v_ in Lc.items()}
    print(f'{len(Lc["x"])} localizations used for the mask')
    H_, W_ = mov.shape[1:]
    busy = np.zeros(mov.shape, bool)
    yy_, xx_ = np.mgrid[-int(a.mask_r):int(a.mask_r) + 1, -int(a.mask_r):int(a.mask_r) + 1]
    disk = (yy_ ** 2 + xx_ ** 2) <= a.mask_r ** 2
    dy_, dx_ = yy_[disk], xx_[disk]
    cx_, cy_ = np.round(Lc['x']).astype(int), np.round(Lc['y']).astype(int)
    for df in range(-1, int(Lc['win'].max()) + 1):
        fr_ = Lc['frame'].astype(int) + df
        ok_ = (fr_ >= 0) & (fr_ < n) & (df <= Lc['win'])
        for oy, ox in zip(dy_, dx_):
            yy2, xx2 = cy_[ok_] + oy, cx_[ok_] + ox
            good = (yy2 >= 0) & (yy2 < H_) & (xx2 >= 0) & (xx2 < W_)
            busy[fr_[ok_][good], yy2[good], xx2[good]] = True
    print(f'masked fraction of pixel-frames: {busy.mean():.3f}')
    mv = mov.copy(); mv[busy] = np.nan
    med = np.nanmedian(mv, axis=0); del mv
    med = np.where(np.isfinite(med), med, np.median(mov, axis=0))
    smooth = ndi.gaussian_filter(ndi.median_filter(med, size=9), 2)
    offset_dev = med - smooth
    dens = np.zeros((H_, W_)); np.add.at(dens, (np.clip(cy_, 0, H_ - 1), np.clip(cx_, 0, W_ - 1)), 1)
    dens = ndi.gaussian_filter(dens, 1.0)
    print(f'corr(offset_dev, loc density) = {np.corrcoef(offset_dev.ravel(), dens.ravel())[0, 1]:.3f}  (target < 0.05)')
    d = mov[6:] - mov[:-6]
    mad = np.median(np.abs(d - np.median(d, 0)), 0) * 1.4826
    var_adu = mad ** 2 / 2
    level_e = np.clip((np.median(mov, 0) - (a.offset + offset_dev)) * a.gain, 0, None)
    rn2_e = var_adu * a.gain ** 2 - level_e
    rn_e = np.sqrt(np.clip(rn2_e, 0.9 ** 2, None))   # floor = global read noise from robust var-vs-level fit (robust trimmed var-vs-level fit of same-step frame differences, development script)
    shift_used = (0, 0)
    if a.readnoise_from:
        REF = np.load(a.readnoise_from)
        rn_ref = REF['readnoise_e'].astype(np.float64)
        if a.readnoise_shift == 'auto':
            hp_ = lambda v: np.clip(np.log(v) - ndi.median_filter(np.log(v), 5), -1, 3)
            A_, B_ = hp_(REF['var_adu']), hp_(var_adu)
            F_ = np.fft.ifft2(np.fft.fft2(A_ - A_.mean()) * np.conj(np.fft.fft2(B_ - B_.mean()))).real / (A_.std() * B_.std() * A_.size)
            iy_, ix_ = np.unravel_index(F_.argmax(), F_.shape)
            dy_s = iy_ if iy_ < F_.shape[0] // 2 else iy_ - F_.shape[0]; dx_s = ix_ if ix_ < F_.shape[1] // 2 else ix_ - F_.shape[1]
            print(f'sensor-ROI shift vs reference (high-pass log variance cross-correlation): dy {dy_s}, dx {dx_s}, peak corr {F_.max():.3f}')
        else:
            dy_s, dx_s = [int(v) for v in a.readnoise_shift.split(',')]
        shift_used = (int(dy_s), int(dx_s))
        yy3, xx3 = np.mgrid[:H_, :W_]
        ys3, xs3 = yy3 + dy_s, xx3 + dx_s
        cov = (ys3 >= 0) & (ys3 < rn_ref.shape[0]) & (xs3 >= 0) & (xs3 < rn_ref.shape[1])
        rn_sh = np.full((H_, W_), np.nan); rn_sh[cov] = rn_ref[ys3[cov], xs3[cov]]
        # own estimate, high-passed (DNA-PAINT signal variance is smooth on the pixel scale; camera excess is per pixel)
        rn2_hp = 0.9 ** 2 + np.clip(rn2_e - ndi.median_filter(rn2_e, 7), 0, None)
        rn_hp = np.sqrt(rn2_hp)
        print(f'read-noise map from {a.readnoise_from} shifted by ({dy_s},{dx_s}): {cov.mean()*100:.1f}% of pixels covered; '
              f'uncovered -> high-passed own estimate. Check on the covered pixels: corr(own raw, ref) {np.corrcoef(rn_e[cov], rn_sh[cov])[0, 1]:.3f}, '
              f'corr(own high-passed, ref) {np.corrcoef(rn_hp[cov], rn_sh[cov])[0, 1]:.3f}; median own raw {np.median(rn_e):.2f} / hp {np.median(rn_hp):.2f} / ref {np.median(rn_sh[cov]):.2f} e; '
              f'ref>3e px: hp median {np.median(rn_hp[cov][rn_sh[cov] > 3]):.2f} vs ref {np.median(rn_sh[cov][rn_sh[cov] > 3]):.2f}')
        rn_e = np.where(cov, rn_sh, rn_hp)
        od_ref = np.full((H_, W_), np.nan); od_ref[cov] = REF['offset_adu'][ys3[cov], xs3[cov]] - a.offset
        print(f'offset deviation vs the shifted reference map: corr {np.corrcoef(offset_dev[cov], od_ref[cov])[0, 1]:.3f}, '
              f'rms difference {np.std(offset_dev[cov] - od_ref[cov]):.2f} ADU (own sd {offset_dev.std():.2f}, ref sd {np.nanstd(od_ref):.2f})')
    H, W = low.shape
    stats = dict(offset_dev_p1=float(np.percentile(offset_dev, 1)), offset_dev_p99=float(np.percentile(offset_dev, 99)),
                 offset_dev_min=float(offset_dev.min()), rn_median=float(np.median(rn_e)), rn_p99=float(np.percentile(rn_e, 99)),
                 rn_max=float(rn_e.max()), frac_rn_gt_2x=float(np.mean(rn_e > 2 * np.median(rn_e))), nframes=n,
                 masked_frac=float(busy.mean()), readnoise_shift=shift_used)
    np.savez_compressed(a.out, offset_adu=(a.offset + offset_dev).astype(np.float32), readnoise_e=rn_e.astype(np.float32),
                        low5=low, var_adu=var_adu, gain=a.gain, readnoise_shift=np.array(shift_used),
                        readnoise_covered=(cov if a.readnoise_from else np.ones((H_, W_), bool)))
    print(json.dumps(stats, indent=1))
