"""Stage a1 (camera): gain, read noise and offset from the movie (docs/AUTO_PIPELINE.md section 2).

Same estimator as simcode/calibration/camera_gain.py (robust variance-vs-level fit of same-step frame differences, frames 6
apart, 5 %-each-side trimmed variance corrected to a normal sigma^2), generalised: the level bins span the movie's own
range (camera_gain.py: fixed 101-139 ADU), EMCCD gets the excess-noise factor 2, dark frames (optional) give offset and
read noise directly. Config values take precedence; the measured values are reported next to them."""
import numpy as np

TRIM_CORR = 1 - 2 * 1.6449 * 0.10314 / 0.9    # variance of a 5 %-each-side trimmed normal / sigma^2 (= camera_gain.py)


def read_frames(movie, n, max_values=2.5e8):
    from simcode.io.movie import open_movie
    M = open_movie(movie)
    H, W = M.shape[-2:]
    n = int(min(n, M.n, max(60, max_values // (H * W))))
    return M.read(0, n).astype(np.float32), M.n


def var_vs_level(mov, nbins=12, min_px=30, rn_excl=0.9):
    """per-pixel mean level (not the median: integer-valued movies) in quantile bins; pixels in the top (1 - rn_excl) of the
    low-level variance map are excluded (sCMOS high-read-noise pixels would otherwise dominate the low-level bins)"""
    d = (mov[6:] - mov[:-6]).reshape(len(mov) - 6, -1)
    lvl = mov.mean(0).ravel()
    mad = (1.4826 * np.median(np.abs(d - np.median(d, 0)), 0)) ** 2 / 2
    ok = mad <= np.quantile(mad, rn_excl)
    bins = np.quantile(lvl[ok], np.linspace(0.02, 0.98, nbins + 1))
    X, Y, N = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = ok & (lvl >= lo) & (lvl < hi)
        if s.sum() < min_px: continue
        x = d[:, s].ravel(); q = np.percentile(x, [5, 95]); xt = x[(x > q[0]) & (x < q[1])]
        X.append(lvl[s].mean()); Y.append(xt.var() / TRIM_CORR / 2); N.append(int(s.sum()))
    return np.array(X), np.array(Y), np.array(N)


def estimate(movie, cam, nframes=12000, log=print):
    """-> dict(gain, offset, readnoise_e, measured..., checks)"""
    mov, ntot = read_frames(movie, nframes)
    X, Y, N = var_vs_level(mov)
    r = dict(n_frames_used=int(len(mov)), n_frames_movie=int(ntot), levels=X.tolist(), var=Y.tolist(), npx=N.tolist())
    excess = 2.0 if cam['type'] == 'emccd' else 1.0
    if len(X) >= 3:
        p = np.polyfit(X, Y, 1)
        pred = np.polyval(p, X); r2 = 1 - ((Y - pred) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12)
        g_meas = excess / p[0] if p[0] > 0 else float('nan')
        r.update(slope=float(p[0]), intercept=float(p[1]), r2=float(r2), gain_measured=float(g_meas),
                 level_span_adu=float(X.max() - X.min()))
    else:
        p = None; r.update(gain_measured=float('nan'), r2=float('nan'), level_span_adu=0.0)
    log(f'camera: variance-vs-level over {len(X)} bins ({r["level_span_adu"]:.1f} ADU span): gain {r["gain_measured"]:.3f} e/ADU, r2 {r["r2"]:.3f}')
    # offset
    if cam.get('offset_adu') is not None:
        off, src = float(cam['offset_adu']), 'config'
    elif cam.get('dark'):
        dk, _ = read_frames(cam['dark'], 2000); off, src = float(np.median(dk)), 'dark'
    else:
        rn_hint = cam.get('readnoise_e') or 1.0
        g = cam.get('gain_e_per_adu') or r['gain_measured']
        # level at which the fitted variance equals the read noise alone (rn_hint e)
        off = float(((rn_hint / g) ** 2 - p[1]) / p[0]) if p is not None and p[0] > 0 else float(np.percentile(mov, 1))
        src = f'estimated (variance intercept, read noise hint {rn_hint} e)'
    gain = float(cam['gain_e_per_adu']) if cam.get('gain_e_per_adu') else r['gain_measured']
    gsrc = 'config' if cam.get('gain_e_per_adu') else 'measured'
    if cam.get('readnoise_e') is not None:
        rn, rsrc = float(cam['readnoise_e']), 'config'
    elif cam.get('dark'):
        dk, _ = read_frames(cam['dark'], 2000)
        rn, rsrc = float(np.median(dk.std(0)) * gain), 'dark'
    elif p is not None:
        rn, rsrc = float(np.sqrt(max(np.polyval(p, off), 0)) * gain), 'variance intercept at the offset'
    else:
        rn, rsrc = 1.0, 'default'
    r.update(gain=gain, gain_source=gsrc, offset=off, offset_source=src, readnoise_e=rn, readnoise_source=rsrc,
             median_level_adu=float(np.median(mov)), bg_e_per_px=float((np.median(mov) - off) * gain))
    for o in (99.3, 100.0, off):
        if p is not None and p[0] > 0: r[f'rn_e_at_{o:g}'] = float(np.sqrt(max(np.polyval(p, o), 0)) * excess / p[0])
    chk = [dict(name='level bins >= 5 (else gain must come from the config)', value=len(X), ok=bool(len(X) >= 5 or gsrc == 'config')),
           dict(name='variance-vs-level r2 > 0.9 (or gain from the config)', value=r['r2'], ok=bool(gsrc == 'config' or (r['r2'] > 0.9))),
           dict(name='measured gain within 20 % of the config value', value=r['gain_measured'],
                ok=bool(gsrc != 'config' or (np.isfinite(r['gain_measured']) and abs(r['gain_measured'] / gain - 1) < 0.2)),
                warn_only=True),
           dict(name='offset known (config or dark frames)', value=src, ok=src in ('config', 'dark'), warn_only=True)]
    r['checks'] = chk
    log(f'camera: gain {gain:.3f} ({gsrc}), offset {off:.2f} ({src}), read noise {rn:.2f} e ({rsrc})')
    return r
