"""Stage a1 (camera): gain, read noise and offset from the movie (docs/QUICKSTART.md, stage camera).

The gain is measured by torch_psf white_photon_transfer (via simcode.psf): the spatially white (beyond-OTF) noise power
of background patches in differences of consecutive frames against their level (offset fixed pattern removed),
grouped by the level of the frames before / after (independent noise). Sample fluctuations (sub-threshold blinks,
haze, the illumination pattern) are band-limited by the OTF and do not enter, and
the fixed pattern does not dilute the slope (the variance-vs-level fit of same-step frame differences per pixel used
until 2026-11-18 gave 0.776 e/ADU on the quick-start example, true 0.56, and 0.56 on 1809_2, where the white-noise
estimate with the movie's offset map gives 0.43). EMCCD: excess-noise factor 2. Dark frames (optional) give offset
and read noise directly. Config values take precedence; the measured values are reported next to them.
Without dark frames the offset and the read noise are not separately identifiable: the one that is not given follows
from the mean level / variance of the patches (offset with a read-noise hint, camera.readnoise_e or 1 e)."""
import numpy as np


def read_frames(movie, n, max_values=2.5e8):
    from simcode.io.movie import open_movie
    M = open_movie(movie)
    H, W = M.shape[-2:]
    n = int(min(n, M.n, max(60, max_values // (H * W))))
    return M.read(0, n).astype(np.float32), M.n


def estimate(movie, cam, nframes=12000, log=print, dev='cuda'):
    """-> dict(gain, offset, readnoise_e, measured..., checks)"""
    from simcode.io.movie import open_movie
    from simcode.psf import white_photon_transfer
    ntot = open_movie(movie).n
    f1 = int(min(nframes, ntot))
    excess = 2.0 if cam['type'] == 'emccd' else 1.0
    _, W = white_photon_transfer(movie, frames=(0, f1), nblocks=int(np.clip(f1 // 600, 1, 25)), block=600, lag=1,
                                 excess_noise=excess, device=dev)
    k, Lm, Vm = W['slope_adu'], W['level_mean_adu'], W['var_mean_adu2']
    var_at = lambda o: max(Vm - k * (Lm - o), 0.0)                       # white variance (ADU^2) at level o
    r = dict(method='torch_psf white_photon_transfer (lag 1)', n_frames_used=f1, n_frames_movie=int(ntot),
             levels=[b[0] for b in W['bins']], var=[b[1] for b in W['bins']], npx=[b[2] for b in W['bins']],
             slope=float(k), intercept=float(W['intercept_adu2']), r2=float(W['r2']), slope_rel_se=float(W['slope_rel_se']),
             slope_spread=float(W['slope_spread']), warning=W['warning'],
             gain_measured=float(W['gain_e_per_adu']), level_span_adu=float(W['level_span_adu']), level_mean_adu=float(Lm),
             var_mean_adu2=float(Vm), fpn_sd_adu=float(W['fpn_sd_adu']))
    log(f'camera: white photon transfer over {len(r["levels"])} bins ({r["level_span_adu"]:.1f} ADU span): gain {r["gain_measured"]:.3f} e/ADU, '
        f'r2 {r["r2"]:.3f}, slope rel. se {r["slope_rel_se"]:.3f}, per-block slope spread {r["slope_spread"]:.3f}' + (f'; {W["warning"]}' if W['warning'] else ''))
    # offset
    if cam.get('offset_adu') is not None:
        off, src = float(cam['offset_adu']), 'config'
    elif cam.get('dark'):
        dk, _ = read_frames(cam['dark'], 2000); off, src = float(np.median(dk)), 'dark'
    else:
        rn_hint = cam.get('readnoise_e') or 1.0
        g = cam.get('gain_e_per_adu') or r['gain_measured']
        off = float(Lm - (Vm - (rn_hint / g) ** 2) / k)                  # level at which the variance is the read noise alone
        src = f'estimated (white-noise variance, read noise hint {rn_hint} e)'
    gain = float(cam['gain_e_per_adu']) if cam.get('gain_e_per_adu') else r['gain_measured']
    gsrc = 'config' if cam.get('gain_e_per_adu') else 'measured'
    if cam.get('readnoise_e') is not None:
        rn, rsrc = float(cam['readnoise_e']), 'config'
    elif cam.get('dark'):
        dk, _ = read_frames(cam['dark'], 2000)
        rn, rsrc = float(np.median(dk.std(0)) * gain), 'dark'
    else:
        rn, rsrc = float(np.sqrt(var_at(off)) * gain), 'white-noise variance at the offset'
    med = float(np.median(read_frames(movie, 200)[0]))
    r.update(gain=gain, gain_source=gsrc, offset=off, offset_source=src, readnoise_e=rn, readnoise_source=rsrc,
             median_level_adu=med, bg_e_per_px=float((med - off) * gain))
    for o in (99.3, 100.0, off):
        r[f'rn_e_at_{o:g}'] = float(np.sqrt(var_at(o)) * excess / k)
    chk = [dict(name='level bins >= 5 (else gain must come from the config)', value=len(r['levels']), ok=bool(len(r['levels']) >= 5 or gsrc == 'config')),
           dict(name='gain determined by the movie: slope rel. se < 0.1 (or gain from the config)', value=r['slope_rel_se'],
                ok=bool(gsrc == 'config' or r['slope_rel_se'] < 0.1)),
           dict(name='measured gain within 20 % of the config value', value=r['gain_measured'],
                ok=bool(gsrc != 'config' or (np.isfinite(r['gain_measured']) and abs(r['gain_measured'] / gain - 1) < 0.2)),
                warn_only=True),
           dict(name='offset known (config or dark frames)', value=src, ok=src in ('config', 'dark'), warn_only=True)]
    r['checks'] = chk
    log(f'camera: gain {gain:.3f} ({gsrc}), offset {off:.2f} ({src}), read noise {rn:.2f} e ({rsrc})')
    return r
