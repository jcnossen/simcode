"""Stage d (psf) of the automatic pipeline: in-situ PSF from the stage-c tracks (docs/AUTO_PIPELINE.md section 2).
  1. base   Zernike 5-21 + focus                       (= the defaults of archive-2026-11-15:simcode/calibration/fit_psf_tracks.py; held-out reference)
  2. bl     + line blur (stage 1 of out/psf3d/run_final.sh); its per-frame intensities -> pattern (k, phases)
  3. ps3    bl frozen + photoselection 3 (gamma, zeta; e_a perpendicular to the fitted k_a; run_final_ps.sh)
  4. held-out: 1-3 refitted on checkerboard parity cb0, tested on cb1 (PSF frozen, x, y, z refit), paired NLL
  5. calib.psf.zip (per axis) / calib_avg.psf.zip (angle-averaged, the network-set PSF), with the camera and its maps: the
     calibrations the later stages use (summary 'calib' / 'calib_avg'); the same content as calib json (calib.json /
     calib_avg.json + camera_maps.npz, also written), psf_images.npy"""
import json, math, os, shutil
import numpy as np, torch
from simcode.pipeline import psf_fit
from simcode import psf as spsf


def _inits(ds):
    """starting points of the multi-start: psf.init = one of / a list of 'zero' | 'f<focus um>' | 'beads' | <_psf.pt>; 'beads' without a bead
    fit is dropped. The base fit is run from each and the lowest NLL wins (bead Zernikes can sit in another basin than the
    in-situ optimum, and zero init can fall into the z / focus gauge basin near z = 0: out/psf3d/run_final.sh)"""
    P = ds.c['psf']['init']; out = []
    for p in (P if isinstance(P, list) else [P]):
        if p == 'zero': out.append(('zero', dict()))
        elif isinstance(p, str) and p.startswith('f') and p[1:].replace('.', '', 1).isdigit():   # 'f0.2': zero Zernikes, focus 0.2 um
            out.append((p, dict(init_fit=dict(zcoef=torch.zeros(17), focus=torch.tensor(float(p[1:]))))))
        elif p == 'beads':
            b = ds.summary('beads')
            if b and not b.get('skipped'):
                out.append(('beads', dict(init_zcoef=torch.load(b['pt'], weights_only=False)['zcoef'])))
        else:
            S = torch.load(p, weights_only=False)
            out.append((os.path.basename(p).replace('_psf.pt', ''), dict(init_fit=dict(zcoef=S['zcoef'], focus=S['focus']))))
    return out or [('zero', dict())]


def psf_images(pt, z=(0.0, 0.03, 0.06, 0.1, 0.15), R=15, dev='cuda'):
    F, _ = psf_fit.model_from_pt(pt, R, dev)
    with torch.no_grad():
        pos = torch.tensor([[0.0, 0.0, zz] for zz in z], device=dev)
        p = F.psf_of(pos, F.G['focus'].expand(len(z)))
        if p.dim() == 3: p = torch.stack([p, p], 1)
    return p.cpu().numpy()          # [nz, 2 angles, R, R]


def calib_dict(ds, fit_js, fit_pt, pat, cam, maps_name, avg):
    c, o = ds.c, ds.c['optics']
    S = torch.load(fit_pt, weights_only=False); J = json.load(open(fit_js)); ex = S.get('extras', {})
    k = pat['k']
    d = dict(source=f'simcode.pipeline in-situ calibration of {c["movie"]} (dataset {c["name"]}): {"angle-averaged" if avg else "per pattern axis"} PSF '
                    f'(Zernike 5-21 + focus + {c["psf"]["blur"]} blur, then frozen + photoselection {c["psf"]["photosel"]}); fit {fit_pt}',
             pixelsize_um=o['pixel_um'], wavelength_um=o['wavelength_um'], NA=float(S['NA']), n_imm=o['n_imm'], n_med=o['n_med'],
             apod=[float(v) for v in S['apod'].tolist()], zernike_noll=list(range(5, 22)), zernike_waves=[float(v) for v in S['zcoef'].tolist()],
             focus_um=float(S['focus']),
             camera=dict(gain_e_per_adu=cam['gain'], offset_adu=cam['offset'], readnoise_e=cam['readnoise_e'], maps=maps_name,
                         note=f'gain {cam["gain_source"]}, offset {cam["offset_source"]}, read noise {cam["readnoise_source"]}; maps = simcode.calibration.camera_maps masked by the stage-c spots'),
             pattern=dict(k_rad_per_px=k, pitch_nm=[float(2 * math.pi / np.hypot(*kk) * o['pixel_um'] * 1e3) for kk in k],
                          depth=pat['mod'], phase=pat['phase'], relint=pat['relint'],
                          note='in-situ estimate from the track intensities (blocked slot order, angle = slot // 3); depth = regression '
                               'modulation (diluted by bg / kinetics, low); used only by the simulators: the pipeline estimates the pattern from the data'),
             tirf_depth_um=o['tirf_depth_um'], emitter_z_median_um=float(J.get('z_median', 0.05)))
    if 'blur_px2' in ex: d['psf_blur_px2'] = ex['blur_px2'][0]
    if 'dipole_M' in ex:
        M = ex['dipole_M']
        if avg: d['psf_dipole_M'] = [[(M[0][i][j] + M[1][i][j]) / 2 for j in range(3)] for i in range(3)]
        else: d['psf_dipole_M_angle'] = M
    d['psf3d'] = dict(fit=fit_pt, gamma=ex.get('gamma'), zeta=ex.get('zeta'), e_dir=ex.get('e_dir'), resid_var=J.get('resid_var'), avg=avg)
    return d


def run(ds):
    c, P = ds.c, ds.c['psf']
    dev = ds.torch_dev(8)
    d = ds.dir('psf'); T = os.path.join(ds.out, 'tracks', 'tracks.npz')
    cam = ds.summary('camera')
    D = np.load(T); D = {k: D[k] for k in D.files}
    rn = P['readnoise_e'] if P['readnoise_e'] is not None else cam['readnoise_e']
    off = P['offset_adu'] if P['offset_adu'] is not None else cam['offset']
    starts = _inits(ds)
    kw = dict(optics=c['optics'], gain=cam['gain'], nsamples=P['nsamples'], iters=P['iters'], zmax=P['zmax_um'], readnoise_e=rn,
              offset_adu=off, lut=P['lut'], lut_sub=P['lut_sub'], dev=dev, log=ds.log)
    out, fits = {}, {}
    def F(name, split='', **k):
        pre = os.path.join(d, f'fit_{name}' + (f'_{split}' if split else ''))
        if os.path.exists(pre + '_psf.json') and not k.get('return_state') and os.path.getmtime(pre + '_psf.json') > os.path.getmtime(T):
            ds.log(f'psf: {pre} exists, reused'); return json.load(open(pre + '_psf.json'))
        ds.log(f'psf: fit {name} {split}')
        return psf_fit.fit(D, pre, split=split, **kw, **k)
    # 1-2 (base from every start; the lowest NLL / px picks the start of the later fits)
    ms = {}
    for tag, kw0 in starts:
        ms[tag] = F(f'base_{tag}', **kw0)
    best = min(ms, key=lambda t: ms[t]['nll_per_px'])
    ini = dict(starts)[best]; ini_note = dict(chosen=best, nll_per_px={t: v['nll_per_px'] for t, v in ms.items()},
                                              focus={t: v['focus'] for t, v in ms.items()})
    ds.log(f'psf: start {best} chosen ({ini_note})')
    fits['base'] = ms[best]
    for t in ms:
        if t != best: fits[f'base_{t}'] = ms[t]
    fits['bl'], st = F(f'bl_{best}', blur=P['blur'], return_state=True, **ini)
    pat = psf_fit.pattern_from_intensities(st, c['pattern']['pitch_px'], dev=dev, log=ds.log)
    torch.save(st, os.path.join(d, 'bl_intensities.pt')); json.dump(pat, open(os.path.join(d, 'pattern.json'), 'w'), indent=1)
    cz = psf_fit.centre_vs_z(st, D, c['optics']['pixel_um'] * 1e3)
    ds.log('psf: crop centre vs z:', json.dumps(cz))
    last = 'bl'
    if P['photosel']:
        fits['ps3'] = F(f'ps3_{best}', photosel=3, kdir=pat['k'], init_fit=os.path.join(d, f'fit_bl_{best}_psf.pt'), freeze_base=True)
        last = 'ps3'
    # 4: held-out
    H = None
    if P['heldout']:
        F(f'base_{best}', 'cb0', **ini); F(f'bl_{best}', 'cb0', blur=P['blur'], **ini)
        names = ['base', 'bl']
        if P['photosel']:
            F(f'ps3_{best}', 'cb0', photosel=3, kdir=pat['k'], init_fit=os.path.join(d, f'fit_bl_{best}_cb0_psf.pt'), freeze_base=True); names.append('ps3')
        H = psf_fit.heldout(D, {n: os.path.join(d, f'fit_{n}_{best}_cb0_psf.pt') for n in names}, 'cb1', nsamples=P['heldout_nsamples'], dev=dev, log=ds.log)
        np.savez(os.path.join(d, 'heldout.npz'), **{f'{n}_{k}': v[k] for n, v in H.items() for k in ('nll', 'chi2')})
        Hs = {n: {k: v for k, v in h.items() if k not in ('nll', 'chi2')} for n, h in H.items()}
        for a, b in zip(names[:-1], names[1:]): Hs[f'{b}_minus_{a}'] = psf_fit.paired(H, a, b)
        json.dump(Hs, open(os.path.join(d, 'heldout.json'), 'w'), indent=1)
    # 5: calib files
    shutil.copy(os.path.join(ds.out, 'cammaps', 'camera_maps.npz'), os.path.join(d, 'camera_maps.npz'))
    js, pt = os.path.join(d, f'fit_{last}_{best}_psf.json'), os.path.join(d, f'fit_{last}_{best}_psf.pt')
    for avg, fn in ((False, 'calib.json'), (True, 'calib_avg.json')):
        json.dump(calib_dict(ds, js, pt, pat, cam, 'camera_maps.npz', avg), open(os.path.join(d, fn), 'w'), indent=1)
        spsf.write_calib_zip(os.path.join(d, fn), device='cpu')   # the same calibration as psf.zip (calib.psf.zip, calib_avg.psf.zip)
    np.save(os.path.join(d, 'psf_images.npy'), psf_images(pt, dev=dev))
    # checks
    fl = fits[last]
    Zn = float(np.sqrt(np.sum(np.square(fl['zernike']))))
    chk = [dict(name='resid var in [0.85, 1.1]', value=fl['resid_var'], ok=0.85 <= fl['resid_var'] <= 1.1),
           dict(name='|Z| < 0.15 waves', value=Zn, ok=Zn < 0.15),
           dict(name='z median inside (0, zmax)', value=fl['z_median'], ok=0.005 < fl['z_median'] < 0.98 * P['zmax_um'], warn_only=True),
           dict(name='the two k in different 45-degree sectors (pipeline angle_index)', value=pat['sectors'], ok=pat['sector_ok']),
           dict(name='pitch inside pattern.pitch_px', value=pat['pitch_px'], ok=all(c['pattern']['pitch_px'][0] < p < c['pattern']['pitch_px'][1] for p in pat['pitch_px']))]
    tilt = max(abs(cz['slope_x_nm_per_100nm']), abs(cz['slope_y_nm_per_100nm']))
    chk.append(dict(name='z-dependent crop-centre offset < 5 nm per 100 nm z', value=tilt, ok=tilt < 5, warn_only=True))
    if H is not None:
        for k_, v in Hs.items():
            if '_minus_' in k_:
                chk.append(dict(name=f'held-out NLL {k_} <= 0 (paired)', value=v, ok=v['mean'] <= 0, warn_only=True))
    summ = {n: {k: v for k, v in f.items() if k not in ('optics',)} for n, f in fits.items()}
    return dict(init=ini_note, centre_vs_z=cz, readnoise_e_model=rn, offset_model=off, fits=summ, pattern=pat, heldout=Hs if H is not None else None,
                calib=os.path.join(d, 'calib.psf.zip'), calib_avg=os.path.join(d, 'calib_avg.psf.zip'), checks=chk)
