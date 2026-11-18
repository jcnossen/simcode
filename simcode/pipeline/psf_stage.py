"""Stage d (psf) of the automatic pipeline: in-situ PSF from the stage-c tracks (docs/AUTO_PIPELINE.md section 2).
  1. base   Zernike 5-21 + focus                       (= the defaults of archive-2026-11-15:simcode/calibration/fit_psf_tracks.py; held-out reference)
  2. bl     + line blur (stage 1 of out/psf3d/run_final.sh); its per-frame intensities -> pattern (k, phases)
  3. ps3    bl frozen + photoselection 3 (gamma, zeta; e_a perpendicular to the fitted k_a; run_final_ps.sh)
  4. held-out: 1-3 refitted on checkerboard parity cb0, tested on cb1 (PSF frozen, x, y, z refit), paired NLL
  4b. sampling spread (psf.spread_k >= 2): 1-3 on cb0 and the held-out test repeated on spread_k independent draws of
     psf.spread_nsamples (default psf.nsamples; at most 3/4 of the cb0 samples) cb0 samples (psf_fit.select salt 0 .. k-1;
     draw 0 = step 4 when the size is psf.nsamples); sd of focus, Zernikes, line blur, gamma / zeta, held-out NLL and the paired differences, for
     independent sets of psf.nsamples samples (psf_fit.spread) -> spread.json, summary 'spread'
  5. calib.psf.zip (per axis) / calib_avg.psf.zip (angle-averaged, the network-set PSF), with the camera and its maps: the
     calibrations the later stages use (summary 'calib' / 'calib_avg'); the same content as calib json (calib.json /
     calib_avg.json + camera_maps.npz, also written), psf_images.npy"""
import json, math, os, shutil, time
import numpy as np, torch
from simcode.pipeline import psf_fit
from simcode import psf as spsf


NOLL_M = [2, 2, 1, 1, 3, 3, 0, 2, 2, 4, 4, 1, 1, 3, 3, 5, 5]      # azimuthal order m of Noll 5..21


def mirrored(z):
    """the in-situ twin of a Zernike set: even-m terms (astigmatism, spherical, ...) negated, odd-m kept. The image of an
    emitter at defocus d under W equals, up to the weak vectorial / supercritical-angle asymmetry, the one at -d under the
    mirrored W, so a bead fit (its sign follows the stage direction) or another calibration can sit on either branch"""
    z = torch.as_tensor(z, dtype=torch.float32).clone()
    for j, m in enumerate(NOLL_M[:len(z)]):
        if m % 2 == 0: z[j] = -z[j]
    return z


def _start(z, f):
    return dict(init_fit=dict(zcoef=torch.as_tensor(z, dtype=torch.float32), focus=torch.tensor(float(f))))


def _inits(ds):
    """starting points of the multi-start: psf.init = one of / a list of
      'zero'          zero Zernikes, focus 0
      'f<focus um>'   zero Zernikes at that focus ('f0.2')
      'a<focus um>'   four astigmatism starts at that focus ('a0.1'): Z5 = +-a, Z6 = +-a (a = psf.astig_init_waves)
      'beads'         the bead-fit Zernikes and their mirrored twin (mirrored()), focus 0.1 um; dropped without a bead fit
      <calib .json> / <_psf.pt>   the Zernikes + focus of an existing calibration (e.g. an earlier in-situ PSF of the set-up)
    The base fit is run from each (on psf.init_nsamples samples) and the lowest training NLL wins.
    The astigmatism starts (out/psf_v3, 2026-11-18): Zernike sets with zero astigmatism are a stationary set of the in-situ
    NLL (the defocus-sign symmetry of the emitter images makes the astigmatism gradient vanish there), so starts with zero
    Zernikes stay at low astigmatism at any focus (simulation: Zernike error 60-80 mwaves vs 14 from the true PSF; 2000
    instead of 600 iterations do not leave it). One start per astigmatism direction (Z5 +-, Z6 +-) breaks the symmetry;
    the starts on the wrong astigmatism branch end at a clearly higher NLL."""
    P = ds.c['psf']; out = []
    isnum = lambda s: s.replace('.', '', 1).isdigit()
    for p in (P['init'] if isinstance(P['init'], list) else [P['init']]):
        if p == 'zero': out.append(('zero', dict()))
        elif isinstance(p, str) and p[:1] in ('f', 'a') and isnum(p[1:]):
            f = float(p[1:])
            if p[0] == 'f': out.append((p, _start(torch.zeros(17), f)))
            else:
                for j in (0, 1):
                    for s, sn in ((1, 'p'), (-1, 'm')):
                        z = torch.zeros(17); z[j] = s * P['astig_init_waves']
                        out.append((f'{p}z{j + 5}{sn}', _start(z, f)))
        elif p == 'beads':
            b = ds.summary('beads')
            if b and not b.get('skipped'):
                zb = torch.load(b['pt'], weights_only=False)['zcoef']
                out += [('beads', _start(zb, 0.1)), ('beads_mir', _start(mirrored(zb), 0.1))]
        elif str(p).endswith('.json'):
            J = json.load(open(p))
            out.append((os.path.basename(p)[:-5], _start(J['zernike_waves'], J['focus_um'])))
        else:
            S = torch.load(p, weights_only=False)
            out.append((os.path.basename(p).replace('_psf.pt', ''), _start(S['zcoef'], S['focus'])))
    return out or [('zero', dict())]


def psf_images(pt, z=(0.0, 0.03, 0.06, 0.1, 0.15), R=15, dev='cuda'):
    F, _ = psf_fit.model_from_pt(pt, R, dev)
    with torch.no_grad():
        pos = torch.tensor([[0.0, 0.0, zz] for zz in z], device=dev)
        vm = F.angle_models()
        p = torch.stack([vm[min(k, len(vm) - 1)].render(pos) for k in range(2)], 1)
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
    # per-pixel camera model (psf.camera_maps): 'cammaps' = the cammaps stage maps, a path = those maps (e.g. the set-up's
    # calibrated maps), null = scalar offset / read noise (rn, off above)
    mp = P['camera_maps']
    mp = os.path.join(ds.out, 'cammaps', 'camera_maps.npz') if mp == 'cammaps' else mp
    maps = psf_fit.load_maps(mp)
    if maps and P['readnoise_extra_e']:        # extra white variance per pixel (model error, e.g. sample background structure)
        maps['readnoise_e'] = np.sqrt(maps['readnoise_e'] ** 2 + float(P['readnoise_extra_e']) ** 2).astype(np.float32)
    if maps: ds.log(f'psf: per-pixel offset / read-noise maps {mp}' + (f' (+ {P["readnoise_extra_e"]} e in quadrature)' if P['readnoise_extra_e'] else ''))
    starts = _inits(ds)
    kw = dict(optics=c['optics'], gain=cam['gain'], nsamples=P['nsamples'], iters=P['iters'], zmax=P['zmax_um'], readnoise_e=rn,
              offset_adu=off, dev=dev, log=ds.log, maps=maps)
    out, fits = {}, {}
    def F(name, split='', salt=0, **k):
        pre = os.path.join(d, f'fit_{name}' + (f'_{split}' if split else '') + (f'_s{salt}' if salt else ''))
        if os.path.exists(pre + '_psf.json') and not k.get('return_state') and os.path.getmtime(pre + '_psf.json') > os.path.getmtime(T):
            ds.log(f'psf: {pre} exists, reused'); return json.load(open(pre + '_psf.json'))
        ds.log(f'psf: fit {name} {split}' + (f' draw {salt}' if salt else ''))
        return psf_fit.fit(D, pre, split=split, salt=salt, **dict(kw, **k))
    # 1-2 (base from every start on psf.init_nsamples samples; the lowest NLL / px picks the start of the later fits; the base
    # fit from that start on psf.nsamples samples). psf.init_blur (default true, with psf.blur != none): the start fits include
    # the blur, so the start is chosen by the NLL of the model that is used (out/psf_v3: on 2.5 nM / 2 the base-only choice
    # missed the start whose blurred fit is 3 nats per sample better)
    ni = min(P['init_nsamples'] or P['nsamples'], P['nsamples'])
    sb = P['blur'] if (P['init_blur'] and P['blur'] != 'none') else None
    ms = {}
    if len(starts) > 1 or ni == P['nsamples']:
        for tag, kw0 in starts:
            if sb: ms[tag] = F(f'start_bl_{tag}', nsamples=ni, blur=sb, **kw0)
            else: ms[tag] = F(f'base_{tag}', **kw0) if ni == P['nsamples'] else F(f'start_{tag}', nsamples=ni, **kw0)
        best = min(ms, key=lambda t: ms[t]['nll_per_px'])
    else:
        best = starts[0][0]
    sel = None
    if len(ms) > 1 and P['init_select'] == 'refit':
        # the start by the frozen-PSF NLL with every sample's x, y, z refitted from the z multi-start (heldout_tracks) on a
        # selection set of cb0 samples (draw salt 2; the cb1 test set stays untouched): the training NLL also depends on which
        # local z optimum each sample ended in from the single z start of the fit (out/psf_v3, 2.5 nM / 2)
        pre_ = lambda t: os.path.join(d, f'fit_{"start_bl_" if sb else ("base_" if ni == P["nsamples"] else "start_")}{t}_psf.pt')
        Hs0 = psf_fit.heldout(D, {t: pre_(t) for t in ms}, 'cb0', nsamples=P['init_select_nsamples'], salt=2, dev=dev, log=ds.log, maps=maps)
        sel = {t: Hs0[t]['nll_per_px'] for t in ms}
        best = min(sel, key=sel.get)
    ini = dict(starts)[best]; ini_note = dict(chosen=best, nsamples=ni, blur=sb, select=P['init_select'], select_nll_per_px=sel,
                                              nll_per_px={t: v['nll_per_px'] for t, v in ms.items()}, focus={t: v['focus'] for t, v in ms.items()})
    ds.log(f'psf: start {best} chosen ({ini_note})')
    same = ni == P['nsamples'] and not sb
    fits['base'] = ms[best] if same else F(f'base_{best}', **ini)
    for t in ms:
        if t != best or not same: fits[(f'start_bl_{t}' if sb else f'base_{t}' if ni == P['nsamples'] else f'start_{t}')] = ms[t]
    fits['bl'], st = F(f'bl_{best}', blur=P['blur'], return_state=True, **ini)
    pat = psf_fit.pattern_from_intensities(st, c['pattern']['pitch_px'], dev=dev, log=ds.log)
    torch.save(st, os.path.join(d, 'bl_intensities.pt')); json.dump(pat, open(os.path.join(d, 'pattern.json'), 'w'), indent=1)
    cz = psf_fit.centre_vs_z(st, D, c['optics']['pixel_um'] * 1e3)
    ds.log('psf: crop centre vs z:', json.dumps(cz))
    last = 'bl'
    if P['photosel']:
        fits['ps3'] = F(f'ps3_{best}', photosel=3, kdir=pat['k'], init_fit=os.path.join(d, f'fit_bl_{best}_psf.pt'), freeze_base=True)
        last = 'ps3'
    names = ['base', 'bl'] + (['ps3'] if P['photosel'] else [])
    def cb0_chain(salt, ns=None):
        """base / bl / ps3 from the chosen start on ns (default psf.nsamples) cb0 samples of draw `salt` -> ({name: fit},
        {name: _psf.pt})"""
        ns = ns or P['nsamples']; t = '' if ns == P['nsamples'] else f'_n{ns}'; k = dict(nsamples=ns)
        pre = lambda n: os.path.join(d, f'fit_{n}_{best}{t}_cb0' + (f'_s{salt}' if salt else '') + '_psf.pt')
        r = dict(base=F(f'base_{best}{t}', 'cb0', salt, **k, **ini), bl=F(f'bl_{best}{t}', 'cb0', salt, blur=P['blur'], **k, **ini))
        if P['photosel']:
            r['ps3'] = F(f'ps3_{best}{t}', 'cb0', salt, photosel=3, kdir=pat['k'], init_fit=pre('bl'), freeze_base=True, **k)
        return r, {n: pre(n) for n in names}
    # 4: held-out
    H = None
    if P['heldout']:
        _, pts = cb0_chain(0)
        H = psf_fit.heldout(D, pts, 'cb1', nsamples=P['heldout_nsamples'], dev=dev, log=ds.log, maps=maps)
        np.savez(os.path.join(d, 'heldout.npz'), **{f'{n}_{k}': v[k] for n, v in H.items() for k in ('nll', 'chi2')})
        Hs = {n: {k: v for k, v in h.items() if k not in ('nll', 'chi2')} for n, h in H.items()}
        for a, b in zip(names[:-1], names[1:]): Hs[f'{b}_minus_{a}'] = psf_fit.paired(H, a, b)
        json.dump(Hs, open(os.path.join(d, 'heldout.json'), 'w'), indent=1)
    # 4b: sampling spread: the cb0 chain (and its held-out NLL on the same cb1 samples) on spread_k independent draws
    spread = None
    if P['spread_k'] and P['spread_k'] >= 2:
        nc = len(psf_fit.candidates(D, 0))
        # draw size: psf.spread_nsamples (default psf.nsamples), at most 3/4 of the cb0 samples (the draws must differ)
        ns = min(P['spread_nsamples'] or P['nsamples'], P['nsamples'], int(0.75 * nc)); same = ns == P['nsamples']
        reps = []
        for salt in range(P['spread_k']):
            t0 = time.time(); fr, pts = cb0_chain(salt, ns)
            h = None
            if P['heldout']:
                h = H if salt == 0 and same else psf_fit.heldout(D, pts, 'cb1', nsamples=P['heldout_nsamples'], dev=dev, log=ds.log, maps=maps)
            reps.append(dict(psf_fit.replicate_values(fr, h, names), **({} if salt == 0 and same else {'wall_s': time.time() - t0})))   # draw 0 = step 4 (reused)
        spread = psf_fit.spread(reps, ns, nc, nsamples=min(P['nsamples'], len(D['rois'])))   # sd for the main fit's sample count
        json.dump(dict(spread, replicates=reps), open(os.path.join(d, 'spread.json'), 'w'), indent=1)
        ds.log('psf: sampling spread (sd at psf.nsamples):', json.dumps({k: v for k, v in spread['sd'].items() if not isinstance(v, list)}))
    # 5: calib files
    shutil.copy(mp or os.path.join(ds.out, 'cammaps', 'camera_maps.npz'), os.path.join(d, 'camera_maps.npz'))   # the maps of the fit
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
    return dict(init=ini_note, centre_vs_z=cz, readnoise_e_model=rn if not maps else 'per-pixel map', offset_model=off if not maps else 'per-pixel map', camera_maps=mp, fits=summ, pattern=pat, heldout=Hs if H is not None else None, spread=spread,
                calib=os.path.join(d, 'calib.psf.zip'), calib_avg=os.path.join(d, 'calib_avg.psf.zip'), checks=chk)
