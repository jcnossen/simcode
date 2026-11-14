"""Stage e (train): PSF-matched model of the dataset, the recipe of runs/psfmatch/chain2.sh + eval.sh
(docs/AUTO_PIPELINE.md section 6), as a wrapper around the existing trainers (unchanged entry points, through
simcode/auto/runwrap.py for the in-memory preset overrides of a non-DMD setup).

Steps (resumable: a trainer whose <dir>/final.pt exists is skipped; quick mode = train.quick_steps per trainer, in its own
directory 'quick/'):
  det -> hd (init hd_sc5nb) and p2det (init_n1 det) -> p2head (det = p2det at 25k steps, init hd) -> p2headL
  gate scenes (simcode.hd.gate, dataset PSF = psf/calib_avg.json) -> dumps of pass 1 (det + hd) and pass 2 (p2det + p2headL)
  -> sim-sigma solve calibrations (analysis/psfmatch/fit_cal_set.py; the _real multiplier copied from the m1809 templates)
  -> acceptance on the held-out seed: sigma zcal of pass 1 and pass 2, pass-2 vs pass-1 RMS / CRLB on the realistic scene
  -> <dir>/model.json (format of simcode/models.json, absolute paths)
Not in this wrapper (reported): the selection-aware sigma rescale sig_u (needs a simulated movie with ground truth,
analysis/psf3d_nets/sigcal.py) and the sim-movie GT sanity gate of runs/psfmatch/eval.sh."""
import json, os, shutil
import numpy as np

T_BASE = dict(det=['-m', 'simcode.det.up2', 'train', '--batch', '16', '--size', '64', '--lr', '1e-3', '--w_off', '1.0', '--nnw', '1',
                   '--seed', '0'],
              hd=['-m', 'simcode.hd.sc4', 'train', '--batch', '16', '--lr', '1.5e-4', '--ch', '96', '--tau', '15,80',
                  '--pair_frac', '0.25', '--dens', '0.0005,0.04', '--med', '40,2500', '--nnw', '1', '--seed', '1', '--w_mr', '2.0', '--w_slope', '50.0',
                  '--bf16', '1', '--pct', '0.1'],
              p2det=['-m', 'simcode.p2.det2', 'train', '--batch', '16', '--size', '64', '--lr', '3e-4', '--pct', '0.05', '--w_off', '1.0', '--seed', '11'],
              p2head=['-m', 'simcode.p2.head2', 'train', '--batch', '16', '--lr', '1.5e-4', '--pct', '0.1', '--w_mr', '2.0', '--w_slope', '50.0',
                      '--seed', '21', '--no_illum', '0', '--pert_model', 'v1'],
              p2headL=['-m', 'simcode.p2.head2', 'train', '--batch', '16', '--lr', '1e-4', '--pct', '0.05', '--w_mr', '2.0', '--w_slope', '50.0',
                       '--seed', '31', '--no_illum', '0', '--hold_v1', '15'])
EVERY = dict(det=None, hd=('2000', '6000'), p2det=('2500', '5000'), p2head=('4000', '8000'), p2headL=('2000', '10000'))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCEN_FIT = 'rs05qm'


def presets(ds):
    """in-memory realsim.PRESETS overrides for a non-DMD setup (pattern k and pitch, camera maps, pixel size); none for the
    DMD-SIMFLUX defaults (the presets ARE that setup). NOTE the step direction: realsim draws phi_s = phi0 + s * spacing with
    spacing > 0; a setup that steps the other way needs the measured (signed) spacing in the convention of realsim.pattern
    (the nanoruler preset error, out/nanoruler/STATUS.md), which is not derived automatically here."""
    c = ds.c; P = ds.summary('psf'); pat = P['pattern']
    pitch0 = 2.07
    if abs(np.mean(pat['pitch_px']) / pitch0 - 1) < 0.05 and abs(c['optics']['pixel_um'] - 0.10833) < 1e-3:
        return {}
    return dict(presets=dict(common=dict(k=pat['k'], cam=os.path.join(ds.out, 'psf', 'camera_maps.npz'),
                                         z_med=P['fits'][list(P['fits'])[-1]].get('z_median', 0.05))),
                px_nm=c['optics']['pixel_um'] * 1e3)


def run(ds):
    c, Tc = ds.c, ds.c['train']
    if Tc['model'] and not Tc['quick']:
        return dict(model=Tc['model'], skipped=True, note='existing model from the config; nothing trained', checks=[])
    quick = Tc['quick']
    D = os.path.join(ds.dir('train'), 'quick' if quick else 'full'); os.makedirs(D, exist_ok=True)
    cal = ds.summary('psf')['calib_avg']
    W = ds.wrapfile(dict(presets(ds), title='auto-train'), 'wrap_train')
    steps = {k: (Tc['quick_steps'] if quick else v) for k, v in Tc['steps'].items()}
    hold = os.path.join(REPO, 'out', 'sc4', 'heldout5.pt')

    def stage(name, extra, need=10):
        d = os.path.join(D, name)
        if os.path.exists(os.path.join(d, 'final.pt')):
            ds.log(f'train: {name} final.pt exists, skipped'); return os.path.join(d, 'final.pt')
        a = T_BASE[name] + ['--out', d, '--steps', str(steps[name]), '--calib', cal] + extra
        ev = EVERY[name]
        if ev:
            a += ['--eval_every', str(max(int(ev[0]), steps[name] + 1) if quick else ev[0]), '--ckpt_every', str(max(int(ev[1]), steps[name] + 1) if quick else ev[1])]
        xt = Tc['extra']['det' if name == 'det' else 'hd' if name == 'hd' else 'p2']
        if xt: a += xt.split()
        ds.sh(['-m', 'simcode.auto.runwrap'] + a[1:] if a[0] == '-m' else a, os.path.join(D, name + '.log'), gpu=ds.gpu(need), wrap=W)
        return os.path.join(d, 'final.pt')

    from simcode import config as SC
    det_init = Tc['init']['det'] or (SC.resolve(SC.model_set(Tc['real_mult_from'])['det']) if quick else None)   # quick: a few steps on top of the template detector
    det = stage('det', ['--init', det_init] if det_init else [])
    hd = stage('hd', ['--det', det, '--init', Tc['init']['hd']] + (['--hold', hold] if os.path.exists(hold) else []))
    p2det = stage('p2det', ['--init_n1', det])
    p25 = os.path.join(D, 'p2det', 'ckpt_25000.pt')
    p2head = stage('p2head', ['--det', p25 if os.path.exists(p25) else p2det, '--init', hd])
    p2headL = stage('p2headL', ['--det', p2det, '--init', p2head])
    name = ('q' if quick else 'a') + c['name']
    M = os.path.join(D, 'models'); os.makedirs(M, exist_ok=True)
    files = {}
    for k, f in (('det', det), ('head', hd), ('p2_det', p2det), ('p2_head', p2headL)):
        files[k] = os.path.join(M, f'{k}_{name}.pt'); shutil.copyfile(f, files[k])
    # ---- gate scenes, dumps, calibrations
    tag = 'g' + c['name']
    O = os.path.join(D, 'psfmatch'); G = os.path.join(O, 'gate_' + tag); FE = os.path.join(O, 'fe_' + tag)
    seeds = '202,301' if quick else '202,301,302,303'
    scen = 'iso,' + SCEN_FIT if quick else 'iso,pair1,pair2,' + SCEN_FIT
    nt = '400' if quick else '3500'
    env = dict(SIMCODE2_GATE_CALIB=cal, SIMCODE2_GATE_DIR=G, SIMCODE_OUT=D)
    if not os.path.exists(os.path.join(G, 'done')):
        ds.sh(['-m', 'simcode.auto.runwrap', 'simcode.hd.gate', '--scen', scen, '--fronts', 'sc', '--sc', 'hd_sc5nb.pt', '--sccal',
               'calib/calib_sc_hd_sc5nbb15_pm_bright.json', '--tag', '_hd_cur', '--out', G, '--ntarget', nt, '--seeds', seeds],
              os.path.join(D, 'gate.log'), gpu=ds.gpu(8), wrap=W, env=env)
        open(os.path.join(G, 'done'), 'w').close()
    if not os.path.exists(os.path.join(FE, 'done')):
        ds.sh(['-m', 'simcode.auto.runwrap', 'simcode.p2.dump', '--scen', scen, '--seeds', seeds, '--ntarget', nt, '--out', FE, '--fronts',
               f's1={files["det"]}:{files["head"]},s2={files["p2_det"]}:{files["p2_head"]}'], os.path.join(D, 'dump.log'), gpu=ds.gpu(8), wrap=W, env=env)
        open(os.path.join(FE, 'done'), 'w').close()
    C = os.path.join(M, 'calib'); os.makedirs(C, exist_ok=True)
    cals = dict(cal_sim=os.path.join(C, f'calib_sc_hd_{name}.json'), cal_real=os.path.join(C, f'calib_sc_hd_{name}_real.json'),
                p2_cal_sim=os.path.join(C, f'calib_p2L_{name}.json'), p2_cal_real=os.path.join(C, f'calib_p2L_{name}_real.json'))
    fcs = os.path.join(REPO, 'analysis', 'psfmatch', 'fit_cal_set.py')
    rm = Tc['real_mult_from']
    from simcode import config as SC
    tmpl = SC.model_set(rm)
    fallback = []
    for fr, base, ks, ksr in (('s1', 'calib/calib_sc_hd_sc5nbb15_pm_bright.json', 'cal_sim', 'cal_real'), ('s2', 'calib/calib_p2L.json', 'p2_cal_sim', 'p2_cal_real')):
        ds.sh(['-m', 'simcode.auto.runwrap', fcs, fr, tag, SCEN_FIT, base, cals[ks], cals[ksr], tmpl[ksr]],
              os.path.join(D, f'fitcal_{fr}.log'), gpu=ds.gpu(4), wrap=W, env=env, check=not quick)
        if quick and not (os.path.exists(cals[ks]) and os.path.exists(cals[ksr])):
            # quick plumbing: networks after a few steps detect nothing -> the fit has no targets; copy the template set's calibrations
            shutil.copyfile(SC.resolve(tmpl[ks]), cals[ks]); shutil.copyfile(SC.resolve(tmpl[ksr]), cals[ksr]); fallback.append(fr)
            ds.log(f'train (quick): calibration fit of {fr} failed (no matched targets), copied the {rm} calibrations')
    # ---- acceptance (held-out seed 301)
    acc_js = os.path.join(D, 'accept.json')
    ds.sh(['-m', 'simcode.auto.runwrap', os.path.join(REPO, 'simcode', 'auto', 'accept.py'), FE, cals['cal_sim'], cals['p2_cal_sim'],
           scen, acc_js], os.path.join(D, 'accept.log'), gpu=ds.gpu(4), wrap=W, env=env, check=not quick)
    if not os.path.exists(acc_js):
        json.dump(dict(zcal={}, rms={}, n={}, p2_over_p1=None, note='acceptance failed (quick plumbing: no matched targets)'), open(acc_js, 'w'))
    A = json.load(open(acc_js))
    lo, hi = Tc['gates']['sigma_zcal']
    chk = []
    for k_, v in A['zcal'].items():
        chk.append(dict(name=f'sigma zcal {k_} in [{lo}, {hi}]', value=v, ok=bool(lo <= v <= hi), warn_only=quick))
    if not A['zcal'] and not quick:
        chk.append(dict(name='acceptance statistics computed', value=A.get('note'), ok=False))
    if A.get('p2_over_p1') is not None:
        chk.append(dict(name=f'pass-2 / pass-1 RMS/CRLB on {SCEN_FIT} (200-2000 ph) <= 1', value=A['p2_over_p1'], ok=A['p2_over_p1'] <= 1.0, warn_only=quick))
    gates = 'NOT EVALUATED' if not chk else 'PASS' if all(k['ok'] for k in chk) else 'FAIL'
    S = {name: dict(_doc=f'simcode.auto model of dataset {c["name"]} ({"QUICK plumbing test, not usable" if quick else "PSF-matched recipe"}); '
                         f'PSF {cal}; sig_u not fitted', _gates=gates, psf=cal, **files, **cals, sig_u='')}
    json.dump(S, open(os.path.join(ds.dir('train'), 'model.json'), 'w'), indent=1)
    return dict(model=name, quick=quick, calib_fallback=fallback, steps=steps, dir=D, files=files, calibs=cals, accept=A, gates=gates,
                model_json=os.path.join(ds.dir('train'), 'model.json'), wrap=json.load(open(W)), checks=chk)
