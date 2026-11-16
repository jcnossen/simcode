"""Stage e (train): PSF-matched model of the dataset, the recipe of runs/psfmatch/chain2.sh + eval.sh
(docs/AUTO_PIPELINE.md section 6), as a wrapper around the trainers simcode.detector.train and simcode.emitter.train (through
simcode/pipeline/run_with_overrides.py for the in-memory preset overrides of a non-DMD setup).

Steps (resumable: a trainer whose <dir>/final.pt exists is skipped; quick mode = train.quick_steps per trainer, in its own
directory 'quick/'):
  detector -> emitter (init models/pretrain_emitter_pass1.pt) and detector_pass2 (zero-initialised from detector: the dedicated pass-2 detector, the
  dataset's own pattern only) -> detector_pass2_mix (fine-tuned on the 50 / 50 mix of its own and generic patterns, lr 3e-4)
  -> detector_pass2_cont (the same mix, lr 1e-4); emitter_pass2 (detector = detector_pass2 at 25k steps, init emitter) ->
  emitter_pass2_long -> emitter_pass2_final (detector = detector_pass2). The pattern draw of the pass-2 stages is fixed in the
  trainers (simulate/batches.P_STD_DEDICATED / P_STD; generic patterns: simulate/acquisition.AcquisitionSim._generic): own
  pattern only in detector_pass2, half own / half generic in every other pass-2 stage (models/README.md)
  gate scenes (simcode.benchmark.gate_scenes, dataset PSF = psf/calib_avg.json) -> dumps of pass 1 (detector + emitter) and pass 2
  (detector_pass2_cont + emitter_pass2_final)
  -> sim-sigma solve calibrations (analysis/psfmatch/fit_cal_set.py; the _real multiplier copied from the m1809 templates)
  -> acceptance on the held-out seed: sigma zcal of pass 1 and pass 2, pass-2 vs pass-1 RMS / CRLB on the realistic scene
  -> <dir>/model.json (format of simcode/models.json, absolute paths)
Not in this wrapper (reported): the selection-aware sigma rescale sig_u (needs a simulated movie with ground truth,
analysis/psf3d_nets/sigcal.py) and the sim-movie GT sanity gate of runs/psfmatch/eval.sh."""
import json, os, shutil
from simcode import config
import numpy as np

T_BASE = dict(detector=['-m', 'simcode.detector.train', '--batch', '16', '--size', '64', '--lr', '1e-3', '--w_off', '1.0', '--nnw', '1',
                   '--seed', '0'],
              emitter=['-m', 'simcode.emitter.train', '--batch', '16', '--lr', '1.5e-4', '--ch', '96', '--tau', '15,80',
                  '--pair_frac', '0.25', '--dens', '0.0005,0.04', '--med', '40,2500', '--nnw', '1', '--seed', '1', '--w_mr', '2.0', '--w_slope', '50.0',
                  '--bf16', '1', '--pct', '0.1'],
              detector_pass2=['-m', 'simcode.detector.train', '--batch', '16', '--size', '64', '--lr', '3e-4', '--pct', '0.05', '--w_off', '1.0', '--seed', '11'],
              detector_pass2_mix=['-m', 'simcode.detector.train', '--batch', '16', '--size', '64', '--lr', '3e-4', '--pct', '0.05', '--w_off', '1.0', '--seed', '61'],
              detector_pass2_cont=['-m', 'simcode.detector.train', '--batch', '16', '--size', '64', '--lr', '1e-4', '--pct', '0.05', '--w_off', '1.0', '--seed', '71'],
              emitter_pass2=['-m', 'simcode.emitter.train', '--batch', '16', '--lr', '1.5e-4', '--pct', '0.1', '--w_mr', '2.0', '--w_slope', '50.0',
                      '--seed', '21', '--no_illum', '0', '--pert_model', 'v1'],
              emitter_pass2_long=['-m', 'simcode.emitter.train', '--batch', '16', '--lr', '1e-4', '--pct', '0.05', '--w_mr', '2.0', '--w_slope', '50.0',
                       '--seed', '31', '--no_illum', '0', '--hold_v1', '15'],
              emitter_pass2_final=['-m', 'simcode.emitter.train', '--batch', '16', '--lr', '1e-4', '--pct', '0.05', '--w_mr', '2.0', '--w_slope', '50.0',
                        '--seed', '41', '--no_illum', '0', '--hold_v1', '0'])
EVERY = dict(detector=None, emitter=('2000', '6000'), detector_pass2=('2500', '5000'), detector_pass2_mix=('5000', '10000'), detector_pass2_cont=('5000', '10000'),
             emitter_pass2=('4000', '8000'), emitter_pass2_long=('2000', '10000'), emitter_pass2_final=('5000', '10000'))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCEN_FIT = 'rs05qm'


def presets(ds):
    """in-memory acquisition.PRESETS overrides for a non-DMD setup (pattern k and pitch, camera maps, pixel size); none for the
    DMD-SIMFLUX defaults (the presets ARE that setup). NOTE the step direction: the acquisition simulator draws phi_s = phi0 + s * spacing with
    spacing > 0; a setup that steps the other way needs the measured (signed) spacing in the convention of acquisition.pattern
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
    hold = Tc['heldout']                     # held-out set of the emitter trainer (python -m simcode.emitter.heldout --set 5), or none
    if hold and not os.path.isfile(hold): raise FileNotFoundError(f'train.heldout {hold}: no such file (python -m simcode.emitter.heldout --set 5 --out <file>)')

    cmds = []                                # every command of the recipe, in order (model.json '_recipe', summary 'recipe')

    def sh(step, args, log, **kw):
        cmds.append(dict(step=step, command='python -u ' + ' '.join(str(v) for v in args), log=log, env=kw.get('env') or {}))
        return ds.sh(args, log, **kw)

    def stage(name, extra, need=10):
        d = os.path.join(D, name)
        a = T_BASE[name] + ['--out', d, '--steps', str(steps[name]), '--calib', cal] + extra
        ev = EVERY[name]
        if ev:
            a += ['--eval_every', str(max(int(ev[0]), steps[name] + 1) if quick else ev[0]), '--ckpt_every', str(max(int(ev[1]), steps[name] + 1) if quick else ev[1])]
        xt = Tc['extra'][name if name in ('detector', 'emitter') else 'pass2']
        if xt: a += xt.split()
        a = ['-m', 'simcode.pipeline.run_with_overrides'] + a[1:] if a[0] == '-m' else a
        if os.path.exists(os.path.join(d, 'final.pt')):
            # the arguments the trainer actually ran with are stored in its checkpoint (model.json '_files' -> checkpoint.args)
            cmds.append(dict(step=name, command='python -u ' + ' '.join(str(v) for v in a), skipped='final.pt existed (trained by an earlier invocation)'))
            ds.log(f'train: {name} final.pt exists, skipped'); return os.path.join(d, 'final.pt')
        sh(name, a, os.path.join(D, name + '.log'), gpu=ds.gpu(need), wrap=W)
        return os.path.join(d, 'final.pt')

    from simcode import config as SC
    det_init = Tc['init']['detector'] or (SC.resolve(SC.model_set(Tc['real_mult_from'])['detector']) if quick else None)   # quick: a few steps on top of the template detector
    det = stage('detector', ['--init', det_init] if det_init else [])
    emitter = stage('emitter', ['--detector', det, '--init', Tc['init']['emitter']] + (['--hold', hold] if hold else []))
    det2 = stage('detector_pass2', ['--illum', '1', '--init', det])
    det2_mix = stage('detector_pass2_mix', ['--init', det2, '--pass1_ref', det])
    det2_cont = stage('detector_pass2_cont', ['--init', det2_mix, '--pass1_ref', det])
    det2_25k = os.path.join(D, 'detector_pass2', 'ckpt_25000.pt')
    emitter2 = stage('emitter_pass2', ['--detector', det2_25k if os.path.exists(det2_25k) else det2, '--init', emitter])
    emitter2_long = stage('emitter_pass2_long', ['--detector', det2, '--init', emitter2])
    emitter2_final = stage('emitter_pass2_final', ['--detector', det2, '--init', emitter2_long])
    name = ('q' if quick else 'a') + c['name']
    M = os.path.join(D, 'models'); os.makedirs(M, exist_ok=True)
    files = {}
    for k, f in (('detector', det), ('emitter', emitter), ('detector_pass2', det2_cont), ('emitter_pass2', emitter2_final)):
        files[k] = os.path.join(M, SC.MODEL_FILES[k]); shutil.copyfile(f, files[k])
    # ---- gate scenes, dumps, calibrations
    tag = 'g' + c['name']
    O = os.path.join(D, 'psfmatch'); G = os.path.join(O, 'gate_' + tag); FE = os.path.join(O, 'fe_' + tag)
    seeds = '202,301' if quick else '202,301,302,303'
    scen = 'iso,' + SCEN_FIT if quick else 'iso,pair1,pair2,' + SCEN_FIT
    nt = '400' if quick else '3500'
    if not os.path.exists(os.path.join(G, 'done')):
        sh('gate', ['-m', 'simcode.pipeline.run_with_overrides', 'simcode.benchmark.gate_scenes', '--scen', scen, '--fronts', 'sc', '--emitter', SC.PRETRAIN_EMITTER, '--cal',
               SC.TEMPLATE_CAL_SIM, '--tag', '_hd_cur', '--calib', cal, '--out', G, '--ntarget', nt, '--seeds', seeds],
              os.path.join(D, 'gate.log'), gpu=ds.gpu(8), wrap=W)
        open(os.path.join(G, 'done'), 'w').close()
    if not os.path.exists(os.path.join(FE, 'done')):
        sh('dump', ['-m', 'simcode.pipeline.run_with_overrides', 'simcode.processing.gate_dump', '--scen', scen, '--seeds', seeds, '--ntarget', nt, '--calib', cal, '--gate_dir', G, '--out', FE, '--fronts',
               f's1={files["detector"]}:{files["emitter"]},s2={files["detector_pass2"]}:{files["emitter_pass2"]}'], os.path.join(D, 'dump.log'), gpu=ds.gpu(8), wrap=W)
        open(os.path.join(FE, 'done'), 'w').close()
    os.makedirs(os.path.join(M, 'calib'), exist_ok=True)
    cals = {k: os.path.join(M, SC.MODEL_FILES[k]) for k in ('cal_sim', 'cal_real', 'cal_sim_pass2', 'cal_real_pass2')}
    fcs = os.path.join(REPO, 'analysis', 'psfmatch', 'fit_cal_set.py')
    rm = Tc['real_mult_from']
    from simcode import config as SC
    tmpl = SC.model_set(rm)
    fallback = []
    for fr, base, ks, ksr in (('s1', SC.TEMPLATE_CAL_SIM, 'cal_sim', 'cal_real'), ('s2', SC.TEMPLATE_CAL_SIM_PASS2, 'cal_sim_pass2', 'cal_real_pass2')):
        sh('fitcal_' + fr, ['-m', 'simcode.pipeline.run_with_overrides', fcs, fr, FE, SCEN_FIT, base, cals[ks], cals[ksr], tmpl[ksr]],
              os.path.join(D, f'fitcal_{fr}.log'), gpu=ds.gpu(4), wrap=W, check=not quick)
        if quick and not (os.path.exists(cals[ks]) and os.path.exists(cals[ksr])):
            # quick plumbing: networks after a few steps detect nothing -> the fit has no targets; copy the template set's calibrations
            shutil.copyfile(SC.resolve(tmpl[ks]), cals[ks]); shutil.copyfile(SC.resolve(tmpl[ksr]), cals[ksr]); fallback.append(fr)
            ds.log(f'train (quick): calibration fit of {fr} failed (no matched targets), copied the {rm} calibrations')
    # ---- acceptance (held-out seed 301)
    acc_js = os.path.join(D, 'accept.json')
    sh('accept', ['-m', 'simcode.pipeline.run_with_overrides', os.path.join(REPO, 'simcode', 'pipeline', 'acceptance.py'), FE, cals['cal_sim'], cals['cal_sim_pass2'],
           scen, acc_js], os.path.join(D, 'accept.log'), gpu=ds.gpu(4), wrap=W, check=not quick)
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
    S = {name: dict(_doc=f'simcode.pipeline model of dataset {c["name"]} ({"QUICK plumbing test, not usable" if quick else "PSF-matched recipe"}); '
                         f'PSF {cal}; sig_u not fitted', _gates=gates, psf=cal, **files, **cals, sig_u='')}
    # ---- how the model was made: the recipe (config, commands, steps, code) and the hash of every file; the trainer arguments
    # stored in each checkpoint are part of the file records
    from simcode.util import runconfig as RC
    recs = {k: RC.model_file(v) for k, v in {**files, **cals}.items()}
    trained = {k: RC.file_record(f) for k, f in (('detector', det), ('emitter', emitter), ('detector_pass2', det2), ('detector_pass2_mix', det2_mix), ('detector_pass2_cont', det2_cont), ('emitter_pass2', emitter2),
                                                 ('emitter_pass2_long', emitter2_long), ('emitter_pass2_final', emitter2_final))}
    recipe = dict(what='PSF-matched recipe of simcode.pipeline stage e (docs/AUTO_PIPELINE.md section 6)', acquisition=c['name'], quick=quick,
                  written=RC.now(), config_file=RC.file_record(c['_file']) if c.get('_file') else None, config=c,
                  code=RC.code_record(), environment=RC.env_record(gpu=False), psf_calib=RC.file_record(cal),
                  steps=steps, trainer_base_args=T_BASE, eval_ckpt_every=EVERY, init=dict(Tc['init'], det_used=det_init), extra_args=Tc['extra'],
                  gate=dict(scenes=scen, seeds=seeds, ntarget=nt, fit_scene=SCEN_FIT, heldout_seed=301), real_mult_from=rm, calib_fallback=fallback,
                  overrides=json.load(open(W)), commands=cmds, trained_files=trained, accept=A, gates=gates,
                  not_included='sig_u (selection-aware sigma rescale) is not fitted by this stage')
    S['_recipe'] = recipe; S['_files'] = recs
    json.dump(S, open(os.path.join(ds.dir('train'), 'model.json'), 'w'), indent=1, default=str)
    return dict(recipe=recipe, file_records=recs, model=name, quick=quick, calib_fallback=fallback, steps=steps, dir=D, files=files, calibs=cals, accept=A, gates=gates,
                model_json=os.path.join(ds.dir('train'), 'model.json'), wrap=json.load(open(W)), checks=chk)


def export_model(model_json, dest):
    """copy a model next to the data (e.g. <data>/simcode_results/model/): the network and calibration files, model.json with
    paths relative to dest, the training recipe ('_recipe': config, commands, steps, git commit) and the sha256 of every file
    ('_files'), and a README.md that points to them and to the run record of the processing. -> dest/model.json"""
    import shutil
    from simcode.util import runconfig as RC
    S = json.load(open(model_json)); os.makedirs(os.path.join(dest, 'calib'), exist_ok=True)
    (name, M), = [(k, v) for k, v in S.items() if not k.startswith('_')]
    O = {}; recs = {}
    for k, v in M.items():
        if k.startswith('_') or not isinstance(v, str) or not os.path.isfile(v): O[k] = v; continue
        rel = os.path.join('calib', os.path.basename(v)) if v.endswith('.json') and k != 'psf' else os.path.basename(v)
        shutil.copyfile(v, os.path.join(dest, rel)); O[k] = rel
        recs[k] = dict(RC.model_file(os.path.join(dest, rel)), path=rel, copied_from=os.path.abspath(v))
    out = {name: O, '_recipe': S.get('_recipe', 'not recorded (model made before 2026-11-14: the trainer arguments are in _files.<net>.checkpoint.args)'),
           '_files': recs, '_exported': dict(when=RC.now(), source=os.path.abspath(model_json), code=RC.code_record(dme=False))}
    json.dump(out, open(os.path.join(dest, 'model.json'), 'w'), indent=1, default=str)
    open(os.path.join(dest, 'README.md'), 'w').write(
        f'# SIMCODE model {name}\n\nTrained networks and solve calibrations of this acquisition.\n\n'
        '- `model.json`: the model (format of `simcode/models.json`, paths relative to this folder).\n'
        '  - `_recipe`: how it was trained: the acquisition config, every command in order, steps per trainer, gate scenes, acceptance, git commit of the code.\n'
        '  - `_files`: size and sha256 of every file; for the networks also the trainer arguments stored in the checkpoint.\n'
        '- How a set of localization tables was calculated with this model: the `run_config.json` (readable copy `run_config.yaml`) in the folder of the '
        'tables: fully resolved settings, sha256 of every network / calibration / movie file, code commit, environment. '
        'Re-run: `python -m simcode.run --config run_config.json --out <new dir>`.\n')
    return os.path.join(dest, 'model.json')
