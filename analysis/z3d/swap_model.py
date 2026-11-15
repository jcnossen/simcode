"""z3d: a retrained model (3D head) takes over its paper name, after its gate comparison passed (analysis/z3d/accept_z.py).
  1. archive: the current model of that name (simcode/models.json) is exported as a model folder <models>/archive_2d/<name>/
     (files, model.json, sha256; loadable with --model <folder>; 2D heads give z = NaN); no config references it
  2. the new model: exported from the staged files (<eval dir>/model_files, out/z3d/runs/eval.sh) as <models>/<name>/
  3. simcode/models.json: the entry <name> points to the files of <models>/<name>/ (relative to the models directory); its _doc
     says what changed. The top-level 2D files stay in <models> until this branch is merged (main still references them).
  --generic (2026-11-14, all pass-2 models generic): the model <name> already is a folder <models>/<name>/ (3D, dedicated pass 2);
     that folder is moved to <models>/archive_p2dedicated/<name>/ (its model.json has relative paths: stays loadable), the new
     model (same pass 1, generic pass 2) is exported from <eval dir>/model_files_<name>/ (out/z3d/runs/eval.sh <name>g).
usage: python analysis/z3d/swap_model.py <name> <eval dir> [--generic] [--dry]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simcode import config
from simcode.pipeline.training import export_model as _export


def export_model(src, dest):
    """pipeline.training.export_model + the sig_u files (a comma pair, which export_model leaves as absolute paths) copied to
    dest/calib/ with relative names in dest/model.json"""
    import shutil
    f = _export(src, dest); D = json.load(open(f)); (nm, M), = [(k, v) for k, v in D.items() if not k.startswith('_')]
    rel = []
    for s in M['sig_u'].split(','):
        shutil.copyfile(s, os.path.join(dest, 'calib', os.path.basename(s))); rel.append(os.path.join('calib', os.path.basename(s)))
    M['sig_u'] = ','.join(rel); json.dump(D, open(f, 'w'), indent=1, default=str)
    return f

KEYS = ('detector', 'emitter', 'cal_sim', 'cal_real', 'detector_pass2', 'emitter_pass2', 'cal_sim_pass2', 'cal_real_pass2')


def main():
    name, E = sys.argv[1], sys.argv[2]; dry = '--dry' in sys.argv
    if '--generic' in sys.argv: return generic(name, E, dry)
    MD = config.models_dir(); MF = os.path.join(E, 'model_files')
    S = json.load(open(config.MODELS_FILE)); old = S[name]
    if old.get('emitter', '').startswith(name + '/'): raise SystemExit(f'{name}: already swapped ({old["emitter"]})')
    p1 = name
    ab = lambda f: os.path.join(MD, f)
    # 1. archive of the current (2D) model
    a = {k: ab(v) for k, v in old.items() if k in KEYS + ('psf',)}
    a['sig_u'] = ','.join(ab(s) for s in old['sig_u'].split(','))
    srcA = os.path.join(E, 'model_archive_src.json')
    # 2. the new model from the staged files
    n = dict(psf=os.path.join(MF, os.path.basename(old['psf'])), detector=os.path.join(MF, os.path.basename(old['detector'])),
             emitter=os.path.join(MF, f'hd_{p1}.pt'), cal_sim=os.path.join(MF, f'calib_sc_hd_{p1}.json'), cal_real=os.path.join(MF, f'calib_sc_hd_{p1}_real.json'),
             detector_pass2=os.path.join(MF, os.path.basename(old['detector_pass2'])), emitter_pass2=os.path.join(MF, f'p2headL_{name}.pt'),
             cal_sim_pass2=os.path.join(MF, f'calib_p2L_{name}.json'), cal_real_pass2=os.path.join(MF, f'calib_p2L_{name}_real.json'),
             sig_u=os.path.join(MF, f'sigu_p1_{p1}.json') + ',' + os.path.join(MF, f'sigu_p2_{name}.json'))
    miss = [v for k, v in list(n.items()) + list(a.items()) for v in v.split(',') if not os.path.isfile(v)]
    if miss: raise SystemExit(f'missing files: {miss}')
    rec = dict(trainer_chain=f'/home/jc/simcode2/out/z3d/runs/chain.sh {name} (logs out/z3d/runs/{name}/)',
               evaluation=f'/home/jc/simcode2/out/z3d/runs/eval.sh {name} (gate comparison {E}/accept_p*.txt)',
               note='head with the 3D position prior (x, y, z + 3 x 3 covariance; z in px relative to the PSF calibration focal plane); '
                    'detectors of the previous model; same steps / schedules / seeds as the previous model')
    srcN = os.path.join(E, 'model_src.json')
    if dry:
        print(json.dumps(dict(archive=a, new=n), indent=1)); return
    json.dump({name: a, '_recipe': 'the 2D paper model (before 2026-11-14), archived'}, open(srcA, 'w'), indent=1)
    json.dump({name: n, '_recipe': rec}, open(srcN, 'w'), indent=1)
    print(export_model(srcA, os.path.join(MD, 'archive_2d', name)))
    fN = export_model(srcN, os.path.join(MD, name))
    M = json.load(open(fN))[name]
    ent = {k: (os.path.join(name, v) if isinstance(v, str) and v and k != 'sig_u' else v) for k, v in M.items()}
    ent['sig_u'] = ','.join(os.path.join(name, s) for s in M['sig_u'].split(','))
    ent['_doc'] = (old.get('_doc', '') + f' | 2026-11-14: retrained with the 3D head (z output; out/z3d), files in {name}/; '
                   f'the previous 2D model is the folder archive_2d/{name}/ (not referenced by any config)')
    S[name] = {k: ent[k] for k in ['_doc'] + [k for k in ent if k != '_doc']}
    json.dump(S, open(config.MODELS_FILE, 'w'), indent=1); open(config.MODELS_FILE, 'a').write('\n')
    print('models.json', name, '->', S[name]['emitter'])


def generic(name, E, dry):
    import shutil
    MD = config.models_dir(); MF = os.path.join(E, f'model_files_{name}')
    S = json.load(open(config.MODELS_FILE)); old = S[name]
    if not old.get('emitter', '').startswith(name + '/'): raise SystemExit(f'{name}: not a model folder ({old["emitter"]}); run the 3D swap first')
    cur = os.path.join(MD, name); arc = os.path.join(MD, 'archive_p2dedicated', name)
    if os.path.exists(arc): raise SystemExit(f'{arc} exists: already swapped?')
    C = json.load(open(os.path.join(cur, 'model.json')))[name]
    if C['detector_pass2'] != f'p2det_{name}.pt': raise SystemExit(f'unexpected layout of {cur}/model.json')
    n = dict(psf=os.path.join(MF, os.path.basename(C['psf'])), detector=os.path.join(MF, f'det_{name}.pt'), emitter=os.path.join(MF, f'hd_{name}.pt'),
             cal_sim=os.path.join(MF, f'calib_sc_hd_{name}.json'), cal_real=os.path.join(MF, f'calib_sc_hd_{name}_real.json'),
             detector_pass2=os.path.join(MF, f'p2det_{name}.pt'), emitter_pass2=os.path.join(MF, f'p2headL_{name}.pt'),
             cal_sim_pass2=os.path.join(MF, f'calib_p2L_{name}.json'), cal_real_pass2=os.path.join(MF, f'calib_p2L_{name}_real.json'),
             sig_u=os.path.join(MF, f'sigu_p1_{name}.json') + ',' + os.path.join(MF, f'sigu_p2_{name}.json'))
    miss = [v for v in n.values() for v in v.split(',') if not os.path.isfile(v)]
    if miss: raise SystemExit(f'missing files: {miss}')
    # pass 1 must be the current model's pass 1 (same files)
    from simcode.util import runconfig as RC
    F = json.load(open(os.path.join(cur, 'model.json')))['_files']
    for k in ('detector', 'emitter', 'cal_sim', 'cal_real', 'psf'):
        if RC.model_file(n[k])['sha256'] != F[k]['sha256']: raise SystemExit(f'{k}: staged file differs from the current pass 1')
    chain = (f'/home/jc/simcode2/out/z3d/runs/chain.sh m1809g (logs out/z3d/runs/m1809g/; pass-2 detector = out/p2generic g1809 '
             f'+ dg2s: analysis/p2generic/chain_gen.sh g1809, cont_det.sh dg2s)' if name == 'm1809' else
             f'/home/jc/simcode2/out/z3d/runs/chain_gen.sh {name} (logs out/z3d/runs/{name}g/)')
    rec = dict(pass1=json.load(open(os.path.join(cur, 'model.json')))['_recipe'], pass2_trainer_chain=chain,
               evaluation=f'/home/jc/simcode2/out/z3d/runs/eval.sh {name}g (vs the dedicated pass 2: {E}/accept_p2_ded.txt)',
               note='pass 1 = the 3D model (unchanged files); pass 2 = GENERIC (2026-11-14: all pass-2 models generic): generic '
                    'illumination-pattern draw (simulate/acquisition.AcquisitionSim._generic), p2det 40k p_std 0.25 -> p2det_c 40k p_std 0.5; heads '
                    'p2head 24k / p2headL 50k p_std 0.25 -> p2headL2 50k p_std 0.5 (models/README.md)')
    if dry:
        print(json.dumps(dict(archive=arc, new=n), indent=1)); return
    srcN = os.path.join(E, f'model_src_{name}.json')
    json.dump({name: n, '_recipe': rec}, open(srcN, 'w'), indent=1)
    os.makedirs(os.path.dirname(arc), exist_ok=True); shutil.move(cur, arc)
    A = json.load(open(os.path.join(arc, 'model.json')))
    A['_archived'] = f'2026-11-14: the dedicated pass 2 (SIMCODE patterns only) of {name}, replaced by the generic pass 2 ({MD}/{name}/)'
    json.dump(A, open(os.path.join(arc, 'model.json'), 'w'), indent=1, default=str)
    fN = export_model(srcN, cur)
    M = json.load(open(fN))[name]
    ent = {k: (os.path.join(name, v) if isinstance(v, str) and v and k != 'sig_u' else v) for k, v in M.items()}
    ent['sig_u'] = ','.join(os.path.join(name, s) for s in M['sig_u'].split(','))
    ent['_doc'] = (old.get('_doc', '') + ' | 2026-11-14: GENERIC pass 2 (generic illumination-pattern training; accepts any measured '
                   f'two-orientation pattern), pass 1 unchanged; the dedicated pass 2 is the folder archive_p2dedicated/{name}/ '
                   '(not referenced by any config)')
    S[name] = {k: ent[k] for k in ['_doc'] + [k for k in ent if k != '_doc']}
    json.dump(S, open(config.MODELS_FILE, 'w'), indent=1); open(config.MODELS_FILE, 'a').write('\n')
    print('models.json', name, '->', S[name]['emitter_pass2'], '| archive', arc)


if __name__ == '__main__':
    main()
