"""z3d: a retrained model (3D head) takes over its paper name, after its gate comparison passed (analysis/z3d/accept_z.py).
  1. archive: the current model of that name (simcode/models.json) is exported as a model folder <models>/archive_2d/<name>/
     (files, model.json, sha256; loadable with --model <folder>; 2D heads give z = NaN); no config references it
  2. the new model: exported from the staged files (<eval dir>/model_files, out/z3d/runs/eval.sh) as <models>/<name>/
  3. simcode/models.json: the entry <name> points to the files of <models>/<name>/ (relative to the models directory); its _doc
     says what changed. The top-level 2D files stay in <models> until this branch is merged (main still references them).
usage: python analysis/z3d/swap_model.py <name> <eval dir> [--dry]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simcode import config
from simcode.auto.train import export_model as _export


def export_model(src, dest):
    """auto.train.export_model + the sig_u files (a comma pair, which export_model leaves as absolute paths) copied to
    dest/calib/ with relative names in dest/model.json"""
    import shutil
    f = _export(src, dest); D = json.load(open(f)); (nm, M), = [(k, v) for k, v in D.items() if not k.startswith('_')]
    rel = []
    for s in M['sig_u'].split(','):
        shutil.copyfile(s, os.path.join(dest, 'calib', os.path.basename(s))); rel.append(os.path.join('calib', os.path.basename(s)))
    M['sig_u'] = ','.join(rel); json.dump(D, open(f, 'w'), indent=1, default=str)
    return f

KEYS = ('det', 'head', 'cal_sim', 'cal_real', 'p2_det', 'p2_head', 'p2_cal_sim', 'p2_cal_real')


def main():
    name, E = sys.argv[1], sys.argv[2]; dry = '--dry' in sys.argv
    MD = config.models_dir(); MF = os.path.join(E, 'model_files')
    S = json.load(open(config.MODELS_FILE)); old = S[name]
    if old.get('head', '').startswith(name + '/'): raise SystemExit(f'{name}: already swapped ({old["head"]})')
    p1 = 'm1809' if name == 'm1809g' else name
    ab = lambda f: os.path.join(MD, f)
    # 1. archive of the current (2D) model
    a = {k: ab(v) for k, v in old.items() if k in KEYS + ('psf',)}
    a['sig_u'] = ','.join(ab(s) for s in old['sig_u'].split(','))
    srcA = os.path.join(E, 'model_archive_src.json')
    # 2. the new model from the staged files
    n = dict(psf=os.path.join(MF, os.path.basename(old['psf'])), det=os.path.join(MF, os.path.basename(old['det'])),
             head=os.path.join(MF, f'hd_{p1}.pt'), cal_sim=os.path.join(MF, f'calib_sc_hd_{p1}.json'), cal_real=os.path.join(MF, f'calib_sc_hd_{p1}_real.json'),
             p2_det=os.path.join(MF, os.path.basename(old['p2_det'])), p2_head=os.path.join(MF, f'p2headL_{name}.pt'),
             p2_cal_sim=os.path.join(MF, f'calib_p2L_{name}.json'), p2_cal_real=os.path.join(MF, f'calib_p2L_{name}_real.json'),
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
    print('models.json', name, '->', S[name]['head'])


if __name__ == '__main__':
    main()
