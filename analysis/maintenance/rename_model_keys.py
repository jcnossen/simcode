"""One-off migration (2026-11-15 reorganisation): the file keys of a model definition det / head / p2_det / p2_head /
p2_cal_sim / p2_cal_real -> detector / emitter / detector_pass2 / emitter_pass2 / cal_sim_pass2 / cal_real_pass2, in
simcode/models.json and in model folders (model.json: the model entry and '_files'; the training record '_recipe' is a record
of what was run and stays as written). The package reads only the new keys.

  python analysis/maintenance/rename_model_keys.py <models.json | model.json | directory (all **/model.json below)> ...
  --check: only report files that still have old keys (exit code 1 if any)
"""
import glob, json, os, sys

KEYS = dict(det='detector', head='emitter', p2_det='detector_pass2', p2_head='emitter_pass2',
            p2_cal_sim='cal_sim_pass2', p2_cal_real='cal_real_pass2')


def ren(d):
    return {KEYS.get(k, k): v for k, v in d.items()}


def convert(S):
    out = {}
    for k, v in S.items():
        if k.startswith('_'):
            out[k] = ren(v) if k == '_files' and isinstance(v, dict) else v
        else:
            out[k] = ren(v) if isinstance(v, dict) else v
    return out


def files_of(args):
    for a in args:
        if os.path.isdir(a): yield from sorted(glob.glob(os.path.join(a, '**', 'model.json'), recursive=True))
        else: yield a


def main(argv):
    check = '--check' in argv
    old = 0
    for f in files_of([a for a in argv if a != '--check']):
        S = json.load(open(f)); C = convert(S)
        if C == S: print('ok     ', f); continue
        old += 1
        if check: print('OLD    ', f); continue
        json.dump(C, open(f + '.tmp', 'w'), indent=1); os.replace(f + '.tmp', f)
        print('renamed', f)
    return 1 if check and old else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
