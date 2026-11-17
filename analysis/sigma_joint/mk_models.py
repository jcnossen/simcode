"""joint prior / intensity covariance (2026-11-17): model folders of the fine-tuned emitter networks for the evaluation runs:
the model's detectors and PSF, the fine-tuned emitter networks (out/sigma_joint/train/<m>/p1, p2: final.pt unless given), and the
model's solve calibrations without the removed scale / sigma factors (kap_bins, kap_p, kap_I2, sig_a, sig_g, sig_ph and their
records); no sigma rescale. The calibration files written here are also the package's new calibration files (--write_repo).
usage: python mk_models.py <out dir> <label> [<m>:<p1 ckpt>:<p2 ckpt> ...] [--write_repo <repo>]"""
import json, os, subprocess, sys

REPO_MAIN = os.path.expanduser('~/simcode2')
T = os.path.expanduser('~/simcode2/out/sigma_joint/train')
DROP = ('sig_a', 'sig_g', 'sig_ph', 'sig_fit_with', 'kap_p', 'kap_I2', 'kap_p_raw', 'kap_I2_raw', 'kap_bins')
BRANCH = dict(m1809='paper', m25='paper', g13='master')


def git_json(branch, path):
    return json.loads(subprocess.check_output(['git', '-C', REPO_MAIN, 'show', f'{branch}:{path}']))


def main():
    argv = sys.argv[1:]
    repo = None
    if '--write_repo' in argv:
        i = argv.index('--write_repo'); repo = argv[i + 1]; del argv[i:i + 2]
    out, label = argv[:2]
    ck = {s.split(':')[0]: s.split(':')[1:] for s in argv[2:]}
    for m, br in BRANCH.items():
        M = git_json(br, f'models/{m}/model.json')[m]
        D = os.path.join(out, f'{m}_{label}'); os.makedirs(os.path.join(D, 'calib'), exist_ok=True)
        defn = {k: os.path.join(REPO_MAIN, 'models', m, M[k]) for k in ('psf', 'detector', 'detector_pass2')}
        p1, p2 = ck.get(m, ['final', 'final'])
        defn['emitter'] = f'{T}/{m}/p1/{p1}.pt'; defn['emitter_pass2'] = f'{T}/{m}/p2/{p2}.pt'
        for k in ('cal_sim', 'cal_sim_pass2'):
            c = {kk: v for kk, v in git_json(br, f'models/{m}/{M[k]}').items() if kk not in DROP}
            c['note_2026_11_17'] = ('the per-photon-bin solve scales kap_p / kap_I2 and the sigma factors sig_a / sig_g / sig_ph were removed: the '
                                    'emitter networks give one joint prior / intensity covariance (simcode/emitter/model.EmitterNet) and the solve '
                                    'reports the posterior sd (analysis/sigma_joint/)')
            json.dump(c, open(os.path.join(D, M[k]), 'w'), indent=1)
            defn[k] = M[k]
            if repo and os.path.isdir(os.path.join(repo, 'models', m)):
                json.dump(c, open(os.path.join(repo, 'models', m, M[k]), 'w'), indent=1)
        defn['sig_u'] = ''
        json.dump({f'{m}_{label}': defn}, open(os.path.join(D, 'model.json'), 'w'), indent=1)
        print('wrote', D, defn['emitter'], defn['emitter_pass2'])


if __name__ == '__main__':
    main()
