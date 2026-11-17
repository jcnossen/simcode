"""sigma layer (2026-11-17): model folders for the training-data runs. Each = the model's networks (the .pt files of the paper
checkout's models/<m>/) and its solve calibrations of pass 1 / pass 2 (git: master for g13, paper for m1809 / m25) WITHOUT the
removed sigma factors (sig_a, sig_g, sig_ph, the unused scalar kap / sig_fit_with keys) and with an IDENTITY sigma layer
(sigma = sqrt(s_pm^2 + spread_pm^2)); no sigma rescale. The solve weights (kap_bins, prior_off, cmult, box, mix_T) are unchanged,
so x, y, z, photons are those of the model; the runs store the layer's raw inputs, from which the fit recomputes any sigma.
usage: python mk_models.py <out dir> [<label>]   (label 'id': identity layer; 'k1': identity layer and the solve weights kap_p = kap_I2
       = 1 in every photon bin (the kap = 1 precision check); the model folders <out>/<m>_<label>/)"""
import json, os, subprocess, sys
from simcode.solve import sigma_layer as SL

REPO_MAIN = os.path.expanduser('~/simcode2')            # the network weights (not in git)
DROP = ('sig_a', 'sig_g', 'sig_ph', 'sig_fit_with', 'kap_p', 'kap_I2', 'kap_p_raw', 'kap_I2_raw')
BRANCH = dict(m1809='paper', m25='paper', g13='master')


def git_json(branch, path):
    return json.loads(subprocess.check_output(['git', '-C', REPO_MAIN, 'show', f'{branch}:{path}']))


def strip(cal):
    return {k: v for k, v in cal.items() if k not in DROP}


def main():
    out = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else 'id'
    for m, br in BRANCH.items():
        M = git_json(br, f'models/{m}/model.json')[m]
        D = os.path.join(out, f'{m}_{label}'); os.makedirs(os.path.join(D, 'calib'), exist_ok=True)
        defn = {}
        for k in ('psf', 'detector', 'emitter', 'detector_pass2', 'emitter_pass2'):
            defn[k] = os.path.join(REPO_MAIN, 'models', m, M[k])
        for k in ('cal_sim', 'cal_sim_pass2'):
            c = strip(git_json(br, f'models/{m}/{M[k]}'))
            c['sigma_layer'] = SL.SigmaLayer().to_dict(note='identity (training-data runs)')
            if label == 'k1':
                b = c['kap_bins']; c['kap_bins'] = dict(ph=b['ph'], kap_p=[1.0] * len(b['ph']), kap_I2=[1.0] * len(b['ph']))
            c['note_sigma_layer'] = f'analysis/sigma_layer/mk_models.py: {br}:models/{m}/{M[k]} without {DROP}, identity sigma layer'
            json.dump(c, open(os.path.join(D, M[k]), 'w'), indent=1)
            defn[k] = M[k]
        defn['sig_u'] = ''
        json.dump({f'{m}_{label}': defn}, open(os.path.join(D, 'model.json'), 'w'), indent=1)
        print('wrote', D)


if __name__ == '__main__':
    main()
