"""v3 retrain: calibration and precision of a new model vs the current one on the held-out v3 sim movies (sims.sh), with the
statistics of analysis/sigma_joint/evalsim.py (rows p >= 0.63 inside the crop, matched one-to-one (1 px, per window) to the true
events ON in the row's window, minus the per-movie median error):
  calibration  z = e / sigma (the run's sx, sy): z_core = truncation-corrected rms on |z| < 4, z_rob = 1.4826 MAD, gross |z| >= 4 %,
               per network-photon bin and per sigma-rank bin u
  precision    robust error sd (1.4826 MAD, nm), rms of |e| < 50 nm, gross > 50 nm %, on the GT windows matched in both runs
  counts       matched rows / rows, and rows with max(sx, sy) <= 16.25 nm
  brightness noise  the c, c_a each run estimated (pattern estimation step 5, log.txt) vs the simulated frame_sd / ang_sd
usage: python eval_sims.py <out json> <model> <run root> <sim> [<sim> ...]   (runs <run root>/<sim>_<cur|new>)"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/simcode2/analysis/sigma_joint'))
import evalsim as ES
SIMS = '/home/jc/simcode2/out/train_v3/eval/sims'
from simcode import table as _SP
if not hasattr(_SP, 'tile_rank'):      # removed from the package with the sigma rescale (2026-11-17); the old definition, for the u bins
    def _tile_rank(x, y, sig, crop=_SP.CROP):
        """per-10-px-tile rank quantile of sigma: (rank + 0.5) / n(tile), stable order"""
        o = np.argsort(sig, kind='stable'); t = _SP.tile_of(x, y, crop)
        u = np.empty(len(sig))
        for tt in np.unique(t):
            it = o[t[o] == tt]
            u[it] = (np.arange(len(it)) + 0.5) / len(it)
        return u
    _SP.tile_rank = _tile_rank

LABELS = ('cur', 'new')


def bnoise(run):
    t = open(os.path.join(run, 'log.txt')).read() if os.path.exists(os.path.join(run, 'log.txt')) else ''
    m = re.findall(r'brightness noise: c ([0-9.]+), c_a ([0-9.]+)', t)
    return [float(v) for v in m[-1]] if m else None


def main():
    outj, model, root, sims = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    res = dict(model=model, sims=sims, passes={}, bnoise={})
    for sim in sims:
        G = np.load(os.path.join(SIMS, sim, 'sim_gt.npz'), allow_pickle=True)
        kn = json.loads(str(G['knobs'])) if 'knobs' in G.files else {}
        res['bnoise'][sim] = dict(true=dict(frame_sd=float(G['frame_sd']), step_sd=float(G['step_sd']), ang_sd=float(kn.get('env', {}).get('SIM_ANG_SD', 'nan'))),
                                  **{lab: bnoise(f'{root}/{sim}_{lab}') for lab in LABELS})
    for ps in ('est', 'est2'):
        D = {lab: [] for lab in LABELS}; common = {lab: [] for lab in LABELS}; per_sim = {}
        for sim in sims:
            GW = ES.gt_windows(*ES.load_gt(os.path.join(SIMS, sim, 'sim_gt.npz')))
            Ls = {lab: ES.load(f'{root}/{sim}_{lab}', ps, GW) for lab in LABELS}
            for lab, L in Ls.items(): D[lab].append(L)
            gs = np.intersect1d(Ls['cur']['gi'], Ls['new']['gi'])
            srt = lambda L, v: v[np.isin(L['gi'], gs)][np.argsort(L['gi'][np.isin(L['gi'], gs)])]
            ph0 = srt(Ls['cur'], Ls['cur']['ph'])
            for lab in LABELS: common[lab].append(dict(e=srt(Ls[lab], Ls[lab]['e']), ph=ph0))
            per_sim[sim] = {lab: dict(matched=len(L['e']), n_base=L['n_base'], cut16=L['cut16'], all=ES.zst(L['e'] / L['s'])) for lab, L in Ls.items()}
            print(f'# {ps} {sim}: ' + ', '.join(f'{lab} matched {len(L["e"])} / rows {L["n_base"]}, <=16.25 nm {L["cut16"]}, z_core {ES.zst(L["e"] / L["s"])["z_core"]:.2f}'
                                                for lab, L in Ls.items()), flush=True)
        R = dict(calib={}, precision={}, per_sim=per_sim)
        for lab, Ls in D.items():
            e = np.concatenate([L['e'] for L in Ls]); s = np.concatenate([L['s'] for L in Ls])
            ph = np.concatenate([L['ph'] for L in Ls]); u = np.concatenate([L['u'] for L in Ls]); z = e / s
            R['calib'][lab] = dict(all=ES.zst(z), cut16=int(sum(L['cut16'] for L in Ls)), n_base=int(sum(L['n_base'] for L in Ls)),
                                   matched=int(sum(len(L['e']) for L in Ls)),
                                   photons=[dict(lo=lo, hi=hi, sigma_med_nm=float(np.median(s[(ph >= lo) & (ph < hi)]) * ES.PX), **ES.zst(z[(ph >= lo) & (ph < hi)]))
                                            for lo, hi in zip(ES.PH[:-1], ES.PH[1:]) if ((ph >= lo) & (ph < hi)).sum() >= 200],
                                   u=[dict(lo=lo, hi=hi, **ES.zst(z[(u >= lo) & (u < hi)])) for lo, hi in zip(ES.UE[:-1], ES.UE[1:])])
        for lab in LABELS:
            e = np.concatenate([c['e'] for c in common[lab]]); ph = np.concatenate([c['ph'] for c in common[lab]])
            R['precision'][lab] = [dict(lo=lo, hi=hi, **ES.prec(e[(ph >= lo) & (ph < hi)])) for lo, hi in zip(ES.PH[:-1], ES.PH[1:]) if ((ph >= lo) & (ph < hi)).sum() >= 200]
            R['precision'][lab + '_all'] = ES.prec(e)
        res['passes'][ps] = R
        for lab, C in R['calib'].items():
            print(f'{model} {ps} {lab:4s} z_core ' + ' '.join(f"{b['lo']:g}:{b['z_core']:.2f}" for b in C['photons']) + ' | u ' +
                  ' '.join(f"{b['z_core']:.2f}" for b in C['u']) + f" | all {C['all']['z_core']:.2f} (rob {C['all']['z_rob']:.2f}, gross {C['all']['gross_pct']:.2f} %)"
                  f" | matched {C['matched']} / {C['n_base']}, <=16.25 nm {C['cut16']}", flush=True)
        for lab in LABELS:
            P = R['precision'][lab]
            print(f'{model} {ps} {lab:4s} sd_rob nm ' + ' '.join(f"{b['lo']:g}:{b['sd_rob']:.2f}" for b in P) + ' | rms<50 ' +
                  ' '.join(f"{b['rms50']:.2f}" for b in P) + ' | gross% ' + ' '.join(f"{b['gross50_pct']:.2f}" for b in P) +
                  f" | all sd_rob {R['precision'][lab + '_all']['sd_rob']:.2f}", flush=True)
    for sim, b in res['bnoise'].items(): print('brightness noise', sim, json.dumps(b), flush=True)
    json.dump(res, open(outj, 'w'), indent=1)


if __name__ == '__main__':
    main()
