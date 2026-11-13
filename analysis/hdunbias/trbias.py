"""Diagnostic: first-harmonic prior bias of a pass-1 head on TRAINING batches (the sc4 trainer's generator, chain2 stage-hd
recipe options, det_m1809 candidates p >= 0.5 matched to GT < 1 px), per photon bin and nn > 2 px vs all; also the
z-normalised first harmonic |c1| (what the --unb_w penalty sees). usage: python analysis/hdunbias/trbias.py <head.pt> [nbatch]"""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/pixbias')
from sim import bias_curve
from simcode import config
from simcode.hd import sc4, data as D
from simcode.hd.sc import load_det_front, PSFCol, patches
from simcode.det.model import select_subpixel
from simcode.util import proctitle
proctitle.set_title('hd-unbias', 'trbias')
hp = sys.argv[1]; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 150
dev = 'cuda'; torch.manual_seed(5)
det, _ = load_det_front('models/det_m1809.pt')
gen = D.Gen(64, 64, 6, seed=77, calib=[config.resolve('models/insitu/calib_1809_2_psf3d_avg.json')])
col = PSFCol(gen.rs['0.5nM'].sim)
head = sc4.SCHead4(6, 96, P=sc4.PP).to(dev); head.load_state_dict(torch.load(hp, map_location=dev, weights_only=False)['model']); head.eval()
X = {k: [] for k in ('ex', 'ey', 'tx', 'ty', 'sx', 'sy', 'ph', 'nn', 'dx', 'dy', 'w')}
with torch.no_grad():
    for it in range(nb):
        o = gen.batch(16, dens=(0.0005, 0.04), med_photons=(40, 2500), pair_frac=0.25, pair_curr=True, paper=True, tau_r=(15, 80),
                      bright_tail=(0.1, 2000, 20000))
        fr = o['frames']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            dd = det(fr)
        lgm = dd['logit'].float(); lgm = lgm[:, 0] if lgm.dim() == 4 else lgm
        c = select_subpixel(lgm, dd['dxy'].float(), 640, p_min=sc4.NBR_PMIN, radius=0.7)
        kp = torch.isfinite(c['logit'])
        nbr = sc4.nbr_map(lgm, *(c[k][kp] for k in ('b', 'iy', 'ix', 'x', 'y', 'logit')))
        c = dict(c, logit=torch.where(c['logit'] >= 0, c['logit'], torch.full_like(c['logit'], -float('inf'))))
        m = sc4.match_gt(o, c)
        if m is None: continue
        off = torch.stack([m['x'] - m['ix'].float(), m['y'] - m['iy'].float()], 1)
        pt = patches(fr, m['b'], m['iy'], m['ix'])
        nbp = sc4.nbr_patches(nbr, m['b'], m['iy'], m['ix'], m['x'], m['y'], torch.sigmoid(lgm[m['b'], m['iy'], m['ix']]))
        out = sc4.head_call(head, pt, off, col(off), nbp)
        G = o['gt']; tx, ty = G['x'][m['g']], G['y'][m['g']]
        fov = G['fov'][m['g']]
        X['w'].append(sc4.nn_weights(fov, m['nn']).cpu())
        X['ex'].append((m['x'] + out['dxy'][:, 0] - tx).cpu()); X['ey'].append((m['y'] + out['dxy'][:, 1] - ty).cpu())
        X['dx'].append((m['x'] - tx).cpu()); X['dy'].append((m['y'] - ty).cpu())
        X['tx'].append(tx.cpu()); X['ty'].append(ty.cpu()); X['sx'].append(out['lsxy'][:, 0].exp().cpu()); X['sy'].append(out['lsxy'][:, 1].exp().cpu())
        X['ph'].append(G['phot'][m['g']].sum(1).cpu()); X['nn'].append(m['nn'].cpu())
X = {k: torch.cat(v).double().numpy() for k, v in X.items()}
sub = lambda t: t - np.round(t)
R = {}
for sel in ('all', 'nn>2'):
    for lo, hi in ((0, 500), (500, 2000), (2000, 1e12), (0, 1e12)):
        k = (X['ph'] >= lo) & (X['ph'] < hi) & ((X['nn'] > 2) if sel == 'nn>2' else True)
        r = dict(n=int(k.sum()))
        for en, (a, b) in (('det', ('dx', 'dy')), ('prior', ('ex', 'ey'))):
            A = [bias_curve(X[e][k], sub(X[t][k]))['A_nm'] for e, t in ((a, 'tx'), (b, 'ty'))]
            r[en + '_A_nm'] = round(float(np.mean(A)), 2)
        c1 = []
        for e, t, s in (('ex', 'tx', 'sx'), ('ey', 'ty', 'sy')):
            z = -X[e][k] / X[s][k]; w = X['w'][k]
            c1.append(abs((w * z * np.exp(2j * np.pi * sub(X[t][k]))).sum() / w.sum()))
        r['zc1'] = round(float(np.mean(c1)), 4)
        R[f'{sel} {lo}-{hi:.0f}'] = r
for k, v in R.items(): print(k, v)
