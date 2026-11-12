"""Prior-bias test of the 4Pi head(s): isolated emitters (patch simulator), 2-8k e, the network's pattern-free position
prior vs the true offset, with the training-like pattern phase (+-0.3 rad) and with an arbitrary one (uniform over 2 pi).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/prior_slope4pi.py v2 v3"""
import sys, json, numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, REPO); sys.path.insert(0, f'{REPO}/analysis/fourpi')
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.head4pi import Head4Pi
from eval_head4pi import run_net
out = {}
for tag in sys.argv[1:]:
    from simcode2.fourpi.loadhead import load_head
    net, ck = load_head(tag)
    lut = LUT4Pi(ck['psf'], P=15)
    for er, nm in ((0.3, 'training-like phase'), (np.pi, 'phase uniform over 2 pi')):
        S = Sim4Pi(lut, seed=21).sample(12000, p_nb=0.0, p_on=1.0, eps_range=er); o = run_net(net, S); N = S['Nt'].sum(1)
        s = ((N > 2000) & (N < 8000)).cpu().numpy()
        r = {}
        for ax, k in ((0, 'x'), (1, 'y')):
            t = S[k].cpu().numpy()[s]; e = (o['dxy'][:, ax] - S[k]).cpu().numpy()[s] * 128
            r[k] = dict(slope=float(np.polyfit(t * 128, e, 1)[0]), median=float(np.median(e)), robust_sd=float(1.4826 * np.median(np.abs(e - np.median(e)))))
        out[f'{tag} {nm}'] = r
        print(f"{tag:3s} {nm:24s}: slope x/y {r['x']['slope']:+.3f}/{r['y']['slope']:+.3f}, median {r['x']['median']:+.1f}/{r['y']['median']:+.1f} nm, robust sd {r['x']['robust_sd']:.1f}/{r['y']['robust_sd']:.1f} nm", flush=True)
json.dump(out, open(f'{REPO}/out/4pi/sim/prior_slope.json', 'w'), indent=1)
