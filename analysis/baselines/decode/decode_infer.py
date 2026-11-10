"""Run a DECODE model on a hd_compare sim movie.
usage: CUDA_VISIBLE_DEVICES=1 python decode_infer.py <movie dir | name> <model> <mode> <out.npz> [maxframes]
  model: 'paper:<name>' (scratchpad copy of the paper model, e.g. paper:decode-1f) or a trained LutPSF model name
  mode : frame  (per-frame DECODE, 3-frame context)
         sum6   (non-overlapping 6-frame sums starting at frame 0 and frame 3 -> window starts w % 3 == 0, as the paper's
                 decode-summed; two passes so that every W3 window of the protocol is covered)
         sum3   (non-overlapping 3-frame sums, the paper's 2.5 nM 'decode-summed3')
Output npz: frame (frame / window start), x, y, sx, sy, phot, sphot, bg, prob; timing (s)."""
import sys, os, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as D

nm, model, mode, out = sys.argv[1:5]
maxframes = int(sys.argv[5]) if len(sys.argv) > 5 else None
tif = (nm.rstrip('/') + '/sim_movie.tif') if os.path.isdir(nm) else f'{D.SP}/sb_{nm}/sim_movie.tif'   # movie dir, or a name under the out root
nm = os.path.basename(nm.rstrip('/'))
t0 = time.time()
if model.startswith('paper:'):
    inst = D.load_paper(model.split(':', 1)[1])
else:
    mf = f'{D.SPD}/lutpsf/{model}/model_final.pt'      # = the last epoch (verified against the final checkpoint)
    inst = D.load_trained(model, 'frame' if mode == 'frame' else 'sum6', model_file=mf if os.path.exists(mf) else None)
t_load = time.time() - t0
torch.cuda.synchronize(); t1 = time.time()
if mode == 'frame':
    parts = [D.infer_movie(inst, tif, 1, 0, maxframes)]
elif mode == 'sum6':
    parts = [D.infer_movie(inst, tif, 6, s, maxframes) for s in (0, 3)]
elif mode == 'sum3':
    parts = [D.infer_movie(inst, tif, 3, 0, maxframes)]
torch.cuda.synchronize(); t_inf = time.time() - t1
R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
print(nm, model, mode, 'n', len(R['frame']), 'load s', round(t_load, 1), 'infer s', round(t_inf, 1), flush=True)
np.savez_compressed(out, **R, timing=json.dumps(dict(load_s=t_load, infer_s=t_inf, model=model, mode=mode)))
print('saved', out)
