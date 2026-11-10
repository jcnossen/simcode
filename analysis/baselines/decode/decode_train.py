"""Train a DECODE model matched to the hd_compare sim movies (see dec_common.py for the simulation settings).
usage: CUDA_VISIBLE_DEVICES=1 python decode_train.py <variant frame|sum6> <model_name> [epochs] [max_hours] [lr (resume phase)]
Models / logs / checkpoints: <scratchpad>/decode_hd/lutpsf/<model_name>/ (param_train.yaml = the training parameters)."""
import sys, time, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as D
from decode.utils import param_io

variant, name = sys.argv[1], sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 400
max_h = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
lr_over = float(sys.argv[5]) if len(sys.argv) > 5 else None   # resume phase: fixed lower lr (cosine-free anneal step)
torch.manual_seed(11); np.random.seed(11)
p = D.make_param(variant, epochs=epochs)
inst = D.LutDecodeInstance(p, name, variant)
param_io.save_params(f'{inst.model_dir}/param_train.yaml', inst.param)
print(inst.param.Scaling, flush=True)
inst.setup_trainer()
if os.path.exists(inst.chkpt_path):
    try:
        inst.load_checkpoint(); print('resumed from checkpoint', flush=True)
        if lr_over is not None:
            inst.set_lr(lr_over); inst.lr_scheduler = torch.optim.lr_scheduler.StepLR(inst.optimizer, step_size=10 ** 6); print('lr set to', lr_over, flush=True)
    except Exception as e:
        print('no resume:', e, flush=True)
t0 = time.time()
for i, loss in enumerate(inst.training_iterator(use_conv_check=lr_over is None)):
    el = (time.time() - t0) / 3600
    print(f'epoch {i} train_loss {float(loss):.4f} lr {inst.optimizer.param_groups[0]["lr"]:.2e} elapsed_h {el:.2f}', flush=True)
    if el > max_h:
        print('time limit reached', flush=True); break
print('done', flush=True)
