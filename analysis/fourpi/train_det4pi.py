"""Train det4pi: the det_n1 detector (simcode/detector/model.Detector, up = 1) on 4Pi scenes.

Input: the registered 4-channel sum of the 6 frames (6 planes, photons) = det_n1's input format, from the full-scene
4Pi simulator (simcode/fourpi/acquisition4pi.py: dense scenes 0.01-2 /um^2, filaments / rings / close pairs, kinetics,
pattern with arbitrary phase, PSF jitter bank, real registration). Targets, loss and nn weights: those of det_n1
(simulate/targets.detector_targets, focal_w, gt_nn, simulate/targets.nn_weights); the GT position is the emitter position (not the image centroid,
which sits ~0.25 px away for the 4Pi PSF), so candidates come out centred on the emitter.
Init from det_n1 (models/det_n1.pt). Data come from `--workers` generator processes (the simulator is the bottleneck).
Usage: CUDA_VISIBLE_DEVICES=0 python analysis/fourpi/train_det4pi.py --out out/4pi/models/det4pi --steps 20000
"""
import argparse, json, os, sys, time
import torch
import torch.multiprocessing as mp
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
from simcode.util.ckpt import save as safe_save   # atomic, disk-guarded


def worker(q, seed, psf, B, size, dens, ksums=(1,)):
    import torch
    from simcode.simulate.multichannel import MultiChannelAcquisitionSim
    st = json.load(open(f'{P4}/psf4pi/{psf}.json'))['psf']
    rs = MultiChannelAcquisitionSim(st, H=size, W=size, seed=seed, n_psf=6)
    while True:
        S = rs.sample(B=B, dens=dens, ksum=ksums[int(torch.randint(len(ksums), (1,)))])
        g = S['gt']
        o = dict(frames=S['det_in'].cpu(), gt=dict(x=g['x'].cpu(), y=g['y'].cpu(), fov=g['b'].cpu(), phot=g['Nt'].cpu(),
                                                    onf=g['q'].cpu()), nb=B)
        q.put(o)


def to_dev(o, dev):
    return dict(frames=o['frames'].to(dev, non_blocking=True), gt={k: v.to(dev) for k, v in o['gt'].items()}, nb=o['nb'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--pct', type=float, default=0.05)
    ap.add_argument('--w_off', type=float, default=1.0); ap.add_argument('--init', default='det_n1.pt')
    ap.add_argument('--psf', default='NPC_f_dxy15'); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dens', default='0.01,2.0'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--ksum', default='1', help='comma list of K (summed windows: K x background, sqrt(K) read noise), one drawn per batch')
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('4pi-det')
    from simcode.detector.model import build
    from simcode.simulate.targets import detector_targets, gt_nn
    from simcode.detector.train import focal_w
    from simcode.detector.model import n_params
    from simcode.simulate.targets import nn_weights
    from simcode.config import resolve
    dev = 'cuda'
    os.makedirs(a.out, exist_ok=True)
    args = dict(up=1, ch=[48, 96, 160], stem_ch=16, fch=32, **{k: v for k, v in vars(a).items()})
    json.dump(args, open(a.out + '/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    net = build(args).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init if os.path.isabs(a.init) else resolve(a.init), map_location=dev, weights_only=False)['model'])
    print('params', n_params(net), flush=True)
    ctx = mp.get_context('spawn'); q = ctx.Queue(maxsize=6)
    dens = tuple(float(v) for v in a.dens.split(','))
    ps = [ctx.Process(target=worker, args=(q, a.seed * 100 + i, a.psf, a.batch, a.size, dens, tuple(int(v) for v in a.ksum.split(','))), daemon=True) for i in range(a.workers)]
    for p in ps: p.start()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps + 1, pct_start=a.pct)
    log = open(a.out + '/log.jsonl', 'a'); t0 = time.time(); S = a.size
    for step in range(a.steps + 1):
        o = to_dev(q.get(), dev)
        with torch.no_grad():
            label, ign, tid, off = detector_targets(o, 1, S, S)
            pm = label > 0
            w = torch.ones(int(pm.sum()), device=dev)
            if pm.any():
                gi = tid[pm]; w = nn_weights(o['gt']['fov'][gi], gt_nn(o['gt'])[gi])
            wmap = torch.zeros_like(label); wmap[pm] = w
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(o['frames'])
        lg = out['logit'].float(); dxy = out['dxy'].float(); ls = out['log_sig'].float()
        l_det = focal_w(lg, label, ign, wmap)
        if pm.any():
            e = (dxy - off).permute(0, 2, 3, 1)[pm]; s = ls[pm][:, None]
            l_off = ((0.5 * (e / s.exp()) ** 2 + s).sum(1) * w).sum() / w.sum().clamp(min=1e-6)
        else:
            l_off = torch.zeros((), device=dev)
        loss = l_det + a.w_off * l_off
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0:
            with torch.no_grad():
                rm = float(e.pow(2).mean().sqrt()) if pm.any() else float('nan')
                rec = float((torch.sigmoid(lg)[pm] > 0.5).float().mean()) if pm.any() else float('nan')
            r = dict(step=step, t=round(time.time() - t0), loss=loss.item(), det=l_det.item(), off=l_off.item(),
                     cell_recall=rec, n_pos=int(pm.sum()), off_rmse_px=rm, lr=sched.get_last_lr()[0], qsize=q.qsize())
            log.write(json.dumps(r) + '\n'); log.flush(); print(json.dumps(r), flush=True)
        if step % a.ckpt_every == 0 and step > 0:
            safe_save(dict(model=net.state_dict(), args=args, step=step), a.out + f'/ckpt_{step}.pt')
    safe_save(dict(model=net.state_dict(), args=args, step=a.steps), a.out + '/final.pt')
    for p in ps: p.terminate()


if __name__ == '__main__':
    main()
