"""DECODE training at the published DECODE training scale (dec_sim1809v3, 2026-11-15).

Simulation recipe = dec_sim1809v2 (out/papersim/decode_v2psf/train_dec_sim1809v2.sh): in-situ PSF v2 LUT
(models/insitu/calib_1809_2_psf_v2.json, 0-0.40 um), emitter heights U(0, 400) nm, v2 camera gain 0.4256 e/ADU, read noise
3.28 e per 6-sum, bg U(6, 30) photons / px / 6-sum (structured), photons lognormal(median 450, log-sd 0.6), on-time 1 sum,
density 4-160 emitters per 40 x 40 px per sum, label_min 60, SigmaMUNet with 3 sums of context (dec_common.LutDecodeInstance,
the ~/decode2 fork imported read-only).

Training schedule = DECODE's published param_run.yaml (experimental_data_workflow): AdamW lr 6e-4, weight decay 0.1,
ReduceLROnPlateau(mode min, factor 0.5, patience 20, cooldown 10, threshold 1e-4) on the test-set loss, batch 64,
32 768 frames per epoch, 1000 epochs (= 32.8 M frames, 512 k iterations). No time cap.

Changes to the v2 training loop (documented in the model's train_v3.json):
  * density drawn per block of --dens_block sums (default 32) inside every epoch, U(4, 160) as before, by thinning a
    Poisson(160) emitter draw (per emitter id, so an emitter is kept / dropped as a whole within a block). v2 drew ONE density
    per epoch, so every epoch (256 iterations) saw a single density; the marginal distribution is the same.
  * own epoch loop (= DecodeInstance.training_iterator: train_val_impl.train, test_ds, scheduler.step(val loss) from epoch 1),
    with an atomic checkpoint every epoch (model, optimizer, scheduler, epoch, RNG states: chkpt/train_state.pt) so a killed run
    resumes exactly; state-dict snapshots every --snap_every epochs (snapshots/model_e<epoch>.pt) for the progress evaluation
    (analysis/decode_checks/v3/eval_watch.py); model_last.pt every epoch; model_final.pt at the end.
  * --init <state dict>: start from these weights (e.g. dec_sim1809v2's model_final.pt) with a fresh optimizer / scheduler
    (= lr reset); without --init: from scratch (with DECODE's first-epoch divergence check / re-initialisation).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/baselines/decode/train_v3.py --name dec_sim1809v3 [--init <pt>] [--epochs 1000]
          [--pseudo 32768] [--batch 64] [--snap_every 5]
Model dir: out/paper2/decode/lutpsf/<name>/ (train_log.csv: one row per epoch)."""
import os, sys, json, time, math, argparse, csv
os.environ.setdefault('TQDM_DISABLE', '1')
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
import decode_paper as DP           # sets sys.path for dec_common and the decode fork; D.SPD = <repo>/out/paper2/decode
D = DP.D
import decode
from decode.utils import param_io
from decode.utils.types import RecursiveNamespace

MAIN = os.path.expanduser('~/simcode2')        # outputs live in the main tree's out/ (worktrees share it)
# training-sim presets (photons = intensity parameter per 6-sum; bg photons / px / 6-sum; read e per 6-sum; dens emitters per
# 40 x 40 px per sum; life = mean on-time in sums). Real-data ranges in camera-v2 units: out/decode_checks/v3/real_stats_v2.json
# (real_stats_v2.py: SIMCODE v2 PSF-v2 runs, p >= 0.63, disjoint windows).
PRESETS = {
    # = dec_sim1809v2 (train_dec_sim1809v2.sh): the paper simulations only
    'sim1809': dict(psf='models/insitu/calib_1809_2_psf_v2.json',
                    vover={"med": 450.0, "lsd": 0.6, "th": 60.0, "life": 1.0, "bg": [6.0, 30.0], "read": 3.28, "label_min": 60.0,
                           "phot_max": 8000.0, "dens": [4.0, 160.0]}),
    # model 1 (user decision 2026-11-15: DECODE mirrors SIMCODE's model set): PSF v2 of 1809_2, for all simulations AND the
    # 1809_2 real data. Sims: photons lognormal(450, 0.6) (v2 recipe), bg 6-30 (D sweep 15, Dexp 7.9-15.8), density <= 160,
    # on-time 1 sum. 1809_2 (v2 units): photons median 736, p1-p99 289-1977; bg p1-p99 7.1-15.8; ~7.5 detected / 40 x 40 / sum;
    # linked on-time median 2 sums. -> photons lognormal(median 600, log-sd 0.7) (p1-p99 118-3060), bg U(6, 30), density
    # U(4, 160), on-time mean 2 sums.
    'm1809': dict(psf='models/insitu/calib_1809_2_psf_v2.json',
                  vover={"med": 600.0, "lsd": 0.7, "th": 60.0, "life": 2.0, "bg": [6.0, 30.0], "read": 3.28, "label_min": 60.0,
                         "phot_max": 8000.0, "dens": [4.0, 160.0]},
                  # network input / output scaling fixed to dec_sim1809v2's (param_out.yaml), so the arm continued from its
                  # weights sees the same input normalisation (autoset would give input_scale 15.3 for these photons)
                  scaling=dict(input_scale=10.774956268096291, input_offset=18.0, bg_max=36.0, phot_max=8000.0)),
    # model 2: PSF v2 of 2.5 nM/2 (SIMCODE m25), for 2.5 nM/2 (Fig 4; 20 nM/2 is not in the paper, its stats only widen the ranges). v2 units: photons median 1088 / 1455, p1-p99
    # 403-3642 / 549-5398; bg p1-p99 11.5-33.6 / 14.5-56.2; detected 16.2 / 18.8 per 40 x 40 per sum (true density higher);
    # read noise 1.39 e / frame (sqrt(6) x = 3.39 per 6-sum). Heights 0-0.40 um as SIMCODE m25 (calibration, clipped).
    # -> photons lognormal(median 1200, log-sd 0.7) (p1-p99 235-6100), bg U(8, 70), density U(4, 80), on-time mean 2 sums.
    'm25': dict(psf='models/m25/calib_2p5nM_m2_psf_v2.json',
                vover={"med": 1200.0, "lsd": 0.7, "th": 80.0, "life": 2.0, "bg": [8.0, 70.0], "read": 3.39, "label_min": 100.0,
                       "phot_max": 16000.0, "dens": [4.0, 80.0]}),
}
SCHED = dict(mode='min', factor=0.5, patience=20, cooldown=10, threshold=1e-4)


def atomic_save(obj, fn):
    tmp = fn + '.tmp'
    torch.save(obj, tmp); os.replace(tmp, fn)


def patch_density_blocks(inst, block, dens):
    """train sim: per-block density by thinning a Poisson(dens max) draw (replaces dec_common's per-epoch density draw)"""
    es = inst.sim_train.em_sampler
    lo, hi = dens
    es._em_avg = float(hi)
    es._emitter_av_total = es._em_avg * es._num_frames_plus / (es.lifetime_avg + 1)
    nblk = int(math.ceil(es.num_frames / block))
    orig = decode.simulation.emitter_generator.EmitterSamplerBlinking.sample.__get__(es)

    def sample():
        em = orig()
        if len(em) == 0: return em
        p = torch.from_numpy(np.random.uniform(lo, hi, nblk) / hi).float()
        nid = int(em.id.max()) + 1
        u = torch.rand(nid)
        b = (em.frame_ix.long() // block).clamp(0, nblk - 1)
        return em[u[em.id.long()] < p[b]]
    es.sample = sample


def build(a):
    P = PRESETS[a.preset]
    D.VARIANTS['sum6p'] = {**D.VARIANTS['sum6p'], **{k: tuple(v) if isinstance(v, list) else v for k, v in P['vover'].items()}}
    desc = DP.setup_psf(P['psf'], 0.0, 0.4)
    p = D.make_param('sum6p', epochs=a.epochs, pseudo=a.pseudo, batch=a.batch)
    p.Simulation.emitter_extent = [[-0.5, 39.5], [-0.5, 39.5], [0.0, 400.0]]; p.Simulation.psf_extent = p.Simulation.emitter_extent
    p.Camera.e_per_adu = 0.4256
    for k, v in P.get('scaling', {}).items(): setattr(p.Scaling, k, v)
    H = p.HyperParameter
    H.optimizer = 'AdamW'; H.opt_param.lr = a.lr; H.opt_param.weight_decay = 0.1
    H.learning_rate_scheduler = 'ReduceLROnPlateau'
    H.learning_rate_scheduler_param = RecursiveNamespace(**SCHED)
    inst = D.LutDecodeInstance(p, a.name, 'sum6p')
    if a.dens_block > 0: patch_density_blocks(inst, a.dens_block, D.VARIANTS['sum6p']['dens'])
    return inst, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='dec_sim1809v3'); ap.add_argument('--init', default=None)
    ap.add_argument('--epochs', type=int, default=1000); ap.add_argument('--pseudo', type=int, default=32768)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--snap_every', type=int, default=5); ap.add_argument('--dens_block', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=0, help='linear lr warm-up over this many iterations (from lr / 100), from scratch only')
    ap.add_argument('--seed', type=int, default=13); ap.add_argument('--preset', default='m1809', choices=sorted(PRESETS))
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('decode-v3-train')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    inst, desc = build(a)
    md = inst.model_dir; os.makedirs(f'{md}/snapshots', exist_ok=True)
    st_fn = f'{md}/chkpt/train_state.pt'
    resumed = os.path.exists(st_fn)
    if not resumed:
        param_io.save_params(f'{md}/param_train.yaml', inst.param)
        json.dump(dict(psf=PRESETS[a.preset]['psf'], desc=desc, halo=0.0, zmax_um=0.4, z_nm='0,400', e_per_adu=0.4256, variant_name='sum6p',
                       variant=D.VARIANTS['sum6p']), open(f'{md}/psf.json', 'w'), indent=1)
        json.dump(dict(args=vars(a), scheduler=SCHED, recipe=f'preset {a.preset} (train_v3.PRESETS) + DECODE published schedule', preset=PRESETS[a.preset],
                       density='per block of %d sums (thinning)' % a.dens_block if a.dens_block > 0 else 'per epoch (dec_common)',
                       started=time.ctime()), open(f'{md}/train_v3.json', 'w'), indent=1)
    inst.setup_trainer()
    if a.dens_block > 0: inst.ds_train.sample(True)        # setup_trainer sampled with the patched sampler already; resample is harmless
    e0 = 0
    if resumed:
        S = torch.load(st_fn, map_location=inst.param.Hardware.device, weights_only=False)
        inst.model.load_state_dict(S['model']); inst.optimizer.load_state_dict(S['optim']); inst.lr_scheduler.load_state_dict(S['lr_sched'])
        e0 = S['epoch'] + 1
        torch.set_rng_state(S['rng_torch'].cpu()); np.random.set_state(S['rng_np'])
        inst.ds_train.sample(True)
        print(f'resumed {st_fn} at epoch {e0}, lr {inst.optimizer.param_groups[0]["lr"]:.2e}', flush=True)
    elif a.init:
        sd = torch.load(a.init, map_location=inst.param.Hardware.device, weights_only=True)
        inst.model.load_state_dict(sd); print('init weights', a.init, flush=True)
    log_fn = f'{md}/train_log.csv'
    cols = ['epoch', 'train_loss', 'val_loss', 'lr', 'epoch_s', 'sim_s', 'elapsed_h', 'recall', 'precision', 'jaccard', 'rmse_lat', 'rmse_ax',
            'rmse_vol', 'efficiency_lat', 'n_train_em']
    if not os.path.exists(log_fn):
        with open(log_fn, 'w', newline='') as f: csv.writer(f).writerow(cols)
    conv = decode.neuralfitter.utils.progress.GMMHeuristicCheck(ref_epoch=1, emitter_avg=float(np.mean(D.VARIANTS['sum6p']['dens'])),
                                                                threshold=inst.param.HyperParameter.auto_restart_param.restart_treshold)
    dev = torch.device(inst.param.Hardware.device)
    if a.warmup > 0 and not a.init and e0 == 0:
        # (2026-11-15) the published lr 6e-4 diverged from scratch here (epoch-0 loss 1.6e8, DECODE auto-restart, again 3.8e5 after
        # the restart): linear warm-up lr / 100 -> lr over the first --warmup optimizer steps, then the plateau schedule
        def wrap(opt):
            step0 = opt.step; n = [0]
            def step(*args, **kw):
                if n[0] < a.warmup:
                    for g_ in opt.param_groups: g_['lr'] = a.lr * (0.01 + 0.99 * n[0] / a.warmup)
                elif n[0] == a.warmup:
                    for g_ in opt.param_groups: g_['lr'] = a.lr
                n[0] += 1
                return step0(*args, **kw)
            opt.step = step
        wrap(inst.optimizer)
    t_start = time.time(); restarts = 0
    i = e0
    while i < a.epochs:
        t0 = time.time()
        train_loss = decode.neuralfitter.train_val_impl.train(model=inst.model, optimizer=inst.optimizer, loss=inst.loss, dataloader=inst.dl_train,
                                                               grad_rescale=inst.param.HyperParameter.moeller_gradient_rescale,
                                                               grad_mod=inst.param.HyperParameter.grad_mod, epoch=i, device=dev, logger=inst.logger)
        val_loss, test_out, res = inst.test_ds(inst.ds_test, epoch=i, dataloader=inst.dl_test)
        if not a.init and not resumed and restarts < 5 and not conv(test_out.loss[:, 0].mean(), i):
            print(f'epoch {i}: pathological loss, re-initialising (DECODE auto-restart)', flush=True)
            restarts += 1; inst._init_model(); inst.setup_trainer(); i = 0
            if a.warmup > 0: wrap(inst.optimizer)
            conv = decode.neuralfitter.utils.progress.GMMHeuristicCheck(ref_epoch=1, emitter_avg=conv.emitter_avg, threshold=conv.threshold)
            continue
        if i >= 1: inst.lr_scheduler.step(val_loss)
        lr = inst.optimizer.param_groups[0]['lr']
        S = dict(model=inst.model.state_dict(), optim=inst.optimizer.state_dict(), lr_sched=inst.lr_scheduler.state_dict(), epoch=i,
                 rng_torch=torch.get_rng_state(), rng_np=np.random.get_state())
        atomic_save(S, st_fn)
        atomic_save(inst.model.state_dict(), f'{md}/model_last.pt')
        if i % a.snap_every == 0 or i == a.epochs - 1:
            atomic_save(inst.model.state_dict(), f'{md}/snapshots/model_e{i:04d}.pt')
        t1 = time.time()
        inst.ds_train.sample(True)
        sim_s = time.time() - t1
        r = res['result']._asdict() if hasattr(res['result'], '_asdict') else {}
        n_em = len(inst.ds_train._emitter) if getattr(inst.ds_train, '_emitter', None) is not None else -1
        row = [i, float(train_loss), float(val_loss), lr, round(time.time() - t0, 1), round(sim_s, 1), round((time.time() - t_start) / 3600, 3)] + \
              [float(r.get(k, np.nan)) for k in ('rec', 'prec', 'jac', 'rmse_lat', 'rmse_ax', 'rmse_vol', 'effcy_lat')] + [n_em]
        with open(log_fn, 'a', newline='') as f: csv.writer(f).writerow(row)
        print('epoch %d train %.2f val %.2f lr %.2e  %.0f s (sim %.0f s)  jac %.3f rmse_lat %.1f' % (i, train_loss, val_loss, lr, row[4], sim_s,
              row[9], row[10]), flush=True)
        i += 1
    atomic_save(inst.model.state_dict(), f'{md}/model_final.pt')
    print('done', flush=True)


if __name__ == '__main__':
    main()
