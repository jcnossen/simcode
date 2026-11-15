"""DECODE baseline of the paper rerun: one model per PSF, trained on DECODE's simulator with that PSF (zpsf spec: the dataset's
averaged in-situ PSF), applied to the real 6-frame-summed movies.

Configuration = the paper's DECODE (decode-3f-g1.3, ~/smlmtorch/scripts/decode_localize.py): SigmaMUNet, 3-frame context
(channels_in 3) on 6-frame sums, depth_union 2, sigmoid intensity scale; driven through dec_common.LutDecodeInstance
(analysis/baselines/decode/dec_common.py: DECODE fork ~/decode2 imported read-only, structured bg, sCMOS-like camera).
Training PSF: --psf <spec> (analysis/baselines/zpsf.py) rendered into DECODE's per-subpixel LUT (21 z planes 0-200 nm x 33 x 33
subpixel shifts, 41 x 41 px; bilinear subpixel / linear z, as dec_common.LutPSF), no halo / PSF jitter (pure averaged PSF;
--halo to add the 0-5 % sigma-4 px halo of the hd_sum6 model). Emitter z uniform 10-150 nm (DECODE emitter_extent).
Training-sim ranges (VARIANTS['sum6p']) cover the 3 paper movies (real_stats_paper2.py -> out/paper2/decode_real_stats.json;
SIMCODE v2 pass-1 rows, p >= 0.63, disjoint windows; per 6-frame sum):
  photons  median 820 / 960 / 1210, log-sd 0.45 / 0.51 / 0.56, p1-p99 300-4800  -> lognormal(median 1000, log-sd 0.6)
  bg       p1-p99 26-36 / 34-80 / 44-110 photons/px                             -> U(22, 120) per 48-sum block (+ structure)
  density  detected 8.2 / 16.7 / 24.1 emitters per 40 x 40 px per sum           -> U(4, 45) per epoch (20 nM true > detected)
  on-time  linked disjoint windows 2-3.5 sums (a lower bound: detection gaps)    -> mean 10.8 sums (65 frames, as hd_sum6;
           mean 5 sums with these photon / bg / density ranges made the training diverge reproducibly in the first epoch)
Camera at inference: the movie's in-situ map (per-pixel offset, gain 0.56; paper_pipeline.py --camera maps), photons =
(sum of 6 frames - 6 offset) x gain; the training camera = Poisson + 0.9 sqrt(6) e read noise.
Output frames: 6-frame sums starting at frame s + 6k for s in --starts (default 0,3 = window starts w % 3 == 0; the paper used
s = 0 only); frame = window start. x / y: DECODE px (pixel centre = integer, x = column), i.e. the smlmtorch / fastpsf convention.

usage:
  train: CUDA_VISIBLE_DEVICES=1 python analysis/baselines/decode/decode_paper.py train --psf <spec> --name <model>
             [--epochs 400] [--max_hours 6] [--pseudo 8192] [--halo 0] [--zmax_um 0.2 --z_nm 10,150 --e_per_adu 0.56]
             (2026-11-15 opt-in for the v2-PSF sims: --zmax_um 0.4 --z_nm 0,400 = LUT and emitter heights 0-0.40 um,
             --e_per_adu 0.4256 = the v2 camera gain; inference reads zmax_um from psf.json)
  infer: CUDA_VISIBLE_DEVICES=1 python analysis/baselines/decode/decode_paper.py infer <1809_2|2.5nM_2|20nM_2> --name <model>
             --out <locs.npz> [--maxframes N] [--starts 0,3]
  models: out/paper2/decode/lutpsf/<model>/ (param_train.yaml, model_final.pt / checkpoint, psf.json)"""
import os, sys, json, time, math, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, REPO]          # not analysis/baselines: its decode/ dir would shadow the decode package
import dec_common as D
import importlib.util
_s = importlib.util.spec_from_file_location('zpsf', os.path.join(os.path.dirname(HERE), 'zpsf.py'))
zpsf = importlib.util.module_from_spec(_s); _s.loader.exec_module(zpsf)
from decode.utils import param_io

D.SPD = f'{REPO}/out/paper2/decode'
D.LUTPSF = f'{D.SPD}/lutpsf.txt'
D.VARIANTS['sum6p'] = dict(med=1000.0, lsd=0.6, th=120.0, life=65.0 / 6, bg=(22.0, 120.0), read=0.9 * math.sqrt(6), pattern=False,
                           jitter=False, label_min=150.0, phot_max=16000.0, dens=(4.0, 45.0))
DATASETS = {'1809_2': ('1809/cover1/2/simflux_5.5ms_500mW.tif', 'camera_maps_1809_2.npz'),
            '2.5nM_2': ('dnapaint/2.5nM/2/cover1_600mW_5.5ms_cf660.tif', 'camera_maps_2p5nM_m2.npz'),
            '20nM_2': ('dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif', 'camera_maps_20nM_m2.npz')}
ZMAX = 0.2


def psf_luts(spec, lz=21, ls=32, R=41, zmax=ZMAX, dev='cuda'):
    """DECODE LUT [lz, ls+1, ls+1, R, R] of a zpsf PSF: lut[iz, iy, ix] = image of an emitter at (fx, fy) = (ix, iy) / ls - 0.5
    px from the centre pixel R // 2, height z = iz / (lz - 1) zmax um"""
    psf = zpsf.load(spec, dev, zmax=max(zmax, 0.2))
    sub = torch.arange(ls + 1, device=dev, dtype=torch.float32) / ls - 0.5
    out = torch.zeros(lz, ls + 1, ls + 1, R, R, device=dev)
    FY, FX = torch.meshgrid(sub, sub, indexing='ij'); FX, FY = FX.reshape(-1), FY.reshape(-1)
    for iz in range(lz):
        z = torch.full_like(FX, iz / (lz - 1) * zmax)
        with torch.no_grad():
            imgs = torch.cat([psf.eval(FX[s:s + 128], FY[s:s + 128], R, z=z[s:s + 128])[0] for s in range(0, len(FX), 128)])
        out[iz] = imgs.reshape(ls + 1, ls + 1, R, R)
    return out, psf.desc


def setup_psf(spec, halo, zmax=ZMAX):
    """zmax (um): top of the LUT (default ZMAX = 0.2; 10 nm planes); the v2-PSF model (2026-11-15) uses 0.40 (psf.json zmax_um)"""
    lut, desc = psf_luts(spec, lz=int(round(zmax / 0.01)) + 1, zmax=zmax)
    D._LUTS = ([lut], zmax, float(halo))
    return desc


def train(a):
    if a.vover:   # test overrides of the variant's sim ranges, e.g. '{"dens": [12, 45]}'
        D.VARIANTS[a.variant] = {**D.VARIANTS[a.variant], **{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.vover).items()}}
    desc = setup_psf(a.psf, a.halo, a.zmax_um)
    torch.manual_seed(11); np.random.seed(11)
    p = D.make_param(a.variant, epochs=a.epochs, pseudo=a.pseudo)
    z0, z1 = (float(v) for v in a.z_nm.split(','))
    p.Simulation.emitter_extent = [[-0.5, 39.5], [-0.5, 39.5], [z0, z1]]; p.Simulation.psf_extent = p.Simulation.emitter_extent
    if a.e_per_adu: p.Camera.e_per_adu = a.e_per_adu
    inst = D.LutDecodeInstance(p, a.name, a.variant)
    param_io.save_params(f'{inst.model_dir}/param_train.yaml', inst.param)
    json.dump(dict(psf=a.psf, desc=desc, halo=a.halo, zmax_um=a.zmax_um, z_nm=a.z_nm, e_per_adu=a.e_per_adu or 0.56, variant_name=a.variant, variant=D.VARIANTS[a.variant]), open(f'{inst.model_dir}/psf.json', 'w'), indent=1)
    inst.setup_trainer()
    if os.path.exists(inst.chkpt_path):
        try: inst.load_checkpoint(); print('resumed from checkpoint', flush=True)
        except Exception as e: print('no resume:', e, flush=True)
    t0 = time.time()
    for i, loss in enumerate(inst.training_iterator(use_conv_check=True)):
        el = (time.time() - t0) / 3600
        print(f'epoch {i} train_loss {float(loss):.4f} lr {inst.optimizer.param_groups[0]["lr"]:.2e} elapsed_h {el:.2f}', flush=True)
        if el > a.max_hours:
            print('time limit reached', flush=True); break
    # always leave a loadable model_final.pt (DecodeInstance writes it at the end of a full run only)
    mf = f'{inst.model_dir}/model_final.pt'
    if not os.path.exists(mf) or os.path.getmtime(mf) < t0:
        torch.save(inst.model.state_dict(), mf)
    print('done', flush=True)


def frames(tif, start, maxframes):
    from smlmtorch.util.multipart_tiff import tiff_read_file
    return tiff_read_file(tif, startframe=start, maxframes=maxframes, use_progbar=False)


def infer(a):
    mdir = f'{D.SPD}/lutpsf/{a.name}'
    pj = json.load(open(f'{mdir}/psf.json'))
    setup_psf(pj['psf'], pj['halo'], pj.get('zmax_um', ZMAX))
    mf = f'{mdir}/model_final.pt'
    inst = D.load_trained(a.name, pj.get('variant_name', 'sum6p'), model_file=mf if os.path.exists(mf) else None)
    rel, camf = DATASETS[a.dataset]
    tif = f'/mnt/data/dmdsf/bn_dnapaint/{rel}'
    cam = np.load(f'{REPO}/models/insitu/{camf}')
    gain, off = float(cam['gain']), cam['offset_adu'].astype(np.float32)
    from smlmtorch.util.multipart_tiff import tiff_get_movie_size
    _, nf = tiff_get_movie_size(tif)
    nf = nf if a.maxframes is None else min(nf, a.maxframes)
    parts = []; t0 = time.time()
    for s in [int(v) for v in a.starts.split(',')]:
        nsum = (nf - s) // 6
        def gen():
            buf = []
            for img in frames(tif, s, nsum * 6):
                buf.append(img.astype(np.float32))
                if len(buf) == 6:
                    yield np.sum(buf, 0); buf = []
        em, shp = D.process_padded(inst, gen(), gain=gain, offset=off * 6, sumframes=1, batch_size=64, infer_batch_size=16)
        em = em[em.frame_ix < nsum] if len(em) else em
        parts.append(dict(frame=(s + 6 * em.frame_ix.numpy()).astype(np.int64), x=em.xyz[:, 0].numpy(), y=em.xyz[:, 1].numpy(),
                          z=em.xyz[:, 2].numpy(), sx=em.xyz_sig[:, 0].numpy(), sy=em.xyz_sig[:, 1].numpy(), phot=em.phot.numpy(),
                          sphot=em.phot_sig.numpy() if em.phot_sig is not None else np.zeros(len(em)),
                          bg=em.bg.numpy() if em.bg is not None else np.zeros(len(em)), prob=em.prob.numpy()))
    torch.cuda.synchronize()
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    T = dict(frame_label=D.FRAME_LABEL, infer_s=time.time() - t0, model=a.name, psf=pj['desc'], dataset=a.dataset, nframes=nf, starts=a.starts, gain=gain,
             offset='camera map ' + camf)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **R, info=json.dumps(T))
    print(a.dataset, a.name, 'n', len(R['frame']), 'infer s', round(T['infer_s'], 1), '->', a.out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'infer'])
    ap.add_argument('dataset', nargs='?')
    ap.add_argument('--psf'); ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=400); ap.add_argument('--max_hours', type=float, default=6.0)
    ap.add_argument('--pseudo', type=int, default=8192); ap.add_argument('--halo', type=float, default=0.0)
    ap.add_argument('--zmax_um', type=float, default=ZMAX); ap.add_argument('--z_nm', default='10,150')
    ap.add_argument('--e_per_adu', type=float, default=None)
    ap.add_argument('--variant', default='sum6p'); ap.add_argument('--vover', default=None); ap.add_argument('--out'); ap.add_argument('--maxframes', type=int, default=None); ap.add_argument('--starts', default='0,3')
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('decode')     # recognizable name in nvidia-smi / ps
    train(a) if a.cmd == 'train' else infer(a)
