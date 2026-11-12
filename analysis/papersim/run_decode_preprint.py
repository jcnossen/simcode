"""The preprint's own DECODE model on a papersim movie (reproduction of the preprint's simulations, Gaussian 1.3 px PSF):
/mnt/data/dmdsf/bn_dnapaint/PSF_calib/calibrated_model/zstack1_3dcal/decode-3f-g1.3/model_2.pt (identical file to
/mnt/data/decode-smlm/simflux-tnw/gauss2D_sig1.3px/decode-3f-g1.3/model_2.pt), loaded as smlmtorch scripts/decode_localize.py
does (its param_out.yaml, frame_block_size 1, no pattern prior), applied as run_pipeline.process did for the simulations:
non-overlapping 6-frame sums from frame 0 (--starts 0, 'decode-summed6'; frame = window start), photons = (ADU - offset) x gain
with the movie's simulated camera. The preprint then applied crlb_filter(0.2 px) before saving; here all rows are stored with
their sigma and the filter is left to the metrics (FRC set: sigma < 0.15 px; Jaccard: all rows, as the preprint loaded them
after its 0.2 px filter -> --crlb 0.2 applies it here too, default on).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/papersim/run_decode_preprint.py <movie dir> [--out <movie>/full/decode_pre/locs.npz]
          [--starts 0] [--crlb 0.2] [--maxframes N]"""
import os, sys, json, time, argparse, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
_s = importlib.util.spec_from_file_location('decode_paper', os.path.join(REPO, 'analysis', 'baselines', 'decode', 'decode_paper.py'))
DP = importlib.util.module_from_spec(_s); _s.loader.exec_module(DP)
import numpy as np, torch
MDIR = '/mnt/data/dmdsf/bn_dnapaint/PSF_calib/calibrated_model/zstack1_3dcal'


def load(model='decode-3f-g1.3', model_file='model_2.pt'):
    D = DP.D
    from decode.utils import param_io
    D._patch_fastpsf()
    p = param_io.load_params(f'{MDIR}/{model}/param_out.yaml')
    p.HyperParameter.pseudo_ds_size = 64; p.HyperParameter.batch_size = 16
    p.Hardware.num_worker_train = 0; p.Hardware.torch_threads = 1; p.Hardware.device_ix = 0
    p.Hardware.device = 'cuda:0'; p.Hardware.device_simulation = 'cpu'
    p.Simulation.frame_block_size = 1; p.HyperParameter.enable_pattern_prior = False
    p.TestSet.test_size = 16; p.TestSet.bg_uniform = 10
    inst = D.DecodeInstance(p, psf_path=f'{MDIR}.mat', model_name=model)
    inst.load_model(f'{MDIR}/{model}/{model_file}')
    return inst


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('movie_dir'); ap.add_argument('--out', default=None)
    ap.add_argument('--starts', default='0'); ap.add_argument('--crlb', type=float, default=0.2); ap.add_argument('--maxframes', type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from simcode.util.proctitle import auto; auto('psim-decpre')
    md = os.path.abspath(a.movie_dir); out = a.out or os.path.join(md, 'full', 'decode_pre', 'locs.npz')
    inst = load()
    tif = os.path.join(md, 'movie.tif'); cam = np.load(os.path.join(md, 'camera_sim.npz'))
    gain, off = float(cam['gain']), cam['offset_adu'].astype(np.float32)
    from smlmtorch.util.multipart_tiff import tiff_get_movie_size
    _, nf = tiff_get_movie_size(tif); nf = nf if a.maxframes is None else min(nf, a.maxframes)
    parts = []; t0 = time.time()
    for s in [int(v) for v in a.starts.split(',')]:
        nsum = (nf - s) // 6
        def gen():
            buf = []
            for img in DP.frames(tif, s, nsum * 6):
                buf.append(img.astype(np.float32))
                if len(buf) == 6: yield np.sum(buf, 0); buf = []
        em, _ = inst.process_iterable(gen(), gain=gain, offset=off * 6, sumframes=1, batch_size=64, infer_batch_size=16)
        em = em[em.frame_ix < nsum] if len(em) else em
        parts.append(dict(frame=(s + 6 * em.frame_ix.numpy()).astype(np.int64), x=em.xyz[:, 0].numpy(), y=em.xyz[:, 1].numpy(),
                          z=em.xyz[:, 2].numpy(), sx=em.xyz_sig[:, 0].numpy(), sy=em.xyz_sig[:, 1].numpy(), phot=em.phot.numpy(),
                          sphot=em.phot_sig.numpy() if em.phot_sig is not None else np.zeros(len(em)),
                          bg=em.bg.numpy() if em.bg is not None else np.zeros(len(em)), prob=em.prob.numpy()))
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n0 = len(R['x'])
    if a.crlb:
        k = np.maximum(R['sx'], R['sy']) < a.crlb; R = {kk: v[k] for kk, v in R.items()}
    T = dict(infer_s=time.time() - t0, model=f'{MDIR}/decode-3f-g1.3/model_2.pt', movie=md, nframes=nf, starts=a.starts, crlb_filter=a.crlb,
             n_before_crlb=int(n0), n=int(len(R['x'])))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **R, info=json.dumps(T)); print(json.dumps(T), flush=True)


if __name__ == '__main__':
    main()
