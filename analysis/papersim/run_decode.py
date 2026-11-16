"""DECODE baseline on a papersim movie: the inference of analysis/baselines/decode/decode_paper.py (same model loading, the
same 6-frame sums starting at s + 6k for s in --starts, the same output npz format) with the movie's simulated camera
(camera_sim.npz: gain 1, flat offset 100 ADU) instead of a real dataset's camera map. decode_paper.py is imported, not changed.

Model: --name <model> in out/paper2/decode/lutpsf/<model>/ (default dec_1809_2 = trained on the averaged 1809/2 PSF, the PSF of
the simulations). Its training ranges (VARIANTS['sum6p'] in decode_paper.py) were set for the real movies: photons lognormal
(median 1000 / 6-sum, log-sd 0.6), bg U(22, 120) photons / px / 6-sum, density U(4, 45) emitters / 40 x 40 px / 6-sum; see
STATUS.md for the coverage of the simulation settings (bg 15 / 6-sum is below the range; densities 2 and 4 exceed it).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/run_decode.py <movie dir> [--name dec_1809_2] [--out <movie dir>/decode/locs.npz]
          [--maxframes N] [--starts 0,3]"""
import os, sys, json, time, argparse, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEC = os.path.join(REPO, 'analysis', 'baselines', 'decode')
_s = importlib.util.spec_from_file_location('decode_paper', os.path.join(DEC, 'decode_paper.py'))
DP = importlib.util.module_from_spec(_s); _s.loader.exec_module(DP)      # sets sys.path for dec_common / the decode fork
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('movie_dir')
    ap.add_argument('--name', default='dec_1809_2')
    ap.add_argument('--out', default=None)
    ap.add_argument('--maxframes', type=int, default=None)
    ap.add_argument('--starts', default='0,3')
    ap.add_argument('--model_file', default=None, help='weights (default <model dir>/model_final.pt), e.g. a frozen snapshot')
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('psim-decode')
    md = os.path.abspath(a.movie_dir)
    out = a.out or os.path.join(md, 'decode', 'locs.npz')
    D = DP.D
    mdir = f'{D.SPD}/lutpsf/{a.name}'
    pj = json.load(open(f'{mdir}/psf.json'))
    DP.setup_psf(pj['psf'], pj['halo'], pj.get('zmax_um', DP.ZMAX))
    mf = a.model_file or f'{mdir}/model_final.pt'
    inst = D.load_trained(a.name, pj.get('variant_name', 'sum6p'), model_file=mf if os.path.exists(mf) else None)
    tif = os.path.join(md, 'movie.tif')
    cam = np.load(os.path.join(md, 'camera_sim.npz'))
    gain, off = float(cam['gain']), cam['offset_adu'].astype(np.float32)
    from smlmtorch.util.multipart_tiff import tiff_get_movie_size
    _, nf = tiff_get_movie_size(tif)
    nf = nf if a.maxframes is None else min(nf, a.maxframes)
    parts = []; t0 = time.time()
    for s in [int(v) for v in a.starts.split(',')]:
        nsum = (nf - s) // 6
        def gen():
            buf = []
            for img in DP.frames(tif, s, nsum * 6):
                buf.append(img.astype(np.float32))
                if len(buf) == 6:
                    yield np.sum(buf, 0); buf = []
        em, shp = DP.D.process_padded(inst, gen(), gain=gain, offset=off * 6, sumframes=1, batch_size=64, infer_batch_size=16)
        em = em[em.frame_ix < nsum] if len(em) else em
        parts.append(dict(frame=(s + 6 * em.frame_ix.numpy()).astype(np.int64), x=em.xyz[:, 0].numpy(), y=em.xyz[:, 1].numpy(),
                          z=em.xyz[:, 2].numpy(), sx=em.xyz_sig[:, 0].numpy(), sy=em.xyz_sig[:, 1].numpy(), phot=em.phot.numpy(),
                          sphot=em.phot_sig.numpy() if em.phot_sig is not None else np.zeros(len(em)),
                          bg=em.bg.numpy() if em.bg is not None else np.zeros(len(em)), prob=em.prob.numpy()))
    torch.cuda.synchronize()
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    T = dict(frame_label=DP.D.FRAME_LABEL, infer_s=time.time() - t0, model=a.name, psf=pj['desc'], movie=md, nframes=nf, starts=a.starts, gain=gain,
             offset='camera_sim.npz (flat %.1f ADU)' % float(off.mean()), train_variant=pj.get('variant'), model_file=mf)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **R, info=json.dumps(T))
    print(md, a.name, 'n', len(R['frame']), 'infer s', round(T['infer_s'], 1), '->', out, flush=True)


if __name__ == '__main__':
    main()
