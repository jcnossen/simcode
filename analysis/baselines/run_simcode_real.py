"""SIMCODE pipeline (paper settings) on a REAL movie (--movie; default model L64_5 = the paper network of the 2.5 nM movie 2;
use --model sf_conv_g1.3_tio2_L64_ni6_3 for the 0.5 nM cover1/1 movie). Code imported read-only from
~/simcode (smlmtorch), same as simcode_base/run_simcode.py; only the drift sign is patched locally (--fix_sign).
  stage 'net'  : --model (default L64_5; the paper network for this movie, probe_25nM.txt) / --ckpt (3) MovieProcessor (batch 16, overlap 5, prob 0.7, gain 0.56, offset 106.34), crlb_filter(0.2)
                 -> net cache (SFDataset .npy)
  stage 'ndi'  : estimate_angles + estimate_phases(5000/bin) -> const offsets -> re-estimate on crlb<0.1 -> depth 0.9 ->
                 mod_error<0.1 -> ndi_fit (eq. 2) -> |ndi-net|<0.2 px  = sfhd-ndi.  fix_sign 0 = paper (mod_at_frame '-='),
                 fix_sign 1 = offsets negated (= '+=' convention, SIMCODE as intended).
usage: CUDA_VISIBLE_DEVICES=2 python run_simcode_real.py net <cache.npy> --movie <movie.tif> [--model ...]
       CUDA_VISIBLE_DEVICES=2 python run_simcode_real.py ndi <cache.npy> <out.npz> --fix_sign 1   (1 = sign-fixed SIMCODE (b))
"""
import argparse, json, os, time, sys
from simcode2 import config as _cfg
SMLMTORCH_MODELS = os.environ.get('SMLMTORCH_MODELS') or _cfg._config().get('smlmtorch_models') or os.path.expanduser('~/smlmtorch/scripts/models')
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import numpy as np, torch
import smlmtorch.util.progbar
smlmtorch.util.progbar.USE_AUTO_TQDM = False
import matplotlib
matplotlib.use('Agg')
from smlmtorch import config_dict
from smlmtorch.nn.localize_movie import MovieProcessor
from smlmtorch.nn.sf_model import SFLocalizationModel
from smlmtorch.simflux.dataset import SFDataset
import smlmtorch.simflux.pattern_estimator as pe

MODELS = SMLMTORCH_MODELS + '/'
# NOTE (2026-11-09): the paper's net output for THIS movie (results/cover1_600mW_5.5ms_cf660/sf_intensities.npy) is reproduced
# by sf_conv_g1.3_L64_5 / checkpoint_3 (simcode_base2/probe_25nM.txt: 22328 vs 22328 locs on frames 0-190, 99.8 % identical positions).
# Camera: the paper's global gain 0.56 / offset 106.34 (config.pickle), as in the 0.5 nM run.
GAIN, OFFSET, PX = 0.56, 106.34, 108.33
PF = np.array([[0, 1, 2], [3, 4, 5]])
FIX_DEPTH, MOD_ERR, MAX_DIST = 0.9, 0.1, 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage'); ap.add_argument('cache'); ap.add_argument('out', nargs='?')
    ap.add_argument('--model', default='sf_conv_g1.3_L64_5'); ap.add_argument('--ckpt', type=int, default=3); ap.add_argument('--fix_sign', type=int, default=0)
    ap.add_argument('--maxframes', type=int, default=None)
    ap.add_argument('--movie', default='', help='movie tif (stage net)')
    ap.add_argument('--offset', type=float, default=106.34, help='global camera offset (ADU); the paper value 106.34 for the 0.5 / 2.5 / 20 nM movies')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.cuda.set_per_process_memory_fraction(0.3)     # 24 GB card shared with another job: stay < ~7 GB
    T = {}
    if a.stage == 'net':
        MDIR = MODELS + a.model
        config = config_dict.load(MDIR + '/config.yaml')
        config = config_dict(**config, detector=dict(prob_threshold=0.7, use_prob_weights=False))
        config.model.num_intensities = PF.size
        config.model.input_subpixel_index = False
        mp = MovieProcessor(SFLocalizationModel, config, f'{MDIR}/checkpoint_{a.ckpt}.pt', device=dev)
        t1 = time.time()
        kw = dict(maxframes=a.maxframes) if a.maxframes else {}
        ds = mp.process(_cfg.resolve(a.movie, 'data'), batch_size=16, batch_overlap=PF.size - 1, gain=GAIN, offset=a.offset, **kw)
        T['net_s'] = time.time() - t1; T['n_net_raw'] = len(ds)
        ds.crlb_filter(0.2); T['n_net'] = len(ds)
        ds.save(a.cache)
        print(json.dumps(T), flush=True)
        return
    ds = SFDataset.load(a.cache)
    ds['pixelsize'] = PX
    T['n_net'] = len(ds)
    t1 = time.time()
    rd = os.path.dirname(os.path.abspath(a.out)) + '/pe_figs/'; os.makedirs(rd, exist_ok=True)
    mp_d = pe.estimate_angles(pitch_minmax_nm=[150, 500], ds=ds, pattern_frames=PF, result_dir=rd, device=dev, moving_window=True)
    mp_d = pe.estimate_phases(ds, mp_d, spots_per_bin=5000, accept_percentile=50, iterations=1, verbose=False, device=dev)
    mp_d_co = mp_d.const_phase_offsets()
    mp_d.print_info(PX)
    pe_ds = ds[:]; pe_ds.crlb_filter(0.1)
    mp_d2 = pe.estimate_phases(pe_ds, mp_d_co, spots_per_bin=5000, accept_percentile=50, iterations=1, verbose=False, device=dev)
    mp_d2.print_info(PX)
    est_depth = mp_d2.depths.copy()
    mp_d2.depths = FIX_DEPTH
    mp_ndi = mp_d2.const_phase_offsets()
    if a.fix_sign: mp_ndi.phase_offset_per_frame = -mp_ndi.phase_offset_per_frame
    T['pattern_s'] = time.time() - t1
    mp_ndi.save(a.out.replace('.npz', '_mp_ndi.pickle'))
    t1 = time.time()
    me = mp_ndi.mod_error(ds)
    sel = me < MOD_ERR
    ndi_in = ds[sel]
    ndi = pe.ndi_fit_dataset(ndi_in, mp_ndi, device=dev, ndi_with_bg=False, fixed_intensity=False)
    dist = np.sqrt(((ndi.pos[:, :2] - ndi_in.pos[:, :2]) ** 2).sum(1))
    keep = dist < MAX_DIST
    ndi_k = ndi[keep]
    T['ndi_s'] = time.time() - t1
    T['n_moderr'] = int(sel.sum()); T['n_ndi'] = int(keep.sum())
    T['est_depth'] = [float(v) for v in est_depth]
    print(json.dumps(T), flush=True)
    idx = np.nonzero(sel)[0][keep]
    np.savez_compressed(a.out, ndi_idx=idx, frame=ds.frame[idx], x=ndi_k.pos[:, 0], y=ndi_k.pos[:, 1],
                        sx=ndi_k.crlb.pos[:, 0], sy=ndi_k.crlb.pos[:, 1], photons=ndi_k.photons,
                        net_x=ds.pos[idx, 0], net_y=ds.pos[idx, 1], moderr_all=me.astype(np.float32),
                        timing=json.dumps(T), fix_sign=a.fix_sign, cache=a.cache,
                        mp_k=mp_ndi.mod['k'], mp_phase=mp_ndi.mod['phase'], mp_depth=mp_ndi.mod['depth'],
                        mp_relint=mp_ndi.mod['relint'], mp_po=mp_ndi.phase_offset_per_frame)


if __name__ == '__main__':
    main()
