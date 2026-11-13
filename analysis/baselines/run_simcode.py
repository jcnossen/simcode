"""[simcode_base2 copy, 2026-11-09: + models ni6_3 / L64_5 (the paper networks, see simcode_config.txt here); memory cap]
SIMCODE pipeline exactly as used for the paper's 0.5 nM (1809/cover1/2) and 2.5 nM (movie 2) results, on a simulated
movie (sc6/sim_movie.py format). Source of truth: ~/smlmtorch/scripts/run_pipeline.py process() as called by
process_paper_datasets.ipynb (results written 2025-01-20; result config.pickle: gain 0.56, offset 106.34, dl_prob_threshold
0.7, pattern_frames [[0,1,2],[3,4,5]], pixelsize 108.33). Code imported read-only from ~/simcode (smlmtorch).
  network      models/sf_conv_g1.3_tio2_L48 (checkpoint_3 = epoch 159 per the commit in force on 2025-01-20; checkpoint_4 =
               epoch 160 per the 2025-02-03 commit), detector prob_threshold 0.7, use_prob_weights False,
               num_intensities 6, input_subpixel_index False; MovieProcessor batch 16, overlap 5 (every window start);
               global gain / offset; sf_int_ds.crlb_filter(0.2 px)
  pattern      estimate_angles(pitch 150-500 nm) on the net intensities; estimate_phases(spots_per_bin 5000, accept 50 %,
               1 iteration); const_phase_offsets; re-estimate on crlb < 0.1 px subset; depths fixed 0.9 (fix_depth default);
               mp_ndi = const_phase_offsets of that
  NDI          mod_error < 0.1 selection; ndi_fit_dataset(ndi_with_bg=False, fixed_intensity=False) (= eq. 2 LSQ on the
               net intensities with the net sigmas); |ndi - net| < 0.2 px -> 'sfhd-ndi'; merge_estimates (MVU, eq. 8)
               with the net positions -> 'sfhd-ndi+smlm'
Camera: the paper used global gain 0.56 / offset 106.34 ADU; the in-situ calibration of the real 0.5 nM movie (and our sim,
which uses that per-pixel offset map) has offset 99.3 ADU mean -> --cam paper reproduces the paper's ~7 ADU (3.9 e-/px/frame)
over-subtraction (followed by SIMCODE's clamp at 0); --cam map uses the correct per-pixel offset map.
Drift-sign bug (found here): estimate_phases stores phase_offset_per_frame = phase(f) - mean phase (model sin(k.r - phase)),
but ModulationPattern.mod_at_frame applies phase = mod.phase - offset (changed from += in smlmtorch commit fc2cc68, 2024-05-05,
i.e. before the paper processing of 2025-01-20) -> the estimated phase drift is applied with the wrong sign (the error is
~2x the drift). --fix_sign 1 negates the offsets (= SIMCODE as intended); the default reproduces the paper.
Variants: --model L64_2 (released ~/simcode/model, checkpoint_1), --pattern gt (true per-frame pattern converted to the
SIMCODE convention exc = relint (1 + depth sin(k.r - phase)); requires --coord_off = our_coord - simcode_coord (measured ~0: -0.006 px)).
usage: CUDA_VISIBLE_DEVICES=1 python run_simcode.py <sim_dir> <out.npz> [--model L48] [--ckpt 3] [--pattern est|gt]
"""
import argparse, json, os, time, sys
from simcode import config as _cfg
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
import smlmtorch.simflux.pattern_estimator as pe
from smlmtorch.simflux.pattern_estimator import ModulationPattern, ModulationType

MODELS = {'L48': SMLMTORCH_MODELS + '/sf_conv_g1.3_tio2_L48',
          'L64_2': os.path.expanduser('~/simcode/model/sf_conv_g1.3_tio2_L64_2'),
          # 2026-11-09 (simcode_base2): the networks that reproduce the paper's real-data net outputs exactly
          'ni6_3': SMLMTORCH_MODELS + '/sf_conv_g1.3_tio2_L64_ni6_3',   # 0.5 nM cover1/1 (100 % identical)
          'L64_5': SMLMTORCH_MODELS + '/sf_conv_g1.3_L64_5'}           # 1809/cover1/2 (99.9 %), 2.5 nM movie 2 (99.8 %)
GAIN, OFFSET, PX = 0.56, 106.34, 108.33
PF = np.array([[0, 1, 2], [3, 4, 5]])
FIX_DEPTH, MOD_ERR, MAX_DIST = 0.9, 0.1, 0.2


def gt_mp(sim_dir, nf, off):
    """true pattern -> ModulationPattern; ours: rel (1 + m cos(k.r_ours - phi)), r_ours = r_sc + off"""
    G = np.load(f'{sim_dir}/sim_gt.npz'); P = G['patt'].astype(np.float64)       # [NF, 5] kx ky phi m rel
    k = P[:, :2]; ph = P[:, 2] - np.pi / 2 - k @ np.asarray(off, np.float64)     # SIMCODE-convention phase per frame
    s = np.arange(len(P)) % 6
    mod = np.zeros(6, dtype=ModulationType)
    for i in range(6):
        m = s == i
        mod['k'][i, :2] = k[m].mean(0); mod['depth'][i] = P[m, 3].mean(); mod['relint'][i] = P[m, 4].mean()
        mod['phase'][i] = np.angle(np.exp(1j * ph[m]).mean())
    mod['relint'] /= mod['relint'].sum()
    nfp = max(nf, len(P)) + 6
    po = np.zeros(nfp)
    fr = np.arange(len(P))
    po[fr] = np.angle(np.exp(1j * (mod['phase'][s] - ph)))                        # mod_at_frame: phase = mod.phase - po[f]
    po[len(P):] = po[len(P) - 6:len(P)][np.arange(nfp - len(P)) % 6] if nfp > len(P) else 0
    return ModulationPattern(PF, mod=mod, moving_window=True, phase_offset_per_frame=po)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim_dir'); ap.add_argument('out')
    ap.add_argument('--model', default='L48'); ap.add_argument('--ckpt', type=int, default=3)
    ap.add_argument('--pattern', default='est'); ap.add_argument('--coord_off', default='0,0')
    ap.add_argument('--cam', default='paper')     # 'paper' = global gain 0.56 / offset 106.34 (as the paper); 'map' = our per-pixel offset map (99.3 ADU mean) + gain 0.56
    ap.add_argument('--net_cache', default='')
    ap.add_argument('--fix_sign', type=int, default=0)   # 1: apply the estimated phase drift with the sign estimate_phases defines it (see below)
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.cuda.set_per_process_memory_fraction(0.2)     # shared 24 GB card: stay < ~5 GB
    T = {}
    mdir = MODELS[a.model]
    ck = 1 if a.model == 'L64_2' else a.ckpt
    t0 = time.time()
    if a.net_cache and os.path.exists(a.net_cache):
        from smlmtorch.simflux.dataset import SFDataset
        ds = SFDataset.load(a.net_cache); T['net_s'] = float('nan')
    else:
        config = config_dict.load(mdir + '/config.yaml')
        config = config_dict(**config, detector=dict(prob_threshold=0.7, use_prob_weights=False))
        config.model.num_intensities = PF.size
        config.model.input_subpixel_index = False
        mp = MovieProcessor(SFLocalizationModel, config, f'{mdir}/checkpoint_{ck}.pt', device=dev)
        torch.cuda.synchronize(); t1 = time.time()
        off = OFFSET if a.cam == 'paper' else torch.tensor(np.load(_cfg.model('insitu/camera_maps_0.5nM.npz'))['offset_adu'], device=dev).float()
        ds = mp.process(f'{a.sim_dir}/sim_movie.tif', batch_size=16, batch_overlap=PF.size - 1, gain=GAIN, offset=off)
        torch.cuda.synchronize(); T['net_s'] = time.time() - t1; T['net_load_s'] = t1 - t0
        n_raw = len(ds)
        ds.crlb_filter(0.2)
        T['n_net_raw'] = n_raw
        if a.net_cache: ds.save(a.net_cache)
    ds['pixelsize'] = PX
    T['n_net'] = len(ds)
    t1 = time.time()
    rd = os.path.dirname(os.path.abspath(a.out)) + '/pe_figs/'; os.makedirs(rd, exist_ok=True)
    if a.pattern == 'est':
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
    else:
        mp_ndi = gt_mp(a.sim_dir, int(ds.numFrames), [float(v) for v in a.coord_off.split(',')])
        est_depth = mp_ndi.depths.copy()
        mp_ndi.print_info(PX)
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
    mrg = pe.merge_estimates(ndi_k, ndi_in[keep])
    torch.cuda.synchronize(); T['ndi_merge_s'] = time.time() - t1
    T['n_moderr'] = int(sel.sum()); T['n_ndi'] = int(keep.sum())
    T['est_depth'] = [float(v) for v in est_depth]
    print(json.dumps(T), flush=True)
    idx = np.nonzero(sel)[0][keep]           # rows of the net dataset that survive -> sfhd-ndi
    np.savez_compressed(a.out, frame=ds.frame, x=ds.pos[:, 0], y=ds.pos[:, 1], sx=ds.crlb.pos[:, 0], sy=ds.crlb.pos[:, 1],
                        photons=ds.photons, I=ds.ibg[:, :, 0], sI=ds.ibg_crlb[:, :, 0], bg=ds.background, moderr=me,
                        ndi_idx=idx, ndi_x=ndi_k.pos[:, 0], ndi_y=ndi_k.pos[:, 1], ndi_sx=ndi_k.crlb.pos[:, 0], ndi_sy=ndi_k.crlb.pos[:, 1],
                        ndi_photons=ndi_k.photons, mrg_x=mrg.pos[:, 0], mrg_y=mrg.pos[:, 1], mrg_sx=mrg.crlb.pos[:, 0],
                        mrg_sy=mrg.crlb.pos[:, 1], timing=json.dumps(T), cam=a.cam, fix_sign=a.fix_sign, model=f'{a.model}/checkpoint_{ck}', pattern=a.pattern,
                        mp_k=mp_ndi.mod['k'], mp_phase=mp_ndi.mod['phase'], mp_depth=mp_ndi.mod['depth'], mp_relint=mp_ndi.mod['relint'],
                        mp_po=mp_ndi.phase_offset_per_frame)


if __name__ == '__main__':
    main()
