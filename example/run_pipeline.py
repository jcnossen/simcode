#%%
"""
This runs all the comparison pipeline steps to generate the results for the paper.
"""
import smlmtorch.util.progbar
# set False if your jupyter notebook does not support javascript plugins
smlmtorch.util.progbar.USE_AUTO_TQDM = False 

import dme
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import smlmtorch.simflux.pattern_estimator as pe
import numpy as np
from smlmtorch.simflux.simulate import angles_to_mod
import matplotlib.pyplot as plt
#from smlmtorch.ui.array_view_pyqt import image_view
from smlmtorch.util.multipart_tiff  import MultipartTiffSaver
from smlmtorch import Dataset

from smlmtorch.simflux.simulate import simulate
from smlmtorch.simflux import SFLocalizer

import tqdm
from fastpsf import Context, GaussianPSFMethods, CSplineMethods, CSplineCalibration

from smlmtorch.simflux.localizer_report import LocalizationReporter
from smlmtorch.simflux.pattern_estimator import ModulationPattern
from smlmtorch import config_dict

from smlmtorch.util.multipart_tiff import tiff_read_all
from smlmtorch.ui.array_view_pyqt import image_view

import matplotlib as mpl
import pickle

from smlmtorch.nn.localize_movie import MovieProcessor
from smlmtorch.nn.sf_model import SFLocalizationModel
from smlmtorch.simflux.dataset import SFDataset
import subprocess

import matplotlib.patches as patches

#%%


def get_result_filenames(tif_path):
    basepath = os.path.dirname(tif_path) + '/'
    tifname = os.path.splitext(os.path.basename(tif_path))[0]

    results_dir = basepath+f"results/{tifname}/"

    return dict(
        decode = results_dir + 'decode-summed6.hdf5',
        sf = results_dir + 'sf.hdf5',
        intensities = results_dir + 'sf_intensities.npy',
        smlm = results_dir + 'smlm.npy',
        simcode_merge = results_dir + 'sfhd-ndi+smlm.hdf5',
        simcode_ndi = results_dir + 'sfhd-ndi.hdf5'
    )

def load_results(tif_path, pixel_size=None):
    filenames = get_result_filenames(tif_path)

    r = dict(
        decode = Dataset.load(filenames['decode']),
        sf = Dataset.load(filenames['sf']),
        intensities = SFDataset.load(filenames['intensities']),
        smlm = SFDataset.load(filenames['smlm']),
        simcode_merge = Dataset.load(filenames['simcode_merge']),
        simcode_ndi = Dataset.load(filenames['simcode_ndi'])
    )

    if pixel_size is not None:
        for k in r:
            r[k]['pixelsize'] = pixel_size

    return r



def get_frc(ds, fn, crlb_filter, zoom=20, 
            smooth=4, use_cache=True, split_ds=False):
    frc_fn = os.path.splitext(fn)[0]+f'-frc2-cf{crlb_filter}.pickle'

    if os.path.exists(frc_fn) and use_cache and os.path.getmtime(frc_fn) > os.path.getmtime(fn):
        with open(frc_fn, 'rb') as f:
            d = pickle.load(f)

    else:
        ds_frc = ds[:]
        ds_frc.crlb_filter(crlb_filter)

        if split_ds: # split dataset into two time periods
            midframe = np.median(ds_frc.frame)
            mask = ds_frc.frame < midframe
            #mask = ds_frc.frame < ds_frc.numFrames//2
        else:
            mask = None
            ds_frc = ds_frc[ds_frc.frame%6==0]

        frc_val, frc_curve, frc_freq = ds_frc.frc(display=False, zoom=zoom, smooth=smooth, mask=mask)
        d = dict(frc=frc_val, curve=frc_curve, freq=frc_freq)

        with open(frc_fn, 'wb') as f:
            pickle.dump(d, f)

    return config_dict(d)


def get_drift_rcc(fn, ds):
    drift_fn = os.path.splitext(fn)[0]+'-drift-rcc.npy'

    if os.path.exists(drift_fn):
        print('loading drift from file: ', drift_fn)
        drift = np.load(drift_fn)
        return drift

    drift = ds.estimateDriftRCC(framesPerBin=4000,zoom=4)
    np.save(drift_fn, drift)
    return drift

def get_drift_dme(fn, ds : Dataset,  framesPerBin, initial_drift=None, **kwargs):
    drift_fn = os.path.splitext(fn)[0]+f'-drift-dme-{framesPerBin}.npy'

    if os.path.exists(drift_fn):
        print('loading drift from file: ', drift_fn)
        drift = np.load(drift_fn, allow_pickle=True)
        return drift

    if initial_drift is None:
        initial_drift = get_drift_rcc(fn, ds)

    drift = ds.estimateDriftMinEntropy(framesPerBin=framesPerBin,
        initialEstimate=initial_drift,
        useCuda=True, display=True, **kwargs)

    np.save(drift_fn, drift)
    return drift

def ndi_fits(ds : SFDataset, mp : ModulationPattern, result_dir, mod_error_threshold=0.1):
    mp_d2_co = mp.const_phase_offsets()
    print(f'computing modulation-enhanced positions using mod error threshold = {mod_error_threshold}')
    sd_sel = mp_d2_co.mod_error(ds) < mod_error_threshold

    ndi_input_ds = ds[sd_sel]
    sd_ndi_ds = pe.ndi_fit_dataset(ndi_input_ds, mp_d2_co, device='cuda:0', ndi_with_bg=False)

    dist = np.sqrt ( ( (sd_ndi_ds.pos[:,:2] - ndi_input_ds.pos[:,:2])**2 ).sum(1) )
    sd_ndi_ds = sd_ndi_ds[dist<0.1]
    sd_ndi_ds.save(result_dir + "raft-ndi.hdf5")

    print(f'remaining after filtering by max distance from original: {len(sd_ndi_ds)}/{len(ndi_input_ds)}')

    sd_combined_ds = pe.merge_estimates(sd_ndi_ds, ndi_input_ds[dist<0.1])
    sd_combined_ds.save(result_dir + "raft-ndi+smlm.hdf5")


def draw_rect(roi, zoom, ax):
    # add rectangle in ground truth at 'roi':
    zroi = np.array(roi)*zoom
    rect = patches.Rectangle(zroi[0], zroi[1][0]-zroi[0][0], zroi[1][1]-zroi[0][1], 
        linewidth=5, edgecolor='white', facecolor='none', transform=ax.transData)
    ax.add_patch(rect)
    return rect


def is_newer_file(file1, file2):
    if not os.path.exists(file2):
        return True
    return os.path.getmtime(file1) > os.path.getmtime(file2)

def compute_rsp(tif_path, datasets, border=4, zoom=20, fixedSigma=1.3):

    # compute diffraction limited mean images
    pc = []
    rmsd = []
    error_maps = []

    border = 2
    zoom=20

    mov = tiff_read_all(tif_path)
    df_mean = np.mean(mov, axis=0, dtype=np.float32)

    for i in range(len(datasets)):
        sr = datasets[i].render(zoom=zoom, clip_percentile=97)
        # remove edges
        sr_border = border*zoom
        sr = sr[sr_border:-sr_border, sr_border:-sr_border]
        df = df_mean[border:-border, border:-border]

        from pearson_corr import compute_correlation

        pc_, rmsd_, error_map_ = compute_correlation(sr, df, sigma=fixedSigma, align=False)

        plt.figure()
        plt.imshow(error_map_)

        pc.append(pc_)
        rmsd.append(rmsd_)
        error_maps.append(error_map_)

    return pc,rmsd, error_maps


def estimate_mp_smlm(smlm_ds, cfg, smlm_spots_per_bin=20000, mod_error_threshold=0.1, fix_depth=None):

    # filtered dataset for pattern estimation
    if smlm_ds.dims==3:
        smlm_pe_ds = smlm_ds[ np.abs(smlm_ds.pos[:,2]) < 0.1]
    else:
        smlm_pe_ds = smlm_ds[:]
        
    print(smlm_pe_ds)

    #mp_ds = ds[ds.chisq<2*cfg.r    oisize**2] # strongly filtered dataset for pattern estimation
    mp_smlm = pe.estimate_angles(pitch_minmax_nm=[150,500], ds=smlm_pe_ds, 
                            pattern_frames = cfg['pattern_frames'], 
                            result_dir = cfg['result_dir'], 
                            device = cfg['device'], moving_window=True)
    

        
    mp_smlm = pe.estimate_phases(smlm_pe_ds, mp_smlm, spots_per_bin=smlm_spots_per_bin, 
                            accept_percentile=50, iterations=8, verbose=False, device=cfg['device'])
    mp_smlm.print_info(cfg['pixelsize'])

    mp_smlm.save(cfg['result_dir']+"mp_smlm.pickle")

    if fix_depth is not None:
        mp_smlm.depths = fix_depth

    col = ['r', 'g', 'k']
    fig,ax=plt.subplots(2, 1,figsize=(10,8),sharex=True)
    #mp_smlm_co.plot_phase_drift(ax=ax, colors=col, label='CO {0}', linestyle='-', lw=1)
    mp_smlm.plot_phase_drift(ax=ax, colors=col, label='Pattern {0}', linestyle='-', lw=2)
    plt.savefig(cfg['result_dir']+'smlm-phase-drift.png')

    lr = LocalizationReporter(smlm_ds, cfg['result_dir'], mp_smlm)
    lr.draw_pattern(0, 2, me_threshold=mod_error_threshold)
    return mp_smlm


def process(camera_gain, camera_offset, pattern_frames, pixelsize, path, smlm_roisize=9, 
            psf_calib=[1.3,1.3], smlm_config={},device='cuda:0', smlm_max_chisq=4,
            use_gt_mp=False, fix_depth=0.9,
            simcode_model_name = 'models/sf_conv_g1.3_tio2_L48',
           simcode_model_class = SFLocalizationModel,
            decode_psf_fn = '/mnt/data/dmdsf/bn_dnapaint/PSF_calib/calibrated_model/zstack1_3dcal.mat',
            decode_model_name = 'decode-3f-g1.3',
            mod_error_threshold=0.1,
            xyfit=False,
            run_decode=True,
            smlm_spots_per_bin = 20000,
            dl_spots_per_bin = 5000,
            max_crlb_xy=0.5,
            dl_prob_threshold=0.7,
            archive_results=False):

    cfg = dict(
        psf_calib=psf_calib,
        roisize = smlm_roisize,
        detection_threshold=3.5,
        pattern_frames= pattern_frames,
        gain = camera_gain,
        offset = camera_offset,
        pixelsize = pixelsize,
        zrange= [-0.3,0.3],
        debug_mode=False,
        psf_sigma_binsize = None,
        result_dir='results'
    )

    cfg.update(smlm_config)
    cfg = config_dict(cfg)

    sfloc = SFLocalizer(path, **cfg, device=device)
    cfg['result_dir'] = sfloc.result_dir

    with open(sfloc.result_dir+'config.pickle', 'wb') as f:
        pickle.dump(dict(**cfg, dl_prob_threshold=dl_prob_threshold), f)

    sfloc.detect_spots(ignore_cache=False, moving_window=True)
    smlm_ds = sfloc.fit_smlm(max_crlb_xy=None, ignore_cache=False, max_chisq=smlm_max_chisq)
    smlm_ds.save(sfloc.result_dir+'smlm.npy')
    print(f"numrois: {sfloc.numrois}. #summed_fits: {sfloc.summed_fits.shape[0]}. numframes: {smlm_ds.numFrames}")

    mp_smlm = estimate_mp_smlm(smlm_ds, cfg, smlm_spots_per_bin, mod_error_threshold, fix_depth)
    mp_smlm_co = mp_smlm.const_phase_offsets()

    col = ['r', 'g', 'k']
    fig,ax=plt.subplots(2, 1,figsize=(10,8),sharex=True)
    #mp_smlm_co.plot_phase_drift(ax=ax, colors=col, label='CO {0}', linestyle='-', lw=1)
    mp_smlm.plot_phase_drift(ax=ax, colors=col, label='Pattern {0}', linestyle='-', lw=2)

    print(mp_smlm_co.depths)

    if use_gt_mp:
        mp_gt_path = os.path.splitext(path)[0]+"_mp_gt.pickle"
        mp_gt = ModulationPattern.load(mp_gt_path)
        mp_sf = mp_gt
        #print(mp_gt.mod)
        #print(mp_smlm_co.mod)
    else:
        mp_sf = mp_smlm_co 
        mp_gt = None
        
    mp_sf.save(sfloc.result_dir+"mp_sf.pickle")

    #me = mp_est.mod_error(smlm_ds)
    me = mp_sf.mod_error(smlm_ds)
    me_sel = me < mod_error_threshold

    sf_ds = sfloc.fit_simflux(mp_sf, smlm_ds[me_sel], iterations=50, lambda_=500, 
                normalizeWeights=True, ignore_cache=True)

    fig,ax=plt.subplots(1,2,figsize=(12,8))
    smlm_ds.crlb_filter(max_crlb_xy=max_crlb_xy, inplace=False).renderFigure(zoom=20, title='SMLM', clip_percentile=99, axes=ax[0])
    sf_ds.crlb_filter(max_crlb_xy=max_crlb_xy, inplace=False).renderFigure(zoom=20, title='SIMFLUX', clip_percentile=99, axes=ax[1])

    if_ = (smlm_ds.crlb_filter(max_crlb_xy=max_crlb_xy,inplace=False).crlb.pos[:,:2].mean() / 
        sf_ds.crlb_filter(max_crlb_xy=max_crlb_xy,inplace=False).crlb.pos[:,:2].mean())
    print(f'(biased) improvement factor (XY): {if_}')

    plt.savefig(sfloc.result_dir+'smlm-vs-sf.png')

    if is_newer_file(path, sfloc.result_dir+'sf_intensities.npy'):

        config = config_dict.load(simcode_model_name + '/config.yaml')
        config = config_dict(**config, 
            detector=dict(
                prob_threshold=dl_prob_threshold,
                use_prob_weights=False
            ))
        config.model.num_intensities = pattern_frames.size
        config.model.input_subpixel_index=False

        torch.cuda.empty_cache()
        mp = MovieProcessor(simcode_model_class, config, simcode_model_name+"/checkpoint_3.pt", device='cuda:0')
        sf_int_ds = mp.process(path, batch_size=16, batch_overlap=pattern_frames.size-1, gain=cfg['gain'], offset=cfg['offset'])
        sf_int_ds.crlb_filter(0.2)
        sf_int_ds.save(sfloc.result_dir+'sf_intensities.npy')
    else:
        sf_int_ds = SFDataset.load(sfloc.result_dir+'sf_intensities.npy')

    #raft_ds=None
    #sf_int_ds = SFDataset.load(sfloc.result_dir+"sf_intensities.npy")
    #sf_int_ds.crlb_filter(0.2)
        
    sf_int_ds['pixelsize'] = cfg.pixelsize

    #mp_ds = ds[ds.chisq<2*cfg.roisize**2] # strongly filtered dataset for pattern estimation
    mp_d = pe.estimate_angles(pitch_minmax_nm=[150,500], ds=sf_int_ds, 
                            pattern_frames = sfloc.pattern_frames, 
                            result_dir = sfloc.result_dir, 
                            device = sfloc.device,
                            moving_window = sfloc.moving_window)


    mp_d = pe.estimate_phases(sf_int_ds, mp_d, spots_per_bin=dl_spots_per_bin, 
                            accept_percentile=50, iterations=1, verbose=False, device=sfloc.device)
    mp_d_co = mp_d.const_phase_offsets()
    mp_d.print_info(sfloc.pixelsize)

    pe_ds = sf_int_ds[:] # reduced size pattern estimation dataset
    pe_ds.crlb_filter(0.1)
    mp_d2 = pe.estimate_phases(pe_ds, mp_d_co, spots_per_bin=dl_spots_per_bin, 
                            accept_percentile=50, iterations=1, verbose=False, device=sfloc.device)

    mp_d2.print_info(sfloc.pixelsize)

    col = ['r', 'g', 'k']
    fig,ax=plt.subplots(2, 1,figsize=(10,8),sharex=True)
    mp_d2.plot_phase_drift(nframes=100000,ax=ax, colors=col, label='Pat {0}', linestyle='-', lw=2)
    plt.savefig(sfloc.result_dir+'mp_dl-phase-drift.png')

    def estim_err(mp):
        npat = mp.pattern_frames.size
        for i in range(len(mp.pattern_frames)):
            pf = mp.pattern_frames[i]
            phase_drift = mp.phase_offset_per_frame
            phase_drift = phase_drift[:len(phase_drift)//npat*npat].reshape(-1,npat)
            phases = phase_drift[:,pf]
            est_err = (phases - phases.mean(1,keepdims=True)).std(1).mean()
            print(f'Phase estimation error angle {i}: {np.rad2deg(est_err):.2f} deg')
            
    print('SMLM: ', end=None); estim_err(mp_smlm)
    print('DL: ', end=None); estim_err(mp_d2)

    col = ['r', 'g', 'k']
    fig,ax=plt.subplots(2, 1,figsize=(10,8),sharex=True)
    mp_d2.plot_phase_drift(nframes=10000,ax=ax, colors=col, label='Pat {0}', linestyle='-', lw=2)
    plt.savefig(sfloc.result_dir+'mp_dl-phase-drift-zoom.png')

    if fix_depth is not None:
        mp_d2.depths = fix_depth

    lr = LocalizationReporter(pe_ds, sfloc.result_dir, mp_d2)
    for i in range(6):
        lr.draw_pattern(i, 2, me_threshold=mod_error_threshold)

    print(sfloc.result_dir)

    mp_d2.save(sfloc.result_dir+"mp_dl.pickle")
    mp_d2_co = mp_d2.const_phase_offsets()

    # Which pattern do we use?
    if mp_gt is None:
        mp_ndi = mp_d2_co
    else:
        mp_ndi = mp_gt
        
        print('using ground truth modulation pattern')
        print(mp_gt.mod)
        mp_d2.mod['phase'] = mp_d2.mod['phase'] % (np.pi *2)
        mp_smlm.mod['phase'] = mp_smlm.mod['phase'] % (np.pi *2)

    # This is the pattern used to fit
    mp_ndi.save(sfloc.result_dir+"mp_ndi.pickle")

    print(f'computing modulation-enhanced positions using mod error threshold = {mod_error_threshold}')
    sd_sel = mp_d2_co.mod_error(sf_int_ds) < mod_error_threshold

    ndi_input_ds = sf_int_ds[sd_sel]
    sd_ndi_ds = pe.ndi_fit_dataset(ndi_input_ds, mp_d2_co, device=sfloc.device, ndi_with_bg=False, fixed_intensity=False)

    max_dist = 0.2
    dist = np.sqrt ( ( (sd_ndi_ds.pos[:,:2] - ndi_input_ds.pos[:,:2])**2 ).sum(1) )
    sd_ndi_ds = sd_ndi_ds[dist<max_dist]
    sd_ndi_ds.save(sfloc.result_dir + "sfhd-ndi.hdf5")

    print(f'remaining after filtering by max distance from original: {len(sd_ndi_ds)}/{len(ndi_input_ds)}')

    sd_combined_ds = pe.merge_estimates(sd_ndi_ds, ndi_input_ds[dist<max_dist])
    sd_combined_ds.save(sfloc.result_dir + "sfhd-ndi+smlm.hdf5")

    #sd_ndi_ds_xy = pe.ndi_fit_dataset(ndi_input_ds, mp_d2_co, device=sfloc.device, ndi_with_bg=False, fixed_intensity=False)
    #dist = np.sqrt ( ( (sd_ndi_ds_xy.pos[:,:2] - ndi_input_ds.pos[:,:2])**2 ).sum(1) )
    #sd_ndi_ds_xy = sd_ndi_ds_xy[dist<max_dist]
    #sd_ndi_ds_xy.save(sfloc.result_dir + "sfhd-ndi-xy.hdf5")

    #tiff_fn = 'C:/dev/zimflux/simdecode/simulated/sim_gauss3D_sites_bg2_I2000_z600.tif'

    #decode_sw_ds = Dataset.load(sfloc.result_dir + "decode-summed3.hdf5")

    if run_decode:
        decode_output_fn = sfloc.result_dir + 'decode-summed6.hdf5'
        if is_newer_file(path, decode_output_fn):
            from decode_localize import decode_localize
            torch.cuda.empty_cache()
            decode_sw_ds=decode_localize(path, decode_model_name, decode_psf_fn, sumframes=6,
                            camera_gain=cfg['gain'], camera_offset=cfg['offset'], batch_size=16)
            decode_sw_ds.frame *= 6 
            decode_sw_ds.crlb_filter(0.2)
            decode_sw_ds.save(decode_output_fn)
        else:
            decode_sw_ds = Dataset.load(decode_output_fn)

    if archive_results:

        def compress_directory_to_7z(directory_path, output_path):
            try:
                subprocess.run(["7z", "a", output_path, directory_path], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"7z command failed: {str(e)}")

        #print(sfloc.result_dir[:-1])
        compress_directory_to_7z(sfloc.result_dir, sfloc.result_dir[:-1] + ".7z")        


    #%%
