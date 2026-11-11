"""
Exact reproduction of the SIMCODE paper's experimental figures (Fig 3 = low density, Fig 4 = medium density).

plot_results() and its helpers (plot_frcs, get_drift, plot_drift, compute_pc) are copied verbatim from
~/smlmtorch/scripts/fig_bn_dnapaint_results.py; get_frc, get_drift_rcc, get_drift_dme and draw_rect verbatim from
~/smlmtorch/scripts/run_pipeline.py (copied instead of imported, because run_pipeline imports the whole training stack).
compute_pc's `pearson_corr` is imported read-only from ~/smlmtorch/scripts. Dataset / SFDataset / ModulationPattern /
LocalizationReporter come from the installed smlmtorch (~/simcode; its Dataset, SFDataset, FRC and render code is
identical to ~/smlmtorch's), DME / RCC from the installed `dme` package (~/dme/dme: dme.dme.dme_estimate + native
libdme_cuda.so = the original DME, not DME v2), as in the original.

Changes vs the original: only the paths (server working dirs under out/paper/, see FIGS) and the save locations
(plot_results(..., fig_out=...) additionally writes <fig_out>.svg/.png and returns the FRC / count / RSP values).
The original writes plots/, drift caches, FRC pickles and undrifted_ds/ next to its inputs and <tif>_mean.npy /
<tif>_rse_cache_*.pkl next to the movie: the inputs are therefore extracted / symlinked into out/paper/<set>/.

usage: CUDA_VISIBLE_DEVICES=1 python analysis/paper/fig_exp.py 3|4
"""
import os, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pickle

from smlmtorch import Dataset, config_dict
from smlmtorch.simflux.dataset import SFDataset
from smlmtorch.simflux.pattern_estimator import ModulationPattern
from smlmtorch.simflux.localizer_report import LocalizationReporter
import smlmtorch.simflux.pattern_estimator as pe
from smlmtorch.util.multipart_tiff import tiff_read_all
import dme

sys.path.append(os.path.expanduser('~/smlmtorch/scripts'))   # for `from pearson_corr import compute_correlation` only

new_rc_params = {
#    "font.family": 'Times',
    "font.size": 15,
    "font.serif": [],
    "svg.fonttype": 'none'} #to store text as text, not as path
mpl.rcParams.update(new_rc_params)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(REPO, 'out', 'paper')

# paper calls: fig_bn_dnapaint_results.py lines 706-722 (Fig 3) and 756-769 (Fig 4); only the paths changed
FIGS = {
    3: dict(basedir=f'{PAPER}/1809_2/published/simflux_5.5ms_500mW/',
            tif_path=f'{PAPER}/1809_2/movie/simflux_5.5ms_500mW.tif',
            renders=[
                ((75,28), 9, {'scalebar_nm': 500, 'clip_percentile': 97}),
                ((75,76), 13, {'scalebar_nm': 500, 'clip_percentile': 97})
            ], render_args=dict(clip_percentile=96),
            pixelsize=108.33,
            fig_out=f'{PAPER}/fig3/fig_exp_lowdensity'),
    4: dict(basedir=f'{PAPER}/2.5nM_2/published/',
            tif_path=f'{PAPER}/2.5nM_2/movie/cover1_600mW_5.5ms_cf660.tif',
            renders=[
                ((50, 15), 30, {'scalebar_nm': 500, 'clip_percentile': 98}),
                ((85, 40), 15, {'scalebar_nm': 500, 'clip_percentile': 98})
            ], render_args={'clip_percentile': 98},
            frc_zoom=15,
            pixelsize=108.33,
            fig_out=f'{PAPER}/fig4/fig_exp_meddensity'),
}


# ---------------- verbatim from run_pipeline.py ----------------
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
    drift = drift.astype(np.float32)
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


def draw_rect(roi, zoom, ax):
    # add rectangle in ground truth at 'roi':
    zroi = np.array(roi)*zoom
    rect = patches.Rectangle(zroi[0], zroi[1][0]-zroi[0][0], zroi[1][1]-zroi[0][1], 
        linewidth=5, edgecolor='white', facecolor='none', transform=ax.transData)
    ax.add_patch(rect)
    return rect


# ---------------- verbatim from fig_bn_dnapaint_results.py (plot_results: + fig_out) ----------------

def plot_frcs(ax, frcs, labels, colors=None):
    L = len(frcs[0].curve)*2
    freq = frcs[0].freq
    freq = freq[5:L//2]

    print(L)
    crop = 200

    ax.axhline(1/7, color='grey', linestyle='--')
    for i in range(len(frcs)):
        curve = frcs[i].curve
        curve = curve[5:L//2]
        l=ax.plot(freq[:-crop], curve[:-crop], label=f'{labels[i]} (FRC={frcs[i].frc:.1f} nm)')

        if colors is not None:
            l[0].set_color(colors[i])

    ax.legend(fontsize=13)
    ax.set_ylabel('FRC')
    # disable y axis
    ax.set_yticklabels([])
    ax.set_xlabel('Spatial freq. [nm^-1]')
    ax.set_title('Fourier ring correlation')


def get_drift(ds, fn, crlb_filter=0.1):
    ds=ds[:]
    ds.crlb_filter(crlb_filter)

    print(f'estimating drift with {crlb_filter}px CRLB filter')
    drift_rcc = get_drift_rcc(fn, ds)

    drift_dme_coarse = get_drift_dme(fn, ds,dims=2, maxspots=1000000,
            framesPerBin=2000, 
            initial_drift=drift_rcc,
            maxNeighbors=500,
            #initialEstimate=None, #initial_drift,
            initializeWithRCC=None)

    drift_dme = get_drift_dme(fn, ds,dims=2, maxspots=1000000,
            framesPerBin=50,
            maxNeighbors=500,
            #initialEstimate=None, 
            initial_drift=drift_dme_coarse,
            #returnCV=True, # also return the 2 drift estimates used to estimate precision
            estimatePrecision=True,
            initializeWithRCC=None)

    return dict(drift = drift_dme, 
                       drift_coarse_dme = drift_dme_coarse, 
                       drift_rcc = drift_rcc)

def plot_drift(fn, drift_dict : dict, pixelsize):
    """
    fig,ax=plt.subplots()
    ax.plot(drift[:,0]*pixelsize, drift[:,1]*pixelsize)
    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    ax.set_title('Drift')
    """

    drift = drift_dict['drift']

    coarse = drift_dict['drift_coarse_dme']
    rcc = drift_dict['drift_rcc']

    styles = ['r', 'b', 'k']
    ofs = 15
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    datasets = [rcc, coarse, drift]
    labels = ['RCC (4000 frames/bin)', 'DME (2000 frames/bin)', 'DME (50 frames/bin)']
    offsets = [-ofs, 0, ofs]

    for i in range(2):
        for j in range(len(datasets)):
            ax[i].plot(datasets[j][:, i] * pixelsize + offsets[j], styles[j], label=labels[j])
        ax[i].set_ylabel(f'{"X" if i == 0 else "Y"} [nm]')

    ax[0].set_title('Drift')
    ax[1].set_xlabel('Frame')

    ax[0].legend()
    plt.savefig(fn, dpi=300)

def compute_pc(tif_path, datasets, border=2, zoom=20, sigma=1.3):
    # compute simple checksum from datasets, based on ds.pos
    checksum = 1
    for ds in datasets:
        checksum += ds.pos.sum()
    checksum += border*zoom*sigma # bad checksum, but good enough for this purpose
    
    cache_fn = tif_path + f'_rse_cache_{checksum}.pkl'
    if os.path.exists(cache_fn):
        with open(cache_fn, 'rb') as f:
            return pickle.load(f)

    from pearson_corr import compute_correlation

    # compute mean image, in a cached way
    mean_img_fn = tif_path + '_mean.npy'
    if os.path.exists(mean_img_fn):
        mean_img = np.load(mean_img_fn)
    else:
        mov = tiff_read_all(tif_path)
        mean_img = np.mean(mov, axis=0, dtype=np.float32)
        np.save(mean_img_fn, mean_img)

    pc = np.zeros(len(datasets))
    rmsd = np.zeros(len(datasets))
    error_maps = []

    for i in range(len(datasets)):
        sr = datasets[i].render(zoom=zoom)

        pc[i], rmsd[i], error_map = compute_correlation(sr, mean_img, sigma=sigma, align=False)
        error_maps.append(error_map)


    with open (cache_fn, 'wb') as f:
        pickle.dump([pc, rmsd, error_maps], f)

    return pc, rmsd, error_maps

def plot_results(basedir, renders, pixelsize,  
                 render_args=None, numframes=None, ignore_cache=False, 
                 tif_path=None,
                 frc_zoom=20, fig_out=None):

    #method_colors = ['blue', 'green', 'orange', 'pink']
    method_colors =['blue', 'green', 'orange', 'red']

    default_render_args = dict(
        zoom=40, 
        clip_percentile=96, 
        scalebar_nm=2000, 
        scalebar_fontsize=20,
        intensities=False,
        transpose=False,
        sigma = 2
        )

    shared_render_args = default_render_args.copy()
    if render_args is not None:
        shared_render_args.update(render_args)    

    outdir = basedir + 'plots/'
    os.makedirs(outdir, exist_ok=True)
    print(basedir)
    
    # Load SIMFLUX-HD dataset
    intensities_fn = basedir + 'sf_intensities.npy'
    intensities_ds = SFDataset.load(intensities_fn)
    intensities_ds['pixelsize'] = pixelsize
    intensities_ds.crlb_filter(0.15)
    print(f'nn model: average background: {intensities_ds.background.mean():.1f} photons/pixel')
    print(f'#frames: {intensities_ds.numFrames}')

    #sfhd_tmp = intensities_ds[:]
    #sfhd_tmp.crlb_filter(0.1)
    #sfhd_tmp.save(basedir + 'sfhd.hdf5')

    smlm_fn = basedir + 'smlm.npy'
    smlm_ds = SFDataset.load(smlm_fn)
    print(f'smlm average spots/frame: {len(smlm_ds)/smlm_ds.numFrames:.1f}')
    print(f'average bg (summed): {smlm_ds.background.mean():.1f} photons/pixel')
    print(f'average bg (intensity fits): {smlm_ds.ibg[:,:,1].mean():.1f} photons/pixel')
    smlm_ds['pixelsize'] = pixelsize
    smlm_drift = get_drift(smlm_ds, smlm_fn, 0.2)

    # Load pattern
    #mp = ModulationPattern.load(basedir + 'mp_dl.pickle')
    #mp.print_info()
    #mp.phase_drift_estimation_error()

    decode_fn = basedir + 'decode-summed6.hdf5'
    decode_ds = Dataset.load(decode_fn)
    decode_ds['pixelsize'] = pixelsize
    decode_ds.pos[:,:2] -= 0.5

    #decode_tmp=decode_ds[:]
    #decode_tmp.crlb_filter(0.1)
    #decode_tmp.save(basedir + 'decode-summed3-crlb_filter01.hdf5')

    sf_fn = basedir + 'sf.hdf5'
    sf_ds = Dataset.load(sf_fn)
    sf_ds['pixelsize'] = pixelsize

    sf_intensities_fn = basedir + "sf_intensities.npy"
    sf_intensities_ds = SFDataset.load(sf_intensities_fn)
    print(f'intensity ds average spots/frame: {len(sf_intensities_ds)/sf_intensities_ds.numFrames:.1f}')

    sf_ndi_fn = basedir + 'sfhd-ndi.hdf5'
    sf_ndi_ds = Dataset.load(sf_ndi_fn)

    sf_combined_fn = basedir + 'sfhd-ndi+smlm.hdf5'
    sf_combined_ds = Dataset.load(sf_combined_fn)

    ds_input_filenames = [smlm_fn, sf_fn, decode_fn, sf_intensities_fn, sf_ndi_fn, sf_combined_fn]
    datasets = [smlm_ds, sf_ds, decode_ds, sf_intensities_ds, sf_ndi_ds, sf_combined_ds]
    
    for ds in datasets:
        ds['pixelsize'] = pixelsize

    print(f'intensity ds average spots/frame: {len(sf_intensities_ds)/sf_intensities_ds.numFrames:.1f}')

    dl_drift = get_drift(sf_intensities_ds, sf_intensities_fn, 0.15)
    plot_drift(outdir + 'drift_dl.png', dl_drift, pixelsize); plt.close()
    plot_drift(outdir + 'drift_dl.svg', dl_drift, pixelsize)

    # clearer filenames for publication
    ds_out_filenames = ['smlm', 'simflux', 'decode', 
                    'simcode-nn-output', 
                    'simcode-pattern-fit', 
                    'simcode-merged']

    undrifted_path = outdir + 'undrifted_ds/'
    os.makedirs(undrifted_path, exist_ok=True)

    for i in range(len(datasets)):
        datasets[i].applyDrift(dl_drift['drift'])

        if numframes is not None:
            datasets[i] = datasets[i][datasets[i].frame<numframes]

        datasets[i].save(undrifted_path + ds_out_filenames[i] + '.hdf5')

    crlb_threshold = 0.15
    frc_kwargs = dict(
        crlb_filter=crlb_threshold,
        split_ds=False,
        use_cache = False,
        zoom = frc_zoom)
    
    ds_frcs = [get_frc(ds, fn, **frc_kwargs) for ds, fn in zip(datasets, ds_input_filenames)]
    
    # plot patterns
    if True:
        mp = ModulationPattern.load(basedir + 'mp_dl.pickle')
        mod_error_threshold = 0.1
        fig,ax=plt.subplots(figsize=(8,6))
        lr = LocalizationReporter(smlm_ds, outdir, mp)
        lr.draw_pattern(0, 2, mod_error_threshold, axes=ax)
        plt.title('Intensity estimation with MLE fitting')
        plt.savefig(outdir + 'pattern_smlm.svg', dpi=300)

        fig,ax=plt.subplots(figsize=(8,6))
        lr = LocalizationReporter(intensities_ds, outdir, mp)
        lr.draw_pattern(0, 2, mod_error_threshold, axes=ax)
        plt.title('Intensity estimation with deep learning model')
        plt.savefig(outdir + 'pattern_simcode.svg', dpi=300)
        plt.close()

    # plot FRC
    fig,ax=plt.subplots(figsize=(8,6))

    frcs = [ds_frcs[0], ds_frcs[1], ds_frcs[2], ds_frcs[5]]
    
    plot_frcs(ax, frcs,
        ['SMLM',  'SIMFLUX', 'DECODE', 'SIMCODE'],
        colors=method_colors)    
    
    plt.savefig(outdir + 'frc.svg', dpi=300)
    plt.close()

    fig,ax=plt.subplots(figsize=(8,8))
    ds = sf_combined_ds[:]
    ds.crlb_filter(frc_kwargs['crlb_filter'])

    render_args = shared_render_args.copy()
    full_fov_args = render_args.copy()
    zoom = 10
    full_fov_args.update(zoom=zoom, scalebar_nm=2000)

    ds.renderFigure(axes=ax, title='Overview', **full_fov_args)
    for corner, zoom_width, plot_render_args in renders:
        roi = corner, (corner[0]+zoom_width, corner[1]+zoom_width)
        draw_rect(roi, zoom, ax)

    plt.savefig(outdir + f'results_overview.svg', dpi=300)

    # giant overview assuming two zooms
    # use gridspec to divide into a 5x4 grid, where the top 2x4 contain two large 2x2 plots
    fig = plt.figure(figsize=(14,20))
    root_gs = fig.add_gridspec(3, 1, hspace=0.1, wspace=0.1, 
                               height_ratios=[1,1,0.4])
    top_gs = root_gs[0].subgridspec(2, 4, hspace=0.2, wspace=0.1)
    mid_gs = root_gs[1].subgridspec(2, 4, hspace=0.2, wspace=0.1)

    if tif_path is not None:
        bottom_gs = root_gs[2].subgridspec(1, 3, hspace=0.1, wspace=0.2, 
                                       width_ratios=[1, 0.3, 0.3])
    else:
        bottom_gs = root_gs[2].subgridspec(1,2, hspace=0.1, wspace=0.2, 
                                       width_ratios=[1, 0.3])

    # top left contains an area to keep empty for now
    #ax = fig.add_subplot(gs[:2, :2])
    #ax.axis('off')

    zoom_colors = ['#77f', '#0f0', '#f77']
    zoom_linestyles = ['-', '--', '-.']

    # top right contains the large overview
    ax = fig.add_subplot(top_gs[:2, :2])
    ds = sf_combined_ds[:]
    ds.crlb_filter(crlb_threshold)
    ds.renderFigure(axes=ax, title='Overview', **full_fov_args)
    for i, (corner, zoom_width, plot_render_args) in enumerate(renders):
        roi = corner, (corner[0]+zoom_width, corner[1]+zoom_width)
        r = draw_rect(roi, zoom, ax)
        #r.set_edgecolor(zoom_colors[i])
        r.set_linestyle(zoom_linestyles[i])

    method_ds = [smlm_ds, sf_ds, decode_ds, sf_combined_ds]
    #method_labels = ['SMLM', 'SIMFLUX', 'DECODE', 'SIMCODE-1', 'SIMCODE-2']
    method_labels = ['SMLM', 'SIMFLUX', 'DECODE', 'SIMCODE']

    if tif_path is not None:    
        pc, rmsd, error_maps = compute_pc(tif_path, method_ds)

        # print pc for each method
        print('Pearson correlation:')
        for i in range(len(method_ds)):
            print(f'{method_labels[i]}: {pc[i]:.3f}')

        rse_ax = fig.add_subplot(bottom_gs[2])
        rse_ax.bar(method_labels, pc, color=method_colors)
        rse_ax.set_title('Pearson corr.')
        # set 45 degree rotation for x labels
        rse_ax.set_xticklabels(method_labels, rotation=45)

    def draw_zoom(gs, zoom_ix, ds, frc_val):
        color = method_colors[method_ds.index(ds)]
        title = method_labels[method_ds.index(ds)]

        ax = fig.add_subplot(gs)
        corner, zoom_width, plot_render_args = renders[zoom_ix]
        render_args = shared_render_args.copy()
        render_args.update(plot_render_args)
        roi = corner, (corner[0]+zoom_width, corner[1]+zoom_width)
        ds = ds.crop(*roi, silent=True)
        ds.crlb_filter(crlb_threshold)

        #ds.renderFigure(axes=ax, title=f'{title} [Zoom {zoom_ix+1}]\nFRC: {frc_val:.1f} nm', **render_args)
        ds.renderFigure(axes=ax, title=f'{title}', **render_args)

        # add colored rectangle around the zoom axes
        rect = plt.Rectangle((0,0), 1, 1, fill=False, 
                             edgecolor=color,#zoom_colors[zoom_ix],
                             ls=zoom_linestyles[zoom_ix],
                             lw=6, clip_on=False, transform=ax.transAxes)
        ax.add_patch(rect)

    # zoom 0
    draw_zoom(top_gs[0, 2], 0, smlm_ds, frcs[0])
    draw_zoom(top_gs[0, 3], 0, sf_ds, frcs[1])
    draw_zoom(top_gs[1, 2], 0, decode_ds, frcs[2])
    draw_zoom(top_gs[1, 3], 0, sf_combined_ds, frcs[3])

    # zoom 1
    draw_zoom(mid_gs[0, 2], 1, smlm_ds, frcs[0])
    draw_zoom(mid_gs[0, 3], 1, sf_ds, frcs[1])
    draw_zoom(mid_gs[1, 2], 1, decode_ds, frcs[2])
    draw_zoom(mid_gs[1, 3], 1, sf_combined_ds, frcs[3])

    #plt.tight_layout()
    if True:
        frc_ax = fig.add_subplot(bottom_gs[0])
        plot_frcs(frc_ax,frcs, 
            labels = method_labels,
            colors = method_colors)
        frc_ax.set_title('FRC')
        # disable y axis
        frc_ax.get_yaxis().set_visible(False)

        # disable legend
        #frc_ax.legend().set_visible(False)

        # bar plot with spot counts
        ax = fig.add_subplot(bottom_gs[1])

        spot_counts = np.array([len(ds.crlb_filter(crlb_threshold, inplace=False)) 
                                for ds in method_ds]) * 1e-6
        ax.bar(method_labels, spot_counts, color=method_colors)

        # rotate x labels
        ax.set_xticklabels(method_labels, rotation=45)

        ax.set_xticks(range(5))
        ax.set_ylabel('Spot count (${10}^6$)')
        ax.set_title('# Localizations')
        ax.set_ylim([0, 4])

    plt.savefig(outdir + 'fig_exp.svg', dpi=300)
    if fig_out is not None:  # [added] save location of the reproduction
        os.makedirs(os.path.dirname(fig_out), exist_ok=True)
        plt.savefig(fig_out + '.svg', dpi=300)
        plt.savefig(fig_out + '.png', dpi=150)

    mp_dl=ModulationPattern.load(basedir + 'mp_dl.pickle')
    mp_sf=ModulationPattern.load(basedir + 'mp_sf.pickle')
    mp_smlm=ModulationPattern.load(basedir + 'mp_smlm.pickle')
    mp_ndi=ModulationPattern.load(basedir + 'mp_ndi.pickle')

    with open(basedir+'config.pickle','rb') as f:
        cfg = pickle.load(f)

    # return all the datasets as a dict
    r = dict(
        cfg=cfg,
        smlm=smlm_ds,
        sf=datasets[1],
        decode=decode_ds,
        sf_intensities=sf_intensities_ds,
        sf_ndi=datasets[4],
        sf_combined=datasets[5],
        mp_dl=mp_dl, # pattern produced by pattern estimation on DL data
        mp_ndi=mp_ndi, # mp_dl, but phases averaged to optimize for fitting
        mp_sf=mp_sf, # pattern used for SIMFLUX fits 
        mp_smlm=mp_smlm # pattern produced from SMLM data
    )

    if tif_path is not None:
        r['pearson'] = [pc, rmsd, error_maps]   

    # [added] the plotted numbers
    r['values'] = dict(labels=method_labels, frc_nm=[f.frc for f in frcs], spot_counts=(spot_counts*1e6).astype(int),
                       pearson=pc if tif_path is not None else None, rmsd=rmsd if tif_path is not None else None,
                       n_frames=[int(d.numFrames) for d in method_ds], frame_max=[int(d.frame.max()) for d in method_ds],
                       frc_all=[f.frc for f in ds_frcs], frc_all_labels=ds_out_filenames,
                       drift_ptp_nm=(np.ptp(dl_drift['drift'], 0)*pixelsize).tolist())
    r['dl_drift'] = dl_drift
    r['frcs'] = ds_frcs

    return r


if __name__ == '__main__':
    import json
    k = int(sys.argv[1])
    cfg = FIGS[k]
    r = plot_results(**cfg)
    vals = {kk: (np.asarray(v).tolist() if not isinstance(v, (int, float)) else v) for kk, v in r['values'].items()}
    print(json.dumps(vals, indent=1))
    json.dump(vals, open(cfg['fig_out'] + '_values.json', 'w'), indent=1)
