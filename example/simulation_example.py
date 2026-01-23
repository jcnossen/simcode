#%%

import smlmtorch.simflux.pattern_estimator as pe
import numpy as np
import matplotlib.pyplot as plt
from smlmtorch.util.multipart_tiff  import MultipartTiffSaver, tiff_read_all
from smlmtorch import Dataset
from smlmtorch.simflux.simulate import simulate, angles_to_mod
from smlmtorch.simflux.localizer import SFLocalizer
import tqdm

from fastpsf import GaussianPSFMethods, Context

from smlmtorch.util.generate_tubules import generate_microtubule_points

import torch
print(torch.__version__)

#%%

np.random.seed(0)
# Generate ground truth microtubule points
W = 50
pixelsize = 100 #nm

pts = generate_microtubule_points(W, depth=0, numtubules = 20, spl_knots=5,
                linedensity=10, nudge_factor=0.1, margin=0.05, spl_degree=2,
                tube_radius=12.5/pixelsize)

gt_ds = Dataset(len(pts), 2, [W,W], pixelsize=pixelsize)
gt_ds.pos = pts[:,:2]
gt_ds.save('ground-truth-points.hdf5')

plt.figure();
plt.scatter(pts[:,0],pts[:,1],s=1);
# invert bottom/top to match imshow
plt.gca().invert_yaxis();
plt.xlabel('Pixels'); plt.ylabel('Pixels');
plt.title("Simulated microtubules - ground truth")

#%%

# Simulate an SMLM recording with modulated illumination
on_fraction = 0.005 # controls density
avg_on_time = 6
nframes = 30000 # +50k frames gives a stable FRC result, shorter will impact FRC based improvement
intensity = 500
psf_sigma = 1.3
psf_calib = [psf_sigma, psf_sigma] # XY
roisize = 10

background = 4 # approximately dna paint background on our dmd setup
path = 'data/movie.tif'

pixelsize = 100
pattern_frames = np.array([  # Defines which frame indices have which modulation pattern
    [0,1,2], # X pattern frame indices
    [3,4,5]  # Y pattern frame indices
])
modulation_depth = 0.90
modulation_angles_deg = [0, 90]
pitch_nm = 220
mod = angles_to_mod([pitch_nm, pitch_nm], pixelsize, modulation_angles_deg, modulation_depth, pattern_frames)
mp_gt = pe.ModulationPattern(pattern_frames, mod)
mov = None

#%%

with Context() as ctx:
  psf = GaussianPSFMethods(ctx).CreatePSF_XYIBg(roisize, psf_calib, cuda=True)
  mov = simulate(path, mp_gt, psf,
      pts[:,:2], numframes=nframes, intensity=intensity, width=W,
      bg=background, avg_on_time=avg_on_time, on_fraction=on_fraction, return_movie=True)

#%%

if mov is None:
   mov = tiff_read_all(path)

# Show the first few frames:
frames = mov[:6]
fig,ax=plt.subplots(1,len(frames),figsize=(15,5))
for i in range(len(frames)):
    ax[i].imshow(frames[i])
    ax[i].axis('off')
    ax[i].set_aspect('equal')
    ax[i].set_title(f'frame {i}')
plt.tight_layout()
plt.figure()
plt.imshow(mov.mean(0)); plt.title('Measurement average')
plt.axis('off')

#%%

# For comparison, we detect spots in a single emitter pipeline first, and run both SMLM and SIMFLUX fitters on the found ROIs

# Change this when you're processing experimental data:
camera_gain = 1
camera_offset = 0

localizer = SFLocalizer(path,
   psf_calib = [psf_sigma, psf_sigma],
   roisize = roisize,
   detection_threshold = 5,
   pattern_frames= pattern_frames,
   gain = camera_gain,
   offset = camera_offset,
   pixelsize = pixelsize,
   zrange = [0,0], # unsupported
   psf_sigma_binsize = None, # if non-zero, it will estimate the PSF width over time with given bin size (in spot counts)
   result_dir='results',
   device='cuda:0')

localizer.detect_spots(ignore_cache=False, moving_window=True)
smlm_ds = localizer.fit_smlm(max_crlb_xy=None, ignore_cache=False)
print(f"numrois: {localizer.numrois}. #summed_fits: {localizer.summed_fits.shape[0]}, numframes: {smlm_ds.numFrames}")

fig,ax = plt.subplots(figsize=(8,8))
smlm_ds.renderFigure(axes=ax,zoom=10,clip_percentile=98)

#%%

# Estimate patterns so we can run simflux fitting.
# Note that in our method, optical setup modulation depths are highly underestimated in high density scenes,
# so in practice we fix based on our estimates from low density measurements
def show_fourier_peaks(ft_img, ix):
   plt.figure()
   plt.imshow(ft_img)
   plt.title(f'Fourier transform of axis {ix}')
mp_est = localizer.estimate_angles(pitch_minmax_nm=[100,300], fft_img_cb=show_fourier_peaks)

#%%
mp_est = localizer.estimate_phases(mp_est, spots_per_bin=5000,
                                accept_percentile=40, iterations=10, verbose=False)
mp_est.depths[:]=0.9
# we also assume constant phase steps over time
mp_est = mp_est.const_phase_offsets()


# filter based on modulation error:
# we calculate the expected intensities based on the SMLM position and our patterns, if too different then they are likely unreliable for a pattern based fit
me = mp_est.mod_error(smlm_ds)
me_sel = me < 0.1

sf_ds = localizer.fit_simflux(mp_est, smlm_ds[me_sel], iterations=50, lambda_=500, ignore_cache=True, normalizeWeights=True, distFromSMLM=0.5)
#lr.scatterplot([ sfloc.sum_ds, sf_ds ], connected=False, labels=['SMLM', 'SF'], limits=None, s=2)

#%%

# Now run the simcode pipeline, first import all the packages
from smlmtorch.util.config_dict import config_dict # dictionary indexable like object
from smlmtorch.nn.sf_model import SFLocalizationModel
from smlmtorch.nn.localize_movie import MovieProcessor
from smlmtorch.simflux.dataset import SFDataset
import torch
import zipfile
import os

simcode_model_class = SFLocalizationModel

simcode_model_zip_path = os.path.split(__file__)[0] + '/../model/model_weights_sf_conv_g1.3_tio2_L64_2.zip'
simcode_model_path = os.path.split(simcode_model_zip_path)[0] + '/sf_conv_g1.3_tio2_L64_2'

# Extract the zip file
with zipfile.ZipFile(simcode_model_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(simcode_model_zip_path))

#%%

config = config_dict.load(simcode_model_path + '/config.yaml')
config = config_dict(**config, # Need to add some inference related parameters
  detector=dict( prob_threshold=0.7, use_prob_weights=False )
)
config.model.num_intensities = pattern_frames.size

mp = MovieProcessor(simcode_model_class, config, simcode_model_path+"/checkpoint_1.pt", device='cuda:0')
# the model processes the data in batches of moving windows of frames [B, L, H, W], where L is the window size (6)
simcode_nn_ds = mp.process(path, batch_size=32,
                           batch_overlap = pattern_frames.size-1,
                           gain = camera_gain,
                           offset = camera_offset)
simcode_nn_ds.crlb_filter(0.3) # reduce the size a bit, it generates a lot of low precision spots
simcode_nn_ds['pixelsize'] = pixelsize

result_dir = localizer.result_dir
# Saving in .npy will keep track of per-pattern spot intensities.
simcode_nn_ds.save(result_dir+'/simcode-nn.npy')
# Saving in hdf5 is Picasso compatible, but will lose per-pattern spot intensities,
# and only store a single intensity per localization
simcode_nn_ds.save(result_dir+'/simcode-nn.hdf5')

#%%

# Now we have the neural network output, we can run the pattern fitting on it.

# Run pattern fits on NN outputs
mp_sc = pe.estimate_angles(pitch_minmax_nm=[150,500], ds=simcode_nn_ds,
                        pattern_frames = pattern_frames, result_dir = localizer.result_dir,
                        device = localizer.device, moving_window = localizer.moving_window)

mp_sc = pe.estimate_phases(simcode_nn_ds, mp_sc, spots_per_bin=10000,
                        accept_percentile=50, iterations=1, verbose=False, device=localizer.device)
mod_pattern_nn = mp_sc.const_phase_offsets()
# Plot phases
fig,ax=plt.subplots(2,1,figsize=(6,4))
mp_sc.plot_phase_drift(nframes=10000,ax=ax, label='Pat {0}', linestyle='-', lw=1)

mod_error_threshold = 0.1
print(f'computing modulation-enhanced positions using mod error threshold = {mod_error_threshold}')
moderr_selected = mod_pattern_nn.mod_error(simcode_nn_ds) < mod_error_threshold

ndi_input_ds = simcode_nn_ds[moderr_selected]
simcode_pattern_fit_ds = pe.ndi_fit_dataset(ndi_input_ds, mod_pattern_nn, device=localizer.device)

# Filter outliers based on the distance from the non pattern-fitted positions
max_dist = 0.2
dist = np.sqrt ( ( (simcode_pattern_fit_ds.pos[:,:2] - ndi_input_ds.pos[:,:2])**2 ).sum(1) )
# Note that our Dataset class supports a mask indexing operation ds[boolean mask]
simcode_pattern_fit_ds = simcode_pattern_fit_ds[dist<max_dist]
simcode_pattern_fit_ds.save(localizer.result_dir + "simcode-pattern-fitted.hdf5")

print(f'remaining after filtering by max distance from original: {len(simcode_pattern_fit_ds)}/{moderr_selected.sum()}')

simcode_merged_ds = pe.merge_estimates(simcode_pattern_fit_ds, ndi_input_ds[dist<max_dist])
simcode_merged_ds.save(result_dir+'/simcode.hdf5')

#%%

fig,ax=plt.subplots(3,2,figsize=(6,10))
gt_ds.renderFigure(axes=ax[0,0],clip_percentile=90,zoom=10,title='Ground truth')
ax[0,1].imshow(mov.mean(0)); ax[0,1].set_title('Measurement average')
# disable axes
ax[0,1].axis('off')
smlm_ds.renderFigure(axes=ax[1,0],zoom=10,clip_percentile=98,title='SMLM')
sf_ds.renderFigure(axes=ax[1,1],clip_percentile=98,zoom=10,title='SIMFLUX')

# SIMCODE (Intermediate) here indicates the neural network output,
# where localization is done by NN but a pattern based fit is not done yet.
simcode_nn_ds.renderFigure(axes=ax[2,0],clip_percentile=98,zoom=10,title='SIMCODE (Int.)')
ax[2,1].axis('off');
simcode_merged_ds.renderFigure(axes=ax[2,1],clip_percentile=96,zoom=10,title='SIMCODE')
#plt.tight_layout()

datasets = [smlm_ds, sf_ds, simcode_nn_ds, simcode_merged_ds]
ds_labels = ['SMLM', 'SIMFLUX', 'SIMCODE (Int.)', 'SIMCODE']

smooth_curve=16
frcs = []

for i in range(len(datasets)):
  ds = datasets[i]
  ds_label = ds_labels[i]
  ds_frc = ds[ds.frame%6==0] # have statistically independent frames, cannot use overlapping sets of 6 frames.

  frc_val, frc_curve, frc_freq = ds_frc.frc(display=False, zoom=20, smooth=smooth_curve, mask=None)
  d = config_dict(frc=frc_val, curve=frc_curve, freq=frc_freq)
  frcs.append(d)

fig,ax=plt.subplots()
crop_end = 100

ax.axhline(1/7, color='grey', linestyle='--')
for i in range(len(frcs)):
  l=ax.plot(frcs[i].freq[:-crop_end], frcs[i].curve[:-crop_end], label=f'{ds_labels[i]} (FRC={frcs[i].frc:.1f} nm)')

ax.legend(fontsize=13)
ax.set_ylabel('FRC')
ax.set_xlabel('Spatial freq. [nm^-1]')
ax.set_title('Fourier ring correlation')

