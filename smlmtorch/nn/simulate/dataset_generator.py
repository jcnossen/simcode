#%%

import numpy as np
import torch
from scipy.special import erf
from smlmtorch.nn.simulate.gaussian_model import gaussian2D, Gaussian2DModel
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt
from smlmtorch.nn.utils.blink_spots import sample_spot_frames
from smlmtorch import config_dict
from smlmtorch.nn.simulate.psf import Gaussian2DPSF, CubicSplinePSF
from smlmtorch.simflux.pattern_estimator import ModulationPattern
from smlmtorch import progbar

import scipy.stats

def _unpack(x, dim):
    for i in range(x.shape[dim]):
        yield x.select(dim, i)

def compute_on_off(img_shape, num_frames,
    pixelsize_nm, # to allow conversion of density into k_on / k_off
    density_um2, # 
    mean_on_time):

    npixels = img_shape[0] * img_shape[1]
    avg_count = density_um2 * npixels * (pixelsize_nm / 1000)**2
    # how many spots are needed to get avg_count on average in num_frames:
    # initial plus the ones that turn off during the sequence
    # plus some extra margin to compensate for the fact that spots can only turn on once
    num_spots = avg_count * (1 + num_frames / mean_on_time) * 1.5

    k_off = 1 / mean_on_time
    #steady state k_on:
    # (num_spots-avg_count) * k_on = avg_count * k_off
    k_on = avg_count * k_off / (num_spots-avg_count)

    print(f'k_on: {k_on:.3f} k_off: {k_off:.2f} num_spots: {int(num_spots)}. avg_count: {avg_count:.1f}')
    return k_on, k_off, int(num_spots), int(avg_count * 2)


def log_normal_parameters(mean, mode):
    from scipy.optimize import fsolve
    # Equations to solve for mu and sigma
    def equations(p):
        mu, sigma = p
        return (mean - np.exp(mu + sigma**2 / 2), mode - np.exp(mu - sigma**2))
    
    mu_guess = np.log(mean)
    sigma_guess = 1
    mu, sigma = fsolve(equations, (mu_guess, sigma_guess))
    
    return mu, sigma

def generate_log_normal_samples(mean, mode, n_samples):
    mu, sigma = log_normal_parameters(mean, mode)
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
    return samples


def _generate_spots(num_movies, num_frames, num_spots, k_on, k_off, pos_min, pos_max, 
                    intensity_distr = 'uniform',
                    intensity_mode = 400, intensity_mean = 1000,
                intensity_mean_min=50, intensity_mean_max=3000, intensity_fluctuation_std=0.1, subframes=10,  intensity_sigma=None):

    t_start, t_end = sample_spot_frames(num_movies * num_spots, k_on/subframes, k_off/subframes, num_frames*subframes)
    t_start = t_start.reshape(num_movies, num_spots)[:,None].repeat(1, num_frames, 1) / subframes
    t_end = t_end.reshape(num_movies, num_spots)[:,None].repeat(1, num_frames, 1) / subframes

    pos = torch.rand(num_movies, num_spots, 3) * (pos_max - pos_min) + pos_min
    pos = pos[:,None].repeat(1, num_frames, 1,1)

    if intensity_distr == 'uniform':
        spot_intensity = torch.rand(num_movies, num_spots) * (intensity_mean_max-intensity_mean_min) + intensity_mean_min
    elif intensity_distr == 'log-normal':
        spot_intensity = torch.tensor(generate_log_normal_samples(intensity_mean, intensity_mode, num_movies * num_spots)).reshape(num_movies, num_spots)
        spot_intensity = torch.clamp(spot_intensity, intensity_mean_min, intensity_mean_max)
    elif intensity_distr == 'normal':
        spot_intensity = torch.randn(num_movies, num_spots) * intensity_sigma + intensity_mean
        spot_intensity = torch.clamp(spot_intensity, intensity_mean_min, intensity_mean_max)
    else:
        raise ValueError(f'Unknown intensity distribution: {intensity_distr}')

    spot_intensity = (spot_intensity[:,None].repeat(1, num_frames, 1) * 
                    (torch.randn(num_movies, num_frames, num_spots) * intensity_fluctuation_std + 1))
    spot_intensity = torch.clamp_min(spot_intensity, 0)  # clamp

    # Calculate active start and end times for each frame
    frametime = torch.arange(num_frames)[None,:,None]
    active_start_times = torch.max(t_start, frametime)
    active_end_times = torch.min(t_end, frametime + 1)
    ontime = (active_end_times - active_start_times).clamp(min=0)
    spot_intensity *= ontime
    active = ((ontime>0) & (spot_intensity>0)).float() 

    spots = torch.stack((active, spot_intensity, *_unpack(pos,-1), t_start, t_end),-1)
    return spots.type(torch.float32)


class SpotDataset(Dataset):
    def __init__(self, 
            spots,
            max_spots = 40,
            track_intensities = 0, # for SIMFLUX models that output a set of intensities
            track_intensities_offset = 0, 
            bg_min = 1,
            bg_max = 20,
            render_args = None):
        
        self.num_movies = spots.shape[0]
        self.num_frames = spots.shape[1]
        num_spots = spots.shape[2]
        self.render_args = render_args
        self.max_spots = max_spots
        self.track_intensities = track_intensities
        self.track_intensities_offset = track_intensities_offset
        self.bg_min = bg_min
        self.bg_max = bg_max

        # make frame numbers relative
        frame_ix = torch.arange(self.num_frames, device=spots.device)[None,:,None]
        spots[...,5] -= frame_ix
        spots[...,6] -= frame_ix

        frame_bg = torch.rand(size=(self.num_movies, self.num_frames)) * (bg_max - bg_min) + bg_min
        spots = torch.cat( (spots, frame_bg[:,:,None,None].expand(-1,-1,num_spots,1)), -1)

        self.spots = spots
        self.update_tracked_intensities()
        self.apply_max_spots()

    def update_tracked_intensities(self):
        if self.track_intensities is not None and self.track_intensities > 0:
            # move intensities to last dim, like other spot parameters
            frame_ix = (torch.arange(self.num_frames)[:,None] - self.track_intensities_offset + torch.arange(self.track_intensities))# - self.track_intensities + 1)
            mask = (frame_ix >= 0) & (frame_ix < self.num_frames)
            frame_ix *= mask  # set invalid indices to 0 so it doens't fail
            intensities = self.spots[:,frame_ix,:,1]
            intensities[:,~mask] = 0
            self.intensity_history = intensities.permute(0,1,3,2) # move intensities to last dim, like other spot parameters
        else:
            self.intensity_history = None

    def apply_max_spots(self):
        if self.max_spots < self.spots.shape[2]:
            # sort by active state, moving all the active spots in each frame to the start of the list
            sorted_indices = torch.argsort(self.spots[...,0], dim=2, descending=True)
            sorted_spots = torch.gather(self.spots, 2, sorted_indices[..., None].expand(-1, -1, -1, self.spots.shape[-1]))

            self.spots_capped = sorted_spots[:,:,:self.max_spots]
            if self.intensity_history is not None:
                intensity_history = torch.gather(self.intensity_history, 2, sorted_indices[...,None].expand(-1,-1,-1, self.track_intensities))
                self.intensity_history_capped = intensity_history[:,:,:self.max_spots]
        else:
            self.spots_capped = self.spots
            self.intensity_history_capped = self.intensity_history

        
    @property
    def pos(self):
        return self.spots[...,2:5].cpu().numpy()

    @property
    def num_spots_per_frame(self):
        return self.spots[:, :, :, 0].sum(dim=2).cpu().numpy()

    def __len__(self):
        return self.num_movies

    def __getitem__(self, idx):
        """
        Output format:
        [frames, max_spots, 8]
        Last col: [active, intensity, x, y, z, t_start, t_end, bg]
        """
        if self.track_intensities:
            return dict(
                spots = self.spots_capped[idx], 
                intensities = self.intensity_history_capped[idx], 
                index=idx)
        return dict(spots = self.spots_capped[idx], index=idx)


class RandomSpotDataset(SpotDataset):
    def __init__(self, 
        num_movies, num_frames,
        img_shape = (32,32),
        density_um2 = 1,
        pixelsize_nm = 100,
        mean_on_time = 6,
        k_on = None,
        k_off = None,
        num_spots = None, 
        max_spots = 30,
        z_range = (0, 0),
        intensity_mean_max = 5000,
        intensity_mean_min = 200,
        intensity_fluctuation_std = 0.1,
        intensity_distr = 'uniform',
        intensity_mode = 300,
        intensity_mean = 600,
        margin = 0,
        subframe_blinking = 10,
        **kwargs):    
    
        self.height, self.width = img_shape
        self.num_movies = num_movies
        self.num_frames = num_frames

        if k_on is None:
            k_on, k_off, num_spots, max_spots = compute_on_off(img_shape, num_frames,
                pixelsize_nm = pixelsize_nm, # to allow conversion of density into k_on / k_off
                density_um2 = density_um2, #
                mean_on_time = mean_on_time)

        self.num_spots = num_spots
        self.max_spots = max_spots
        self.k_on = k_on
        self.k_off = k_off

        self.z_range = z_range
        self.intensity_mean_max = intensity_mean_max
        self.intensity_mean_min = intensity_mean_min
        self.intensity_fluctuation_std = intensity_fluctuation_std
        self.margin = margin
        self.subframe_blinking = subframe_blinking
        self.height, self.width = img_shape

        # [active, intensity, x, y, z, t_start, t_end, bg]
        pos_min = torch.tensor([-margin, -margin, z_range[0]], dtype=torch.float32)
        pos_max = torch.tensor([self.width+margin, self.height+margin, z_range[1]], dtype=torch.float32)
        spots = _generate_spots(num_movies, num_frames, num_spots, k_on, k_off, pos_min, pos_max,
            intensity_distr, intensity_mode, intensity_mean,
            intensity_mean_min, intensity_mean_max, intensity_fluctuation_std, subframe_blinking)

        super().__init__(spots, **kwargs)

 
class SimulatedSMLMDataset(Dataset):
    def __init__(self, psf, render_device='cpu', only_crlb=False, use_jit=False, render_args=None):
        self.device = render_device
        self.height, self.width = psf.shape

        if render_args is not None:
            self.read_noise_mean = render_args.read_noise_mean
            self.read_noise_std = render_args.read_noise_std
        else:
            self.read_noise_mean = 0
            self.read_noise_std = 0

        self.psf = psf

        if use_jit:
            self._psf_module_jit = torch.jit.script(psf)
        else:
            self._psf_module_jit = psf

    def render(self, spots_ds, batch_size=32, compute_crlb=False, use_tqdm = True):
        num_movies, self.num_frames, self.max_spots, _ = spots_ds.spots.shape

        # Preallocate all the memory
        self.spots_ds = spots_ds
        self.data = torch.zeros((num_movies, self.num_frames, self.height, self.width), dtype=torch.float32)
        self.readnoise = torch.zeros((num_movies, self.height, self.width), dtype=torch.float32)

        if compute_crlb:
            self.crlb = torch.zeros((num_movies, self.num_frames, self.max_spots, 5), dtype=torch.float32)
        else:
            self.crlb = None

        # use pbar
        if use_tqdm:
            pbar = progbar(total=num_movies, miniters=10)
        else:
            pbar = None

        dl = DataLoader(spots_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        #batches = torch.split(torch.arange(len(spots)), batch_size, dim=0)

        for batch in dl:
            index = batch['index']
            batch_spots = batch['spots']
            self.data[index], self.readnoise[index] = self.process_labels(batch_spots)

            if compute_crlb:
                self.crlb[index] = self.compute_crlb(batch_spots)

            if pbar is not None:
                pbar.set_description('Rendering frames...')
                pbar.update(len(batch_spots))
        if pbar is not None:
            pbar.close()

    def _psf_params(self, labels):
        # labels [frames, max_spots, 8]
        # [active, x, y, z, intensity, 0]
        batchsize, num_frames, max_spots, num_features = labels.shape
        params_ = labels.view(-1, num_features)[:, [0, 2, 3, 4, 1]]
        paramsWithBg = torch.zeros((len(params_), 6), dtype=torch.float32, device=self.device)
        paramsWithBg[:, :5] = params_
        return paramsWithBg

    def compute_crlb(self, labels):
        batchsize, num_frames, max_spots, num_features = labels.shape

        paramsWithBg = self._psf_params(labels)
        # Set bg to real values for CRLB calculation
        paramsWithBg[:,5] = labels[:,:,:,7].flatten()
        # Add nonzero intensity to not make crlb calcs get NaNs 
        paramsWithBg[:,4] = torch.clamp_min(paramsWithBg[:,4],1)
        crlb = torch.zeros_like(paramsWithBg[:,1:])
        active = paramsWithBg[:,0] > 0
        crlb[active] = self.psf.crlb(paramsWithBg[active,1:])[1]
        crlb = crlb.view(batchsize, num_frames, max_spots, 5)
        # crlb format: [frames, max_spots, 5], 
        crlb = crlb[:,:,:, [3,0,1,2,4]]
        # last column is: [N, x, y, z, bg]
        return crlb.cpu()

    def process_labels(self, labels):
        # labels [frames, max_spots, 8]
        # [active, x, y, z, intensity, 0]
        batchsize, num_frames, max_spots, num_features = labels.shape
        params_ = labels.view(-1, num_features)[:, [0, 2, 3, 4, 1]]
        bg = labels[:,:,0,7,None,None].to(self.device)

        paramsWithBg = torch.zeros((len(params_), 6), dtype=torch.float32, device=self.device)
        paramsWithBg[:, :5] = params_
        paramsWithBgZero = paramsWithBg.clone()

        # Render PSFs with bg=0
        psfs = paramsWithBg[:,0, None, None] * self.psf(paramsWithBgZero[:,1:])

        frames = psfs.view(batchsize, num_frames, max_spots, self.height, self.width).sum(dim=2)
        frames = torch.clamp(frames + bg, min=0)
        samples = torch.poisson(frames)
        readnoise = torch.clip(self.read_noise_mean + 
            self.read_noise_std * torch.randn((batchsize, self.height, self.width), dtype=torch.float32, device=self.device), 0)
        samples += readnoise[:, None] * torch.randn((batchsize, num_frames, self.height, self.width), dtype=torch.float32, device=self.device)
        return torch.clip(samples,0).cpu(), readnoise.cpu()
    
    def draw_frame(self, movie_ix, frame_ix):
        fig, ax = plt.subplots()
        ax.imshow(self.data[movie_ix, frame_ix])
        ax.set_title(f'Frame {frame_ix}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        spots = self.spots_ds[movie_ix]['spots']
        on_targets = spots[frame_ix, :, 0] > 0
        # draw on_targets
        ax.scatter(spots[frame_ix, on_targets, 2], 
                spots[frame_ix, on_targets, 3], c='r', s=10)
        plt.show()

    def __len__(self):
        return len(self.spots_ds)
    
    def __getitem__(self, idx):
        labels = self.spots_ds[idx]

        if self.crlb is not None:
            labels = dict(**labels, crlb=self.crlb[idx])

        return self.data[idx], self.readnoise[idx], labels


class SMLMDataGenerator:
    def __init__(self, psf, img_shape, device, spot_ds_type = RandomSpotDataset, render_batch_size=8, **config):
        self.device = device
        #self.use_jit = config.psf.use_jit if 'use_jit' in config.psf else True
        if psf.type == 'Gaussian2D':
            self.psf = Gaussian2DPSF(psf.sigma, img_shape)
            self.use_jit = True
        elif psf.type == 'CubicSpline':
            self.psf = CubicSplinePSF(psf.calib_file, img_shape)
            self.use_jit = False
        else:
            assert False, f'PSF type {psf.type} not supported'
        self.render_batch_size = render_batch_size
        self.config = dict(**config, img_shape=img_shape)
        self.spot_ds_type = spot_ds_type

    def generate(self, size):
        labels_ds = self.spot_ds_type(size, **self.config)
        ds = SimulatedSMLMDataset(self.psf, render_device=self.device, use_jit=self.use_jit)
        ds.render(labels_ds, batch_size = self.render_batch_size)
        return ds

    def forward(self, size):
        return self.generate(size)

def _render_example():
    import matplotlib.pyplot as plt
    import numpy as np

    frame_size = 32
    config = config_dict(
        num_frames = 12,
        img_shape = (frame_size, frame_size),
        num_spots = 50,
        max_spots = 40,
        k_on = 0.1,
        k_off = 0.05,
        intensity_mean_min = 200,
        intensity_mean_max = 10000,
        intensity_fluctuation_std = 0.5, # I_frame = I_spot * (1 + fluctuation * randn())
        #intensity_min = 0,
        bg_min = 0.5,
        bg_max = 20,
        render_args = dict(
            read_noise_mean = 2,
            read_noise_std = 1
        )
    )

    spots_ds = RandomSpotDataset(1024*2, **config)

    psf = Gaussian2DPSF(1.5, [frame_size,frame_size])

    # Create the SimulatedSMLMDataset
    simulated_smlm_dataset = SimulatedSMLMDataset(psf, 'cuda')
    simulated_smlm_dataset.render(spots_ds, batch_size=64, compute_crlb=False)

    # Visualize a specific frame
    movie_idx = 0
    frame_idx = 0

    frame, camera_calib, spots_ds = simulated_smlm_dataset[movie_idx]
    spots = spots_ds['spots']
    plt.imshow(frame[frame_idx])
    plt.scatter(spots[frame_idx, spots[frame_idx,:,0]>0, 2], spots[frame_idx, spots[frame_idx,:,0]>0, 3], c='r', s=10)

    # add text at the points to indicate the start and end frame (indices 5 and 6):
    for i in range(spots.shape[1]):
        if spots[frame_idx, i, 0] > 0:
            plt.text(spots[frame_idx, i, 2], spots[frame_idx, i, 3], f'{spots[frame_idx, i, 5]:.0f}-{spots[frame_idx, i, 6]:.0f}', color='w')

    plt.show()

if __name__ == "__main__":
    #_render_example()

    spots = _generate_spots(2, 8, 3, 0.99, 0.01, torch.tensor([5,5,0]), 
        torch.tensor([5,5,0]), 
        intensity_distr='log-normal',
        intensity_mode=200,
        intensity_mean=500, 
        intensity_mean_min=30,
        intensity_mean_max=2000,
        intensity_fluctuation_std=0.5,
        subframes=10)

    active = spots[:,:,:,0]>0
    intensities = spots[:,:,:,1][active]

    plt.figure()
    plt.hist(intensities, bins=100)

    # spots shape:
    # [num_movies, num_frames, num_spots, 8]
    # [active, intensity, x, y, z, t_start, t_end, bg]
    spots[0,:,0,1]=torch.arange(8)*100+100

    # render
    
    spots_ds = SpotDataset(spots, 
        track_intensities=6, 
        track_intensities_offset=2)
    print(spots_ds.intensity_history)
    

    #ds = SimulatedSMLMDataset(psf)
    #ds.render(, batch_size=1, compute_crlb=False)
    #plt.figure()
    #plt.imshow(ds.data[0,0])
    
