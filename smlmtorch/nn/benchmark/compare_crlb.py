"""
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from smlmtorch.nn.utils.batch_utils import batch_cat, move_to_device, batch_dict_index
from smlmtorch.nn.simulate import SimulatedSMLMDataset
from smlmtorch import config_dict
from smlmtorch.nn.simulate.dataset_generator import SpotDataset
from smlmtorch import pb_range, progbar

class CRLBPlotGenerator:
    def __init__(self, model, num_samples, psf, sim_config, 
            param_list = ['x', 'y', 'N', 'bg'],
            psf_param_mapping = ['x', 'y', 'N', 'bg'],
                    device=None, background = 2):
        """
        param_list: which of the model.params are plotted
        crlb_mapping: for each item in param_list, which of the PSF param's crlb have to be plotted with it
        """

        self.model = model
        self.param_list = param_list
        self.param_ix = np.array([model.param_names.index(p) for p in param_list])
        self.psf_param_mapping = psf_param_mapping
        self.device = device

        self.numsamples  = num_samples 
        self.psf = psf

        self.sim_config = sim_config
        self.background = background

    def process(self, writer, step, data, model_output, targets):
        if self.photon_range is not None:
            self.plot_photon_range2(self.numsamples, sim_config=self.sim_config,
                log_writer = writer, log_step = step, 
                background = self.background, show_plot=False)

    def estimate_precision(self, params, n_frames, batchsize=8):
        """
        params: [batchsize, 5], 
        last axis being [N, x, y, z, bg]
        """
        params_ = params[:,[1,2,3,0,4]]
        params_crlb = params_*1
        params_crlb[:,[3,4]] *= n_frames # scale photon count and bg with number of frames

        _, crlb = self.psf.crlb(params_crlb.to(self.device))
        crlb = crlb.cpu()

        expval = self.psf.forward(params_.to(self.device))
        expval = expval.cpu()
        images = expval[:,None].repeat(1, n_frames, 1, 1)

        bs, h, w = expval.shape
        lookahead_frames = 0# self.model.lookahead_frames
        # generate empty lookahead frames. The NN is trained to output the targets 'lookahead' frames later
        images = torch.cat((images, torch.ones(bs, lookahead_frames, h, w) * params[:,4,None,None,None]), 1)
        images = torch.poisson(images)

        # add readnoise
        readnoise = torch.zeros((1,*images.shape[-2:]), dtype=torch.float32, device=self.device)

        # batch eval:
        with torch.no_grad():
            outputs = []
            for b in torch.split(images, batchsize):
                out_ = self.model(b.to(self.device), readnoise.expand(len(b),-1,-1))[0]
                outputs.append(out_.cpu())
        output = torch.cat(outputs, 0)

        outputs = eval_single_emitters(output)[:, -1-lookahead_frames]
        n_params = outputs.shape[-1]//2
        predicted = outputs[:, self.param_ix]
        predicted_error = outputs[:, n_params + self.param_ix]

        # map params in param_ix to psf_param_mapping
        psf_params = ['x', 'y', 'z', 'N', 'bg']
        ix = [psf_params.index(p) for p in self.psf_param_mapping]
        params = params_[:, ix]
        crlb = crlb[:, ix]

        error = predicted - params
        return predicted, crlb, error, predicted_error
    
    def plot_photon_range(self, photons, background = 1, n_frames=1,
            log_writer = None, log_step = 0, show_plot=True, log_label='CRLB-', plot_title=None):
        # plot the CRLB for a range of photon counts
        predicted = []
        crlb = []
        errors = []
        rmsd = []
        pred_error = []

        self.model.eval()

        # N, x, y, z, bg
        xy_spread = 8
        param_scale = [0, xy_spread, xy_spread,0,0]

        for i in pb_range(len(photons)):
            params = torch.tensor(param_scale)[None] * (torch.rand(self.numsamples, 5, dtype=torch.float32)-0.5)
            params[:, 1:3] += self.psf.shape[0] // 2
            #params[:, 2] += torch.linspace(-3, 3, self.numsamples)
            params[:, 0] = photons[i]
            params[:, 4] = background
            predicted, crlb_, errors_, pred_error_ = self.estimate_precision(params, n_frames)
            crlb.append( crlb_.mean(0) )
            errors.append( errors_.std(0) )
            rmsd.append( (errors_**2).mean(0).sqrt() )
            pred_error.append( pred_error_.mean(0))

        crlb = torch.stack(crlb).cpu().numpy()
        errors = torch.stack(errors).cpu().numpy()
        rmsd = torch.stack(rmsd).cpu().numpy()
        pred_error = torch.stack(pred_error).cpu().numpy()

        def add_plot(ax, crlb, errors, rmsd, pred_error, label):
            if crlb is not None:
                ax.plot(photons, crlb, 'k:', label='CRLB')
            ax.plot(photons, errors, 'kx-', label='Std.Dev.')
            ax.plot(photons, rmsd, 'bo-', label='RMSD')
            ax.plot(photons, pred_error,'k--', label='Predicted error')
            ax.loglog()
            ax.set_xlabel('Photons')
            ax.set_ylabel(label)
            ax.legend() 

        if show_plot:
            fig, ax = plt.subplots(1,errors.shape[-1], figsize=(3*errors.shape[-1],3))
            for i in range(len(ax)):
                add_plot(ax[i], crlb[:,i], errors[:,i], rmsd[:,i], pred_error[:,i], self.param_list[i])
            plt.tight_layout()
            if plot_title is not None: plt.suptitle(plot_title)
            plt.show()

        if log_writer is not None:
            for i in range(errors.shape[-1]):
                fig,ax = plt.subplots(1,1)
                add_plot(ax, crlb[:,i], errors[:,i], rmsd[:,i], pred_error[:,i], self.param_list[i])
                if plot_title is not None: plt.title(plot_title)
                log_writer.add_figure(f'{log_label}{self.param_list[i]}', fig, global_step=log_step, close=True)

            #log_writer.add_scalar(f"Intensity estimation RMSD") rmsd.mean()

        return fig

    def plot_photon_range2(self, num_samples, sim_config, background = 2,
            num_bins = 10, eval_frame=-1, batch_size=8,
            log_writer = None, log_step = 0, show_plot=True):

        # plot the CRLB for a range of photon counts
        self.model.eval()

        sim_config = config_dict(sim_config)        

        spots_ds = SingleSpotDataset(xy_spread=8, z_spread=0.1,
                photon_min=sim_config.intensity_mean_min, photon_max=sim_config.intensity_mean_max, 
                num_bins=num_bins, 
                num_samples=num_samples, num_frames=sim_config.num_frames,
                track_intensities=sim_config.track_intensities,
                track_intensities_offset=sim_config.track_intensities_offset,
                intensity_fluctuation_std=sim_config.intensity_fluctuation_std,
                background=background, shape = self.psf.shape)

        simulated = SimulatedSMLMDataset(self.psf)
        simulated.render(spots_ds, compute_crlb=True, use_tqdm=False)

        spots = spots_ds.spots
        summed = spots[:,:1].clone()
        # sum the intensities and backgrounds over all frames 
        summed[:,0,0, 1] = spots[:,:,0,1].sum(-1)
        summed[:,0,0, -1] = spots[:,:,0,-1].sum(-1)

        sum_ds = SimulatedSMLMDataset(self.psf)
        sum_crlb = sum_ds.compute_crlb(labels = summed)

        dl = DataLoader(simulated, batch_size=batch_size, shuffle=False, num_workers=0)

        outputs = []
        labels = []
        with torch.no_grad():
            for data, camera_calib, labels_ in progbar(dl):
                data, camera_calib = data.to(self.device), camera_calib.to(self.device)
                #labels_ = move_to_device(labels, self.device)
                output = self.model(data, camera_calib)[0]
                outputs.append(output.cpu())
                labels.append(labels_)

        labels = batch_cat(labels)
        crlb = simulated.crlb[labels['index']]
        crlb_names = ['N', 'x', 'y', 'z', 'bg']
        labels.pop('index')
        output = batch_cat(outputs)

        locs = eval_single_emitters(output, frame_ix=-1)[:,:,None]
        #locs_frame = locs[:,[-1]]
        #labels = batch_dict_index(labels, (slice(None),[-1],0))
        predicted, predicted_error, true_vals, names = self.model.compare_pairs(locs, labels, -1)
        errors = predicted - true_vals

        crlb_ix = [crlb_names.index(name) if name in crlb_names else None for name in names]
        crlb = crlb[:,-1,0]

        if num_bins is None:
            ...
        else:
            # average over bins
            std_errors = errors.reshape(num_bins, num_samples, -1).std(1)
            rmsd = (errors.reshape(num_bins, num_samples, -1)**2).mean(1).sqrt()

            crlb = crlb.reshape(num_bins, num_samples, -1).mean(1)
            predicted_err = predicted_error.reshape(num_bins, num_samples, -1).mean(1)
            #photons = true_vals[:,names.index('N')].reshape(num_bins, num_samples).mean(1)
            photons = spots_ds.photon_bins

            n_ax = len(names)

            def add_plot(ax, crlb, errors, rmsd, sigma, label):
                if crlb is not None:
                    ax.plot(photons, crlb, 'k:', label='CRLB')
                ax.plot(photons, errors, 'kx-', label='Error')
                ax.plot(photons, rmsd, 'bo-', label='RMSD')
                ax.plot(photons, sigma,'k--', label='Predicted error')
                ax.loglog()
                ax.set_xlabel('Photons')
                ax.set_ylabel(label)
                ax.legend() 

            if show_plot:
                if n_ax > 4:
                    fig, ax = plt.subplots(2, n_ax//2, figsize=(3*n_ax,6))
                    ax = ax.flatten()
                else:
                    fig, ax = plt.subplots(1, n_ax, figsize=(3*n_ax,3))
                for i in range(len(ax)):
                    add_plot(ax[i], crlb[:,crlb_ix[i]] if crlb_ix[i] is not None else None, 
                             std_errors[:,i], rmsd[:,i], predicted_err[:,i], names[i])
                plt.tight_layout()
                plt.show()

            if log_writer is not None:
                for i in range(n_ax):
                    fig,ax = plt.subplots(1,1)
                    add_plot(ax, crlb[:,crlb_ix[i]] if crlb_ix[i] is not None else None, 
                             std_errors[:,i], rmsd[:,i], predicted_err[:,i], names[i])
                    log_writer.add_figure(f'crlb_{names[i]}', fig, global_step=log_step, close=True)



def eval_single_emitters(output, frame_ix=-1):
    batch_size, n_frames, n_features, height, width = output.shape
    
    #output = output[:,frame_ix]
    edge = 5
    output = output[:, :, :, edge:-edge, edge:-edge]

    n_gauss = (n_features - 1) // 2
    prob = output[:,:, 0]

    #prob = prob * (prob > 0.2)
    max_ix = prob[:, frame_ix].reshape(batch_size,-1).argmax(-1)
    return output[:,:,1:].reshape(batch_size,n_frames,n_gauss*2,-1)[range(batch_size),:,:,max_ix]

class SingleSpotDataset(SpotDataset):
    def __init__(self, xy_spread, z_spread, photon_min, photon_max, 
                num_bins, # if None, spread around, otherwise log-spaced bins of photon counts
                 num_samples, num_frames, intensity_fluctuation_std, 
                 background, shape, 
                 track_intensities = None,
                 track_intensities_offset = 0,
                 render_args=None):
        self.xy_spread = xy_spread
        self.z_spread = z_spread
        self.photon_min = photon_min
        self.photon_max = photon_max
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.intensity_fluctuation_std = intensity_fluctuation_std
        self.background = background
        self.height, self.width = shape
        self.render_args = render_args
        self.intensities = None

        # N, x, y, z, bg
        param_scale = [0, xy_spread, xy_spread,0,0]

        # generate points evenly distributed over a log range
        if num_bins is None:
            log_min, log_max = np.log(photon_min), np.log(photon_max)
            photon_means = torch.exp(torch.rand(num_samples) * (log_max - log_min) + log_min)
        else:
            photon_means = torch.logspace(np.log(photon_min), np.log(photon_max), num_bins, base=np.e)
            # each photon count should have num_samples in the bin
            photon_means = photon_means.repeat_interleave(num_samples)

        # apply intensity fluctuations, in the frame dimension
        photons = photon_means[:,None].repeat(1, num_frames)
        photons = photons + (1 + torch.randn_like(photons) * intensity_fluctuation_std)
        self.photon_bins = photons.reshape(num_bins, -1).mean(-1)

        # generate labels for the SimulatedSMLM dataset
        # Output format:
        # [frames, max_spots, 8]
        # Last col: [active, intensity, x, y, z, t_start, t_end, bg]
        spots = torch.zeros(len(photons), num_frames, 8)
        spots[:,:,0] = 1
        spots[:,:,1] = photons
        spots[:,:,2] = self.width/2
        spots[:,:,3] = self.height/2
        spots[:,:,2:4] += (torch.rand(len(photons), 2)[:,None]-0.5) * xy_spread
        spots[:,:,4] = (torch.rand(len(photons))[:,None]-0.5) * z_spread
        spots[:,:,5] = 0
        spots[:,:,6] = num_frames-1
        spots[:,:,7] = background

        # make frame numbers relative
        frame_ix = torch.arange(num_frames)[None,:,None]
        spots[...,5:7] -= frame_ix

        if track_intensities is not None:
            frame_ix = (torch.arange(num_frames)[:,None] + track_intensities_offset + torch.arange(track_intensities) - track_intensities + 1) % num_frames
            self.intensities = spots[:,frame_ix,1][:,:,None]

        self.spots = spots[:,:,None] # 1 spot per frame

    @property
    def max_spots(self):
        return 1

    def __len__(self):
        return len(self.spots)

    def __getitem__(self, ix):
        if self.intensities is not None:
            return dict(spots=self.spots[ix], index=ix, intensities=self.intensities[ix])
        else:
            return dict(spots=self.spots[ix], index=ix)


