
"""
GPT4 prompt:

create a PerformanceLogger class that writes the different features to tensorboard as images. 
One visualization should use LocalizationDetector() to render the x,y coordinates into one of 
the movie frames. The PerformanceLogger should receive a batch of [batchsize, frames, height, width] 
and also the output of the model for that batch [batchsize, frames, height, width, features]. The X,Y coordinates are features index 1 and 2. 

"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from smlmtorch.nn.localization_detector import LocalizationDetector
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from smlmtorch.nn.localization_detector import LocalizationDetector
from smlmtorch.nn.benchmark.matcher import GreedyHungarianMatcher
from smlmtorch import config_dict

class PerformanceLogger:
    """
    """
    def __init__(self, writer, 
                param_idx_map, 
                intensity_plot_index = 0,
                render_frame_offset = 0, # the frame index that is displayed combined with targets from frame 0.
                xy_hist_range = 0.5,
                kdeplot_params = ['N'],
                histplot_params = ['x', 'y'],
                log_dir='runs', prob_threshold=0.5, match_distance_px=2,):
        self.writer = writer
        self.plot_generators = []
        self.prob_threshold = prob_threshold
        self.match_distance_px = match_distance_px
        self.param_idx_map = config_dict(param_idx_map)
        self.xy_hist_range = xy_hist_range
        self.render_frame_offset = render_frame_offset
        self.kdeplot_params = kdeplot_params
        self.histplot_params = histplot_params

    def add_plot_generator(self, plot_generator):
        self.plot_generators.append(plot_generator)

    @staticmethod
    def array_to_figure(array, feature_name=None, feature_idx=None):
        fig, ax = plt.subplots()
        m=ax.imshow(array, cmap='gray')
        # set title
        if feature_name is not None:
            ax.set_title(f'Feature {feature_name} - [{feature_idx}]')
        fig.colorbar(m,ax=ax)
        return fig
    
    def visualize_localizations(self, frame, localizations, targets):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='gray')

        # Draw true localizations using broadcasting:
        active = targets[targets[:, 0] == 1]
        ax.scatter(active[:, 1+self.param_idx_map.x], active[:, 1+self.param_idx_map.y], c='y', marker='o',s=35)

        # Draw found localizations with extra large marker
        ax.scatter(localizations[:, 1+self.param_idx_map.x], 
                    localizations[:, 1+self.param_idx_map.y], c='b', marker='x', s=20)

        ax.legend(['True', 'Found'])

        # Set limits to image size
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)

        # remove axis ticks``
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def log(self, step, data, model_output, model, targets, close_figs=True):
        # Get batch size, number of frames, height, and width
        batch_size, _, height, width = data.shape

        model_output = model_output.detach().cpu()
        targets = targets.detach().cpu()

        # Pick a random batch and frame
        b = np.random.randint(batch_size)
        f = 0# model_output.shape[1]-1#np.random.randint(num_frames)

        # Log the movie frame
        #fig = self.array_to_figure(frame)
        #self.writer.add_figure(f'Frame', fig, step, close=close_figs)

        # Run LocalizationDetector
        with torch.no_grad():
            detector = LocalizationDetector(prob_threshold=self.prob_threshold)
            idx, localizations = detector.detect(model_output)
            loc_batch_idx = idx['batch']
            loc_frame_idx = idx['frame']

        # Visualize localizations
        frame = data[b, self.render_frame_offset+f].cpu().numpy()
        active = targets[b, 0]
        targets_f = active[active[:, 0] == 1]
        loc_f = localizations[(loc_batch_idx==b) & (loc_frame_idx==0)]
        fig_localizations = self.visualize_localizations(frame, loc_f, targets_f)
        self.writer.add_figure(f'Localizations', fig_localizations, step, close=close_figs)

        self.plot_histograms(localizations, step, close=close_figs)

        # Compare GT
        self.compare_gt(localizations, loc_batch_idx, loc_frame_idx, targets, step, 
                threshold_dist_px=self.match_distance_px, close=close_figs)

        for gen in self.plot_generators:
            gen.process(self.writer, step, data, model_output, targets)

        # Log other features as separate figures
        model_output_local = model.to_global_coords(model_output, revert=True)

        n_params = len(self.param_idx_map)
        idx_map = {'p':0, 
            **{k:v+1 for k,v in self.param_idx_map.items()},
            **{k+'_sig':v+1+n_params for k,v in self.param_idx_map.items()}
        }
        for name, idx in idx_map.items():
            feature_map = model_output_local[b, f, idx, :, :].cpu().numpy()
            fig_feature = self.array_to_figure(feature_map, name, idx)
            self.writer.add_figure(f'{name}', fig_feature, step, close=True)

    def compare_gt(self, loc_data, loc_batch_idx, loc_frame_idx, targets, 
                            step, threshold_dist_px, close=True):
        pairs = []
        for b in range(targets.shape[0]):
            for f in range(targets.shape[1]):
                active = targets[b, f]
                targets_f = active[active[:, 0] == 1]
                loc_f = loc_data[(loc_batch_idx==b) & (loc_frame_idx==f)]
                xy_idx = [1+self.param_idx_map.x,1+self.param_idx_map.y]
                loc_xy = loc_f[:, xy_idx]
                target_xy = targets_f[:, xy_idx]
                frame_pairs = GreedyHungarianMatcher(loc_xy, target_xy).match(threshold_dist_px)
                pairs.append([loc_f[frame_pairs[:,0]], targets_f[frame_pairs[:,1]]])

        # mean distance
        n_gauss = (loc_data.shape[1]-1)//2
        predicted = torch.cat([loc[:,1:n_gauss+1] for loc,target in pairs])
        target = torch.cat([target[:,1:n_gauss+1] for loc,target in pairs])
        predicted_errors = torch.cat([loc[:,1+n_gauss:] for loc,target in pairs])
        errors = predicted - target
        n_matches = np.sum([len(loc) for loc,target in pairs])
        n_locs = loc_data.shape[0]
        n_true = targets[:,:,:,0].sum()
        if n_true>0:
            jaccard = n_matches / (n_locs + n_true - n_matches)
            self.writer.add_scalar('Jaccard', jaccard, step)

        if len(errors)>0:
            for axis_name in self.histplot_params:
                ix = self.param_idx_map[axis_name]
                self.writer.add_scalar(f'Error/{axis_name}', (errors[:,ix]**2).mean().sqrt(), step)
                fig,ax = plt.subplots()
                ax.hist(errors[:,ix], bins=100, range=[-self.xy_hist_range,self.xy_hist_range])
                ax.set_xlabel(f'{axis_name} position [px]')
                self.writer.add_figure(f'{axis_name} Error Histogram', fig, step, close=close)
                #self.writer.add_histogram(f'Error/{axis_name}-pred', predicted_errors[:,axis], step)

            # Use seaborn to plot a density grid of true vs predicted intensity
            for axis_name in self.kdeplot_params:
                ix = self.param_idx_map[axis_name]
                if ix is None:
                    print(f'kde plot: missing axis {axis_name}')
                    continue

                try:
                    fig, ax = plt.subplots()
                    sns.kdeplot(x=target[:,ix], y=predicted[:,ix], ax=ax,fill=True, alpha=0.3, bw_adjust=0.5)
                    # add scatter plot with fraction of the points
                    N = 200
                    sns.scatterplot(x=target[:N, ix], y=predicted[:N, ix], ax=ax, alpha=0.5)
                    
                    #sns.kdeplot(target[:,N_index], predicted[:,N_index], ax=ax)
                    ax.set_xlabel(f'True {axis_name}')
                    ax.set_ylabel(f'Predicted {axis_name}')
                    self.writer.add_figure(f'{axis_name} prediction', fig, step, close=close)
                except Exception as e:
                    print(f'kde plot exception: {e}')

    def plot_histograms(self, localizations, step, close=True):
        fig, ax = plt.subplots()
        ax.hist(localizations[:, 1+self.param_idx_map.x] % 1, bins=50)
        ax.set_xlabel('X Position [px]')
        self.writer.add_figure('X Outputs Histogram', fig, step, close=close)

        fig, ax = plt.subplots()
        ax.hist(localizations[:, 1+self.param_idx_map.y] % 1, bins=50)
        ax.set_xlabel('Y Position [px]')
        self.writer.add_figure('Y Outputs Histogram', fig, step, close=close)
