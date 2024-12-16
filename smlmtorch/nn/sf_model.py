"""
Two models:

SFIntensityEstimatorModel - Estimates intensities without having the pattern info
SFLocalizerModel - Estimates position using the pattern info
"""
import numpy as np
import torch
import torch.nn as nn
from smlmtorch.nn.gmm_loss import GMMLoss
from smlmtorch.nn.unet import UNet, Conv
from smlmtorch.nn.localization_model import LocalizationModel

def _get_target_tensor(targets, enable3D, use_on_prob):
    batchsize, numframes, maxspots, features = targets['spots'].shape

    # and the targets from 0:-self.frame_output_offset
    # [active, intensity, x, y, z, t_start, t_end, bg]
    spots = targets['spots']
    intensities = targets['intensities']
    on_state = intensities != 0
    # intensities shape: [batchsize, numframes, maxspots, num_intensities]

    active = spots[:,:,:,[0]]
    bg = spots[:,:,:,[7]]
    xyz = spots[:,:,:,2:5] if enable3D else spots[:,:,:,2:4]

    if use_on_prob:
        combined_targets = torch.cat([active, xyz, bg, intensities, on_state], dim=3)
        n_intensities = intensities.shape[-1]
        n_main = 1 + xyz.shape[-1]
        target_mask = torch.ones((batchsize,numframes,maxspots,n_main+n_intensities*2), device=xyz.device, dtype=torch.float32)
        # only estimate intensities for frames in which the spot is active
        target_mask[:,:,:,n_main:n_main+n_intensities] = on_state 
    else:
        combined_targets = torch.cat([active, xyz, bg, intensities], dim=3)
        target_mask = None

    return combined_targets, target_mask


class SFLocalizationModelLoss(torch.nn.Module):
    def __init__(self, num_intensities, param_names, enable3D, use_on_prob,
                track_intensities_offset, use_torch_compile=False,
                **kwargs):
        super().__init__()
        self.gmm_loss = GMMLoss(len(param_names), **kwargs)
        self.enable3D = enable3D
        self.use_on_prob = use_on_prob
        self.num_intensities = num_intensities
        self.param_names = param_names
        self.track_intensities_offset = track_intensities_offset

        if use_torch_compile:
            self.gmm_loss = torch.compile(self.gmm_loss)

    @property
    def param_idx_map(self):
        return {name: i for i, name in enumerate(self.param_names)}

    def forward(self, model_output, targets, return_loss_dict=False):
        batchsize, model_numframes, features, height, width = model_output.shape

        combined_targets, target_mask = _get_target_tensor(targets, enable3D=self.enable3D, use_on_prob=self.use_on_prob)

        # first frame to use as targets for loss
        numframes = combined_targets.shape[1]
        start_frame = self.track_intensities_offset
        end_frame = numframes - self.num_intensities + 1 + self.track_intensities_offset

        combined_targets = combined_targets[:, start_frame:end_frame]
        if target_mask is not None:
            target_mask = target_mask[:, start_frame:end_frame]

        loss = self.gmm_loss(model_output, combined_targets, target_mask)
        loss = loss.mean()

        if return_loss_dict:
            return loss, dict(targets = combined_targets, outputs = model_output)
        return loss


class SFLocalizationModel(LocalizationModel):
    """
    Network structure:

    6 frame input:
    UNet1 per frame
    UNet2 shared (like decode, but with 6 window context)
    Shared conv -> x,y,z,bg
    (Per-frame unet output + shared conv output) -> Nx
    """
    def __init__(self, num_intensities, 
        max_intensity, max_bg, 
        xyz_scale,  # scaling afer tanh activation, so 1,1 allows subpixel positions of -1 to 1
        enable3D,
        unet_shared_features,
        unet_combiner_output_features,
        unet_combiner_features,
        unet_intensity_features,
        output_intensity_features,
        output_head_features, # features that go from combiner to output heads
        ie_input_features=None, # features that go from combiner to intensity estimator inputs. Total combiner outputs = output_head_features + ie_input_features * num_intensities
        unet_batch_norm=True, 
        use_on_prob = False, # output on-probablities for each frame, to reduce biasing the intensities
        modulated_estim = None, # enable modulation-enhanced estimation
        unet_act=nn.ELU(),
        **kwargs):

        if not enable3D:
            xyz_scale = xyz_scale[:2]
            xyz_names = ["x", "y"]
        else:
            xyz_names = ["x", "y", "z"]

        param_scale = [ *xyz_scale, max_bg ]
        param_names = [ *xyz_names, "bg" ]

        param_scale.extend([max_intensity]*num_intensities)
        param_names.extend([f"N{i}" for i in range(num_intensities)])

        if use_on_prob:
            param_scale.extend([1]*num_intensities)
            param_names.extend([f"p{i}" for i in range(num_intensities)])

        xy_param_ix = [0, 1]

        super().__init__(param_scale = param_scale, param_names = param_names,  
            xy_param_ix=xy_param_ix, enable3D=enable3D, **kwargs)

        self.modulated_estim = modulated_estim
        self.use_on_prob = use_on_prob
        self.num_intensities = num_intensities
        self.numdims = 3 if enable3D else 2
        self.output_head_features = output_head_features
        self.unet_intensity_features = unet_intensity_features
        self.ie_input_features = ie_input_features
        self.unet_combiner_output_features = unet_combiner_output_features
        
        self.unet_shared = UNet(self.model_input_features, unet_shared_features, unet_shared_features[0], unet_act, batch_norm = unet_batch_norm)

        #self.main_features = 1+len(xyz_names)+1 # p,x,y[,z],bg
        combiner_output_feat = self.unet_combiner_output_features + ie_input_features * num_intensities 
        self.unet_combiner = UNet(unet_shared_features[0]*num_intensities,
            unet_combiner_features, combiner_output_feat, unet_act, batch_norm = unet_batch_norm)
        
        self.unet_intensity = UNet(unet_shared_features[0] + ie_input_features, 
            unet_intensity_features, unet_intensity_features[0], unet_act, batch_norm = unet_batch_norm)

        main_features = 1+2*(len(xyz_names)+1) # p,x,y[,z],bg
        # p,(x,y,z,bg)
        self.output_heads_layer_main = nn.Sequential(
            Conv(self.unet_combiner_output_features, self.output_head_features * main_features, activation=nn.ELU(), batch_norm=False),
            nn.Conv2d(self.output_head_features * main_features, main_features, kernel_size=1, stride=1, groups=main_features)
        )
        intensity_outputs = 4 if use_on_prob else 2
        self.output_heads_intensities = nn.Sequential(
            Conv(unet_intensity_features[0], output_intensity_features * intensity_outputs, activation=nn.ELU(), batch_norm=False),
            nn.Conv2d(output_intensity_features * intensity_outputs, intensity_outputs, kernel_size=1, stride=1, groups=intensity_outputs)
        )

        if self.modulated_estim is not None:
            unet_sf_features = self.modulated_estim['unet_features']
            sf_inputs = self.unet_combiner_output_features + self.unet_intensity_features[0]*num_intensities
            sf_inputs += num_intensities * 2 # x and y phase
            # add a final UNet that combines all the information, including patterns, into a modulation-enhanced estimate
            self.unet_sf = UNet(sf_inputs, unet_features, 
                unet_sf_features[0], unet_act, batch_norm = unet_batch_norm)
            self.output_heads_layer_sf = nn.Sequential(
                Conv(unet_sf_features[0], self.output_head_features * main_features, activation=nn.ELU(), batch_norm=False),
                nn.Conv2d(self.output_head_features * main_features, main_features, kernel_size=1, stride=1, groups=main_features)
            )

        # output_heads_layer needs to be conv2d compatible,
        # and gets passed [batchsize, frames, features, height, width]

        self.apply(self.weight_init)

    @property
    def min_frames(self):
        return self.num_intensities

    def create_loss(self, **config):
        return SFLocalizationModelLoss(self.num_intensities, self.param_names,  
            self.enable3D, use_on_prob = self.use_on_prob, **config)

        """
        ie_frames = self.num_intensities
        # if nonzero, p[frame] contains the info for images[frame-1]
        # 
        la = self.lookahead_frames 

        src_frame_ix = torch.arange(frames)[:,None] - la - ie_frames + 1 + torch.arange(ie_frames)[None]
        param_frame_ix = torch.arange(frames)[:,None].expand(-1, ie_frames)
        """

    def model_forward(self, images, hidden_state = None):
        # x shape: [batchsize, frames, height, width]
        batch_size, frames, channels, height, width = images.size()
        assert self.model_input_features == channels
        
        # Reshape the input tensor and combine the batchsize and frames dimensions
        x = images.view(batch_size * frames, channels, height, width)
        
        x = self.unet_shared(x)
        x = x.view(batch_size, frames, -1, height, width)
        
        nblocks = frames - self.num_intensities + 1
        frame_ix = torch.arange(nblocks, device=x.device)[:,None] + torch.arange(self.num_intensities, device=x.device)[None]

        x_repeated = x[:,frame_ix]
        # shape is now [batchsize, nblocks, num_intensities, unet_features[0], height, width]
        # merge intensities dimension with features
        x_repeated = x_repeated.view(batch_size * nblocks, -1, height, width)
        x_combiner = self.unet_combiner(x_repeated)

        x_main = self.output_heads_layer_main(x_combiner[:, :self.unet_combiner_output_features])

        x_ie_input = x_combiner[:,self.unet_combiner_output_features:].reshape(-1, self.ie_input_features, height, width)
        # x_ie_input: [batchsize * nblocks * num_intensities, ie_input_features, height, width]
        # combine shared outputs with combined outputs for intensity estimator
        x_ie_input = torch.cat([x_ie_input, x_repeated.view(batch_size * nblocks * self.num_intensities, -1, height, width)], dim=1)
        x_ie_0 = self.unet_intensity(x_ie_input)
        x_ie = self.output_heads_intensities(x_ie_0)
    
        main_features = x_main.shape[1]
        main_params = (main_features-1)//2

        if self.use_on_prob:
            x_ie = x_ie.view(batch_size * nblocks, self.num_intensities * 4, height, width)
            # intensity estimator will have N0 N0_sig p0 p0_sig, ...
            intensities = x_ie[:, ::4]
            intensity_sigma = x_ie[:, 1::4]

            on_probs = x_ie[:, 2::4]
            on_probs_sig = x_ie[:, 3::4]

            # cat the intensity outputs to the main outputs
            # p, mu (N0,..., x,y,z, start_frame, bg), sigma (N0,..., x,y,z, start_frame, bg)
            x = torch.cat([
                x_main[:,:1],  # p
                x_main[:,1:1+main_params], # xyzbg
                intensities, # Intensity outputs
                on_probs,
                x_main[:,1+main_params:],
                intensity_sigma, # pred error Nx
                on_probs_sig
            ], dim=1)
        else:
            x_ie = x_ie.view(batch_size * nblocks, self.num_intensities * 2, height, width)
            # intensity estimator will have N0 N0_sig, ...
            intensities = x_ie[:, ::2]
            intensity_sigma = x_ie[:, 1::2]

            # cat the intensity outputs to the main outputs
            # p, mu (N0,..., x,y,z, start_frame, bg), sigma (N0,..., x,y,z, start_frame, bg)
            x = torch.cat([
                x_main[:,:1],  # p
                x_main[:,1:1+main_params], # xyzbg
                intensities, # Intensity outputs
                x_main[:,1+main_params:],
                intensity_sigma # pred error Nx
            ], dim=1)

        x = x.view(batch_size, nblocks, -1, height, width)

        return x, None

    def compare_pairs(self, locs, targets, frame_ix):
        """ Called by performance loggers to parse the model outputs into plottable items """
        t = _mf_get_target_tensor(targets, self.enable3D, self.use_on_prob)

        locs = locs[:, frame_ix,0]
        t = t[:, frame_ix,0]

        nparams = locs.shape[-1] // 2
        #assert (nparams == 4 and not self.enable3D) or (nparams == 5 and self.enable3D)
        predicted = locs[:, :nparams]
        predicted_error = locs[:, nparams:]
        true_vals =  t[:, 1:nparams+1]

        return predicted, predicted_error, true_vals, self.param_names


if __name__ == '__main__': 
    model = SFLocalizationModel(
        enable_readnoise = False,
        enable3D = False,
        unet_shared_features=[12, 24], 
        unet_combiner_features=[16, 32],
        ie_input_features=8,
        unet_batch_norm = True,
        input_scale = 0.01, # get pixel values into useful range
        input_offset = 3,
        output_head_features = 24,
        num_intensities=6,
        max_bg=100,
        max_intensity=10000,
        xyz_scale=[1,1], 
        unet_intensity_features=[16, 32],
        output_intensity_features=8
    )


