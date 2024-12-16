"""
Thoughts on multi-frame model:

use labels from one frame earlier, using slightly more context, boosting jaccard
the more context -> better jacard. How much context is enough? Limited by RNN memory, so lets limit to 6 frames

Current state (track_intensities = 6, track_intensities_offset = 5, target_frame_offset = 0)
* * * * * *
0 1 2 3 4 5
          L
Here, jaccard is limited because frames after 5 are unknown

Optimal would be intensity estimates close to the label (so the spot is likely to be active)
track_intensities = 6, track_intensities_offset = 2, target_frame_offset = 4

    * * * * * *
0 1 2 3 4 5 6 7 8 9
          L 
"""
import torch
import torch.nn as nn
import numpy as np

from smlmtorch.nn.unet import UNet, Decoder, Conv
from smlmtorch.nn.gmm_loss import GMMLoss
from smlmtorch.nn.localization_detector import LocalizationDetector

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if weights is None:
            weights = [1] * len(losses)
        self.weights = weights

    def forward(self, outputs, targets):
        return torch.sum(torch.stack([self.weights[i] * self.losses[i](outputs[i], targets) for i in range(len(self.losses))]))

def repeat_grid_1D(stride, length, device):
    x = torch.arange(stride, device=device)
    return x[None].expand(length//stride, -1).flatten()

def repeat_grid(grid_stride, height, width, device):
    y = repeat_grid_1D(grid_stride, height, device)
    x = repeat_grid_1D(grid_stride, width, device)
    x = x[None,:].expand(height, -1)
    y = y[:,None].expand(-1, width)
    return torch.stack([x,y])

class LocalizationModel(nn.Module):
    def __init__(self, enable3D, param_scale,
                enable_readnoise=False,
                 output_scale=None, input_scale=1, input_offset=0, param_names=None,
                 eps_features=None, sigmoid_features=None, tanh_features=None, 
                 input_subpixel_index=True,
                 input_channels=1, # just image
                 input_upscale_features=None, output_grid_scale=None,
                 xy_param_ix=[1,2]):
        
        """
        Base class for localization models
        """
        super().__init__()

        self.output_grid_scale = output_grid_scale
        self.input_upscale_features = input_upscale_features
        self.input_upscale_layer = None
        self.input_subpixel_index = input_subpixel_index
        self.xy_param_ix = xy_param_ix

        self.enable_readnoise = enable_readnoise
        self.input_channels = input_channels+1 if enable_readnoise else input_channels
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.param_names = param_names
        self.feature_names = ['p', *param_names, *[f'{p}_sig' for p in param_names]]
        self.gmm_params = len(param_names)
        self.enable3D = enable3D

        assert len(param_scale) == self.gmm_params
        
        self.output_features = 1 + 2*self.gmm_params
        output_scale = torch.tensor([1, *param_scale, *param_scale], dtype=torch.float32)

        self.register_buffer('output_scale', output_scale, persistent=False)

        if eps_features is not None:
            self.eps_features = eps_features
        else:
            self.eps_features = 1 + self.gmm_params + np.arange(self.gmm_params)

        if sigmoid_features is not None:
            self.sigmoid_features = sigmoid_features
        else:
            self.sigmoid_features = [0, *(1 + self.gmm_params + np.arange(self.gmm_params))]

        if tanh_features is not None:
            self.tanh_features = tanh_features
        else:
            self.tanh_features = np.arange(1,self.gmm_params+1)

        # A resizing layer to process at a higher resolution than the input image
        # We'll add two extra channels that encode the subpixel index within the orginal pixels
        if self.input_upscale_features is not None:
            input_chan = self.input_channels
            if input_subpixel_index:
                input_chan += 2

            self.input_upscale_layer = nn.Sequential(*[
                Decoder(input_chan, self.input_upscale_features[i], activation=nn.ELU(), 
                    batch_norm=False, skip_channel=False) for i in range(len(self.input_upscale_features))
            ])
            self.model_input_features = self.input_upscale_features[-1]
            self.output_grid_scale = 2**-len(self.input_upscale_features)
        else:
            self.output_grid_scale = 1
            self.model_input_features = self.input_channels

    @property
    def min_frames(self):
        return 1

    def to_global_coords(self, output, revert=False):
        batchsize, frames, features, height, width = output.shape
        output = output.clone()
        f = -1 if revert else 1
        output[:,:,1+self.xy_param_ix[1]] += f * self.output_grid_scale * torch.arange(height, dtype=torch.float32, device=output.device)[None, None, :, None]
        output[:,:,1+self.xy_param_ix[0]] += f * self.output_grid_scale * torch.arange(width, dtype=torch.float32, device=output.device)[None, None, None, :]
        return output

    @staticmethod
    # apply kaiming normal initialization recursively
    def weight_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def prepare_input(self, images, camera_calib=None):
        batchsize, frames, height, width = images.shape
        images.unsqueeze_(2) # add channel dimension
        images = (images.float() - self.input_offset) * self.input_scale
        if self.enable_readnoise:
            readnoise = camera_calib[:,None,None].expand(-1, frames, 1, height, width)
            return torch.cat([images, readnoise], dim=2)
        return images

    def apply_output_scaling(self, x):
        batch_size, frames, features, height, width = x.shape

        x[:, :, self.tanh_features] = x[:, :, self.tanh_features].tanh()
        x[:, :, self.sigmoid_features] = x[:, :, self.sigmoid_features].sigmoid() 

        # trick from decode to avoid near-zero sigmas
        x[:, :, self.eps_features] = torch.clip(x[:, :, self.eps_features], min=0) + 1e-3

        if self.output_scale is not None:
            x *= self.output_scale[None, None, :, None, None]  # scale up the parameters so the loss can be calculated in the real scale
        return x

    def forward(self, images, camera_calib=None, model_state=None):
        x = self.prepare_input(images, camera_calib)
        batchsize, frames, features, height, width = x.shape

        x = x.view(-1, features, height, width)  
        if self.input_upscale_layer is not None:
            if self.input_subpixel_index:
                # add subpixel index to the input, first create a repeated grid
                grid = repeat_grid(2**len(self.input_upscale_features), height, width, x.device)
                x = torch.cat([x, grid[None].expand(len(x),-1,-1,-1)], dim=1)

            x = self.input_upscale_layer(x)

        _, features, height, width = x.shape
        x = x.view(batchsize, frames, features, height, width)  

        r = self.model_forward(x, model_state)
        x = r[0]
        hidden_state_output = r[1]
        batchsize, frames, features, height, width = x.shape

        x = x.view(batchsize * frames, -1, height, width)

        _, features, height, width = x.shape
        x = x.view(batchsize, frames, -1, height, width)
        x = self.apply_output_scaling(x)
        x = self.to_global_coords(x)

        if len(r) > 2:
            return (x, hidden_state_output, *r[2:])
        return x, hidden_state_output

    def create_output_upscale(self, input_features):
        if self.output_upscale_features is not None:
            ch = [input_features, *self.output_upscale_features]
            self.output_upscale_layer = nn.Sequential(*[
                Decoder(ch[i], ch[i+1], activation=nn.ELU(),
                    batch_norm=False, skip_channel=False) for i in range(len(self.output_upscale_features))
            ])
            input_features = self.output_upscale_features[-1]
            self.output_grid_scale *= 2**-len(self.output_upscale_features)
        return input_features



class SingleFrameGMMLoss(torch.nn.Module):
    def __init__(self, param_names, **kwargs):
        super().__init__()
        self.gmm_loss = GMMLoss(len(param_names), **kwargs)
        self.param_names = param_names

    @property
    def param_idx_map(self):
        return {name: i for i, name in enumerate(self.param_names)}

    def forward(self, model_output, targets, return_loss_dict=False):
        batchsize, numframes, features, height, width = model_output.shape
        
        loss = self.gmm_loss(model_output, targets['spots']).mean()

        if return_loss_dict:
            return loss, dict(targets = targets['spots'], outputs = model_output)
        return loss



class SingleFrameUNetModel(LocalizationModel):
    def __init__(self, unet_features, unet_act = nn.ELU(), 
                        unet_batch_norm = False, upscale1=None,  **kwargs):
        
        super().__init__(**kwargs)

        self.unet = UNet(self.model_input_features, unet_features, unet_features[0], unet_act, batch_norm = unet_batch_norm)
        self.create_output_layers(unet_features[0])
        self.apply(self.weight_init)

    def model_forward(self, images, state):
        # x shape: [batchsize, frames, height, width]
        batch_size, frames, channels, height, width = images.size()

        # Reshape the input tensor and combine the batchsize and frames dimensions
        x = images.view(batch_size * frames, channels, height, width)

        x = self.unet(x)
        # shape is now [batchsize * frames, unet_features, height, width]

        x = x.view(batch_size, frames, -1, height, width)
        return x, None

    @property
    def lookahead_frames(self):
        return 0

    def create_loss(self, **kwargs):
        return SingleFrameGMMLoss( self.param_names, **kwargs)

    def compare_pairs(self, locs, labels, frame_ix):
        """ Called by performance loggers to parse the model outputs into plottable items """
        locs = locs[:, frame_ix,0]
        nparams = locs.shape[-1] // 2
        assert (nparams == 4 and not self.enable3D) or (nparams == 5 and self.enable3D)
        predicted = locs[:, :nparams]
        predicted_error = locs[:, nparams:]
        true_vals = labels['spots'][:, frame_ix,0][:, 1:nparams+1]
        return predicted, predicted_error, true_vals, self.param_names



if __name__ == '__main__':
    # Generate some random input data
    batchsize = 2
    frames = 5
    height = 64
    width = 64
    x = torch.randn(batchsize, frames, height, width)

    # x,y,z,I,start,end,bg
    phot_max = 30000
    bg_max = 30
    param_scale=[phot_max,1,1,50,50, bg_max]
    
    input_scaling = 1/100 # max photon count

    # Create an instance of the LocalizationModel class
    model = BidirectionalRNNUNetModel(
        enable3D = False,
        enable_readnoise = False,
        unet1_features=[32, 64], 
        unet2_features=[32, 64, 128],
        rnn_features = 32,
        unet_batch_norm = False,
        output_head_features = 32,
        param_scale = param_scale,
        input_scale = 1/100,
        input_offset = 4,
        param_names = ['phot', 'x', 'y', 's', 'e', 'bg']
    )

    # Pass the input through the model
    output, h = model(x, None)
    print(output.shape) 


