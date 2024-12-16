"""
GPT4 prompt:
write a class that can detect the localizations from the processed data [batch, frames, pixels, features]. 
Feature 0 contains the probability. Reshape pixels to height,width again, and use non maximum suppression 
with a maxpool of 3x3 pixels to select the best features above a certain threshold (prob > 0.7). 
Output will be a list of detected localizations per frame, per movie. 

"""

import torch
from torch.nn.functional import conv2d

class LocalizationDetector:
    def __init__(self, prob_threshold=0.7, pool_size=3, border=1, use_prob_weights=True):
        self.prob_threshold = prob_threshold
        self.pool_size = pool_size
        self.border = border
        self.use_prob_weights = use_prob_weights
        self.conv_filter = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]).float()

    def detect(self, data, num_loc_frames = 0):
        batch, frames, features, height, width = data.shape

        # Select probability feature (feature 0) and apply threshold
        if num_loc_frames == 0:
            prob_map = data[:,:,0]
        else:
            prob_map = data[:,num_loc_frames-1:,0]

        conv_filter = self.conv_filter.to(data.device)

        #conv2d(prob_map[None,None], cross_filter, padding=1)
        prob_map = conv2d(prob_map.reshape(-1, 1, height, width), conv_filter, padding=1)
        prob_map = prob_map.reshape(batch, frames, height, width)

        #data_conv = conv2d(data.reshape(-1, features, height, width),
        #                   self.conv_filter, padding=1)
        if self.use_prob_weights:
            unfolded = torch.nn.functional.unfold(data.reshape(-1,features,height,width), kernel_size=(3,3),padding=1)
            unfolded = unfolded.reshape(-1, features, 9, height, width)
            unfolded *= conv_filter.flatten()[None, None, :, None, None]
            prob_norm = unfolded[:,[0]] / (unfolded[:, [0]].sum(2, keepdim=True)+1e-6)
            data = (prob_norm * unfolded).sum(2)
            data = data.reshape(batch, frames, features, height, width)

        prob_map = prob_map * (prob_map > self.prob_threshold)

        # Add x and y coordinates to features
        #data[..., 2,:,:] += torch.arange(width, device=data.device)[None,None,None,:]
        #data[..., 3,:,:] += torch.arange(height, device=data.device)[None,None,:,None]

        # Non-maximum suppression
        pooled = torch.nn.functional.max_pool2d(prob_map, kernel_size=self.pool_size, 
                                                stride=1, padding=self.pool_size // 2)
        remaining = (prob_map == pooled) * prob_map

        # shape: [batch, frames, height, width]
        # set pixel borders of remaining to zero
        if self.border>0:
            remaining[..., :self.border, :] = 0
            remaining[..., -self.border:, :] = 0
            remaining[..., :, :self.border] = 0
            remaining[..., :, -self.border:] = 0

        remaining = remaining.cpu()
        data = data.cpu()

        # Get localizations
        frame_localizations = torch.nonzero(remaining, as_tuple=False)
        if len(frame_localizations) == 0:
            return None, torch.zeros((0, features))

        batch_idx, frame_idx, y_idx, x_idx = frame_localizations.t()

        # Repeat over numframes
        if num_loc_frames > 0:
            batch_idx = batch_idx[:,None].repeat(1,num_loc_frames)
            frame_idx = (frame_idx[:, None] - num_loc_frames + 1 + torch.arange(num_loc_frames, device=data.device)[None, :])
            x_idx = x_idx[:,None].repeat(1,num_loc_frames)
            y_idx = y_idx[:,None].repeat(1,num_loc_frames)

        localizations = data[batch_idx, frame_idx, :, y_idx, x_idx]
        return dict(batch = batch_idx, frame= frame_idx, x = x_idx, y = y_idx), localizations
