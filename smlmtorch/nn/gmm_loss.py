#%%
import numpy as np
import math
import torch
import torch.nn as nn
import math
from typing import Optional


def gmm_log_prob(inputs, mu, sigmas, logprob_norm, input_mask):
    # mu shape: (batch, num_gaussians, num_pixels)  
    inputs = inputs[:,:,:,None]  # shape (batch, max_targets, num_gaussians, 1)
    means = mu[:, None]    #  shape (batch, 1, num_gaussians, num_pixels)
    sigmas = sigmas[:, None]
    #assert log_comp_prob.isnan().sum() == 0

    log_normal = -((inputs - means) ** 2) / (2 * sigmas**2) - torch.log(sigmas) #- math.log(math.sqrt(2 * math.pi))
    if input_mask is not None:
        log_normal = log_normal * input_mask[...,None]
    log_gmm_prob = torch.logsumexp(logprob_norm[:,None] + log_normal.sum(2), dim=2)
    return log_gmm_prob



class GMMLoss(nn.Module):
    def __init__(self, num_gaussians, epsilon=1e-8, gmm_components = 0,
                which_feature_chan = None, which_target_chan = None,
                 target_scale=None, count_loss_weight=False):
        super(GMMLoss, self).__init__()

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.gmm_components = gmm_components # if zero, gmm_componets is equal to width*height

        self.which_feature_chan = which_feature_chan
        self.which_target_chan = which_target_chan

        if target_scale is not None:
            self.register_buffer('target_scale', target_scale)
        else:
            self.target_scale = None

        self.count_loss_weight = count_loss_weight

    @property
    def input_features(self):
        return self.num_gaussians * 2 + 1

    def forward(self, features, targets, target_mask : Optional[torch.Tensor] = None):
        # shape of features: [batch_size, frames, features, height, width]
        # shape of targets: [batch_size, frames, max_true,1+ self.num_gaussians]
        # check dimensions:
        if self.which_feature_chan is not None:
            features = features[:,:, self.which_feature_chan, :, :]

        if self.which_target_chan is not None:
            targets = targets[..., self.which_target_chan]

        assert targets.shape[-1] == self.num_gaussians+1
        assert features.shape[1] == targets.shape[1]

        batch_size, frames, nfeatures, h, w = features.shape
        assert nfeatures == self.input_features

        # combine batch and frames dimensions
        features = features.reshape(-1, nfeatures, h, w)
        max_spots = targets.shape[2]
        targets = targets.reshape(-1, max_spots, self.num_gaussians+1)
        if self.target_scale is not None:
            targets[..., 1:] *= self.target_scale
        
        mask = targets[..., 0]
        true_values = targets[..., 1:]

        # Extract the probabilities, means, and standard deviations from the features
        prob = features[:, 0]
        mu = features[:, 1:self.num_gaussians+1]
        sigma = features[:, 1+self.num_gaussians:]

        # clamp sigma to prevent numerical issues
        #sigma = torch.clamp(sigma, min=self.epsilon)

        sigma = torch.clamp(sigma, 1e-10)
        prob = torch.clamp(prob, 1e-20)

        mu = mu.reshape(-1, self.num_gaussians, h*w)
        sigma = sigma.reshape(-1, self.num_gaussians, h*w)
        prob = prob.reshape(-1, h*w)
        true_values = true_values.reshape(-1, max_spots, self.num_gaussians)        
        logprob_norm = torch.log(prob) - (torch.log(prob.sum(-1, keepdim=True)))

        if self.gmm_components != 0:
            # This allows us to train on a much larger number of pixels without the GMM getting super large
            # Note that the normalization is still done over all pixels, so the network will learn to push down the near-zero values
            logprob_exp = logprob_norm[:,None].expand(-1, self.num_gaussians, -1)
            values, indices = torch.topk(logprob_exp, self.gmm_components, dim=-1, largest=True, sorted=False)
            mu = torch.gather(mu, 2, indices)
            sigma = torch.gather(sigma, 2, indices)
            logprob_norm = values[:,0]

        #log_probs = gmm_log_prob(true_values, mu, sigma, prob)
        if target_mask is not None:
            target_mask = target_mask.reshape(*true_values.shape)
        log_probs = gmm_log_prob(true_values, mu, sigma, logprob_norm, target_mask)
        #print(f'gmm log prob: {log_probs.mean()} mu mean: {mu.mean()} sigma mean: {sigma.mean()} prob mean: {prob.mean()} target mean: {true_values.mean()}')

        loss_gmm = torch.sum(log_probs * mask, -1)
        # shape of mask: [batch_size, frames, max_true]
        # shape of prob: [batch_size, frames, height, width]

        # add a loss to encourage the network to push down the values of empty pixels
        #cleanup_loss = (((1-prob)[:,:,None] * means).abs().mean()+ 
        #                ((1-prob)[:,:,None] * sigma).abs().mean())

        if self.count_loss_weight>0:
            p_mean = prob.sum(-1)
            p_var = (prob - prob ** 2).sum(-1)
            # clamp p_var to prevent numerical issues
            p_var = torch.clamp(p_var, min=self.epsilon)

            # compute log prob of N(p_mean, p_var):
            counts = mask.reshape(-1, max_spots).sum(-1)
            count_log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(p_var) - 0.5 * (counts - p_mean) ** 2 / p_var
    
            loss = -loss_gmm - count_log_prob * self.count_loss_weight
        else:
            loss = -loss_gmm

        return loss.reshape(batch_size, frames)




if __name__ == '__main__':
    # Initialize the GMMLoss class with CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gmm_loss = GMMLoss(num_gaussians=3)

    # Create some random tensors with the required dimensions
    movies, frames, h, w, max_true = 2, 3, 4, 5, 6

    features = torch.rand(movies, frames, gmm_loss.input_features, h, w).to(device)
    targets = torch.rand(movies, frames, max_true, gmm_loss.num_gaussians+1).to(device)
    targets[..., 0] = torch.rand(movies, frames, max_true) < 0.5  # Create a mask

    # Ensure the sum of probabilities across the pixels is 1
    features[..., 0] = features[..., 0] / torch.sum(features[..., 0], dim=-1, keepdim=True)

    # Compute the loss
    loss = gmm_loss(features, targets)
    print("Loss:", loss)

