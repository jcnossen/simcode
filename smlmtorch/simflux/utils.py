# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:44:18 2022

@author: jelmer
"""

import torch
import smlmtorch.levmar as lm
from typing import Optional

from smlmtorch.crlb import poisson_crlb


class IntensityAndBgModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, params, const_:Optional[torch.Tensor]=None):
        I = params[:,0]
        bg = params[:,1]
        
        if const_ is None:
            raise ValueError('Expecting psf as constant param.')

        psf = const_
        
        mu = I[:,None,None] * psf + bg[:,None,None]
        deriv = torch.stack((psf, torch.ones_like(psf, device=psf.device)),-1)
        return mu,deriv
    
class IntensityEstimator(torch.nn.Module):
    
    def __init__(self, psf_model, device, ibg_param_ix=[2,3], iterations=50, lambda_=1e2):
        super().__init__()
        
        self.psf_model = psf_model
        self.ibg_param_ix = ibg_param_ix
        self.iterations = iterations
        self.lambda_ = lambda_
        
        param_range = torch.tensor([[1,1e6], [1,1000]], device=device)
        self.ibg_model = IntensityAndBgModel()
        self.ibg_mle = lm.LM_MLE(self.ibg_model, param_range, iterations=iterations, lambda_=lambda_)
        
    def forward(self, params, data):
        nspots = data.shape[0]
        patterns = data.shape[1]
        
        params_ = params*1
        params_[:,self.ibg_param_ix[0]] = 1
        params_[:,self.ibg_param_ix[1]] = 0
        
        mu,jac = self.psf_model.forward(params_)
        
        params_ = params[:,0].repeat_interleave(patterns,0)
        samples = data.reshape((nspots*patterns, data.shape[2], data.shape[3]))

        psf = mu.repeat_interleave(patterns, 0)
        
        initial_ibg = torch.zeros((nspots, patterns,2),device=data.device)
        initial_ibg[:,:,0] = data.sum((2,3)) - data.shape[2] * data.shape[3] * params[:,4,None] / patterns
        initial_ibg[:,:,0] = params[:,4,None] / patterns
        initial_ibg = initial_ibg.reshape((-1,2))
        
        estim = self.ibg_mle(samples, initial_ibg, psf)
        
        ibg_mu, ibg_jac = self.ibg_model(estim, psf)
        ibg_crlb = poisson_crlb(ibg_mu, ibg_jac)
        
        chisq = ( (samples - ibg_mu) ** 2 / (ibg_mu+0.01) ).sum((1,2)).reshape((nspots,patterns))
        
        return estim.reshape((nspots,patterns,2)), ibg_crlb.reshape((nspots,patterns,2)), chisq
    
    
    