# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:06:28 2022

@author: jelmer
"""
import torch
import time
from torch import Tensor
from typing import List, Tuple, Union, Optional, Final

import matplotlib.pyplot as plt
import numpy as np

import smlmtorch.gaussian_psf as g2d
import smlmtorch.levmar as lm
from smlmtorch.crlb import poisson_crlb_select_axes


from smlmtorch.grad_descent import BatchAdaptiveGradientDescent, PoissonLikelihoodLoss

def sf_modulation(xyz:Tensor, mod:Tensor):
    dims = xyz.shape[1]
    k = mod[:, :, :dims] 
    phase = mod[:, :,-2]
    depth = mod[:,:,-3]
    relint = mod[:,:,-1]

    em_phase = (xyz[:,None] * k).sum(-1) - phase
    
    deriv = depth[:,:,None] * k * torch.cos(em_phase)[:,:,None] * relint[:,:,None] # [spots,patterns,coords]
    intensity = (1+depth * torch.sin(em_phase))*relint

    return intensity,deriv    



class SIMFLUXModel(torch.nn.Module):
    """
    params: [batchsize, xyzIb]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint
    """

    def __init__(self, psf, divide_bg=False, divide_I=False, dims=3):
        super().__init__()
        self.psf = psf
        self.divide_bg = divide_bg
        self.divide_I = divide_I
        self.dims = dims

    def forward(self, params, mod:Optional[Tensor]=None):
        if mod is None:
            raise ValueError('expecting modulation patterns')
            
        # mu = bg + I * P(xyz) * PSF(xyz)
        pos = params[:, :self.dims]
        I = params[:,[-2]]
        bg = params[:,[-1]]
        
        if self.divide_bg:
            bg /= mod.shape[1]
            
        if self.divide_I:
            int_total = mod[:,:,-1].mean(-1) * mod.shape[1]
            I /= int_total 
            
        mod_intensity, mod_deriv = sf_modulation(pos, mod)
        
        normalized_psf_params = torch.cat((pos, 
                                           torch.ones(I.shape,device=params.device),
                                           torch.zeros(I.shape,device=params.device)), -1)
    
        psf_ev, psf_deriv = self.psf(normalized_psf_params)
    
        phot = I[:,None,None]
        mu = bg[:,None,None] + mod_intensity[:,:,None,None] * phot * psf_ev[:, None, :,:]
    
        deriv_pos = phot[...,None] * (mod_deriv[:,:, None,None] * psf_ev[:, None,...,None] + 
                          mod_intensity[:,:,None,None,None] * psf_deriv[:,None,...,:self.dims])
    
        deriv_I = psf_ev[:,None] * mod_intensity[:,:, None,None]
        
        deriv_bg = torch.ones(deriv_I.shape, device=params.device)
    
        return mu, torch.cat((deriv_pos, 
                              deriv_I[...,None], 
                              deriv_bg[...,None]),-1)

@torch.jit.script
class ModulatedIntensitiesModel:
    """
    params: [batchsize, ...]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint
    
    basically simflux_model but with all psf derivatives set to zero, so they don't contribute any FI.
   
    psf_ev is now just 1. 
    This brings up a problem for the background, which is defined per pixel.
    As a hack, bg_factor can be set to the number of pixels that the PSF roughly covers, 
    divided by number of patterns (as in normal simflux)
    """
    def __init__(self, dims:int=3, bg_factor:float=1):
        #super().__init__()
        self.dims = dims
        self.bg_factor = bg_factor
        
    def __call__(self,params, mod:Optional[Tensor]=None):
        return self.forward(params, mod)

    def forward(self, params, mod:Optional[Tensor]=None):
        if mod is None:
            raise ValueError('expecting modulation patterns')
        xyz = params[:, :self.dims]
        I = params[:,[self.dims]]
        bg = params[:,[self.dims+1]] * self.bg_factor
            
        mod_intensity, mod_deriv = sf_modulation(xyz, mod)
        
        mu = bg + mod_intensity * I
        
        deriv_xyz = I[...,None] * mod_deriv
        deriv_I = mod_intensity
        deriv_bg = torch.ones(deriv_I.shape, device=params.device)
    
        return mu, torch.cat((deriv_xyz, 
                              deriv_I[...,None], 
                              deriv_bg[...,None]),-1)
    
        

    
def estimate_precision(x, photoncounts, phot_ix, psf_model, mle, plot_axes=None, 
                       axes_scaling=None, axes_units=None, skip_axes=[],
                       const_=None, jit=True):
    crlb = []
    prec = []
    rmsd = []
    runtime = []
       
    if jit:
        mle = torch.jit.script(mle)
    
    for i, phot in enumerate(photoncounts):
        x_ = x*1
        x_[:,phot_ix] = phot
        mu, deriv = psf_model(x_,const_)
        smp = torch.poisson(mu)
        
        initial = x_*(torch.rand(x_.shape,device=x.device) * 0.2 + 0.9) 
        #initial = x*1
        
        t0 = time.time()
        estim = mle(smp, initial, const_)
                
        errors = x_ - estim

        t1 = time.time()
        runtime.append( len(x) / (t1-t0+1e-10)) # 
        
        prec.append(errors.std(0))
        rmsd.append(torch.sqrt((errors**2).mean(0)))
        
        crlb_i = poisson_crlb_select_axes(mu, deriv, skip_axes=skip_axes)
        crlb.append(crlb_i.mean(0))
        
    print(runtime)

    crlb = torch.stack(crlb).cpu()
    prec = torch.stack(prec).cpu()
    rmsd = torch.stack(rmsd).cpu()
        
    if plot_axes is not None:
        figs = []
        for i, ax_ix in enumerate(plot_axes):
            fig,ax = plt.subplots()
            figs.append(fig)
            ax.loglog(photoncounts, axes_scaling[i] * prec[:, ax_ix], label='Precision')
            ax.loglog(photoncounts, axes_scaling[i] * crlb[:, ax_ix],'--',  label='CRLB')
            ax.loglog(photoncounts, axes_scaling[i] * rmsd[:, ax_ix],':k', label='RMSD')
            ax.legend()
            ax.set_title(f'Estimation precision for axis {ax_ix}')
            ax.set_xlabel('Photon count [photons]')
            ax.set_ylabel(f'Precision [{axes_units[i]}]')
            
        return crlb, prec, rmsd, figs
        
    return crlb, prec, rmsd

    
    
def make_mod3(depth=0.9, kxy=2, kz=0.01):
    K = 6
    
    mod = np.array([
       [0, kxy,kz, depth, 0, 1/K],
       [kxy, 0,kz, depth, 0, 1/K],
       [0, kxy,kz, depth, 2*np.pi/3, 1/K],
       [kxy, 0,kz, depth, 2*np.pi/3, 1/K],
       [0, kxy,kz, depth, 4*np.pi/3, 1/K],
       [kxy, 0,kz, depth, 4*np.pi/3, 1/K],
      ])
    #mod[:,2]=1
    return torch.Tensor(mod)    


def test_gauss_fixed_sigma():
    np.random.seed(0)

    N = 500
    roisize = 9
    thetas = np.zeros((N, 4))
    thetas[:,:2] = roisize/2 + np.random.uniform(-roisize/8,roisize/8, size=(N,2))
    thetas[:,2] = 1000#np.random.uniform(200, 2000, size=N)
    thetas[:,3] = 5# np.random.uniform(1, 10, size=N)

    dev = torch.device('cuda')
    
    param_range = torch.Tensor([
        [0, roisize-1],
        [0, roisize-1],
        [1, 1e9],
        [1.0, 1e6],
    ]).to(dev)

    sigma = 1.5
    psf_model = g2d.Gaussian2DFixedSigmaPSF(roisize, sigma)
    thetas = torch.Tensor(thetas).to(dev)
        
    #sf_model = lambda theta: psf_model(sf_model())
    
    mod = make_mod3().to(dev)

    sf_model = SIMFLUXModel(psf_model,divide_bg=True,dims=2)
            
    photoncounts = np.logspace(2, 4, 10)

    smlm_mle = lm.LM_MLE(psf_model, param_range, iterations=50, lambda_=1e3)
    sf_mle = lm.LM_MLE(sf_model, param_range, iterations=50, lambda_=1e3)

    #    sfi_crlb, sfi_prec, sfi_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3, mle=
    #                       psf_model = sfi_model_3D, const_=mod[None],
    #                       skip_axes=[2])
    
    smlm_crlb, smlm_prec, smlm_rmsd = estimate_precision(thetas, photoncounts, phot_ix=2, mle=smlm_mle,
                       psf_model = psf_model)

    sf_crlb, sf_prec, sf_rmsd = estimate_precision(thetas, photoncounts, psf_model=sf_model, mle=sf_mle, phot_ix=2, 
                       const_=mod[None])


    
    plot_axes=[0,1,2,3]
    axes_scaling=[100,100,1,1]
    axes_units=['nm', 'nm', 'photons', 'ph/bg']

    for i, ax_ix in enumerate(plot_axes):
        fig,ax = plt.subplots()
        #ax.loglog(photoncounts, axes_scaling[i] * astig_prec[:, ax_ix], label='Precision (SF)')
        ax.loglog(photoncounts, axes_scaling[i] * sf_crlb[:, ax_ix], '--b',  label='CRLB (SF)')
        ax.loglog(photoncounts, axes_scaling[i] * sf_rmsd[:, ax_ix], 'ob', ms=4, label='RMSD (SF)')

        ax.loglog(photoncounts, axes_scaling[i] * smlm_crlb[:, ax_ix], '--g',  label='CRLB (SMLM)')
        ax.loglog(photoncounts, axes_scaling[i] * smlm_rmsd[:, ax_ix], 'og', ms=4,label='RMSD (SMLM)')

        ax.legend()
        ax.set_title(f'Estimation precision for axis {ax_ix}')
        ax.set_xlabel('Photon count [photons]')
        ax.set_ylabel(f'Precision [{axes_units[i]}]')

    plt.show()
    

def test_gauss_astig():
    gauss3D_calib = [
        [1.0,-0.12,0.2,0.1],
         [1.05,0.15,0.19,0]]
    
    np.random.seed(0)

    N = 200
    roisize = 9
    thetas = np.zeros((N, 5))
    thetas[:,:2] = roisize/2 + np.random.uniform(-roisize/8,roisize/8, size=(N,2))
    thetas[:,2] = np.random.uniform(-0.3,0.3,size=N)
    thetas[:,3] = 1000#np.random.uniform(200, 2000, size=N)
    thetas[:,4] = 200# np.random.uniform(1, 10, size=N)

    dev = torch.device('cuda')
    
    param_range = torch.Tensor([
        [0, roisize-1],
        [0, roisize-1],
        [-1, 1],
        [1, 1e9],
        [1.0, 1e6],
    ]).to(dev)

    gauss3D_calib = torch.tensor(gauss3D_calib).to(dev)
    thetas = torch.Tensor(thetas).to(dev)
    mu, jac = g2d.gauss_psf_2D_astig(thetas, roisize, gauss3D_calib)

    smp = torch.poisson(mu)
    plt.figure()
    plt.imshow(smp[0].cpu().numpy())
        
    psf_model = g2d.Gaussian2DAstigmaticPSF(roisize, gauss3D_calib)
    #sf_model = lambda theta: psf_model(sf_model())
    
    #mod = make_mod3().to(dev)
    mod = make_mod3(kxy=0.5,kz=4).to(dev)
    
    psf = g2d.Gaussian2DAstigmaticPSF(roisize, gauss3D_calib)
    #lm_estimator = torch.jit.script( LM_MLE(psf, param_range, iterations=100, lambda_=1e3) )

    #sfi_model_2D = ModulatedIntensitiesModel(dims=2,bg_factor=20 / len(mod))
    #sfi_model_3D = ModulatedIntensitiesModel(dims=3,bg_factor=20 / len(mod))
    
    #sfi_mu, sfi_jac = sfi_model_3D(thetas, mod[None])
    #sfi_crlb = lm.compute_crlb(sfi_mu, sfi_jac, skip_axes=[2])

    sf_model = SIMFLUXModel(psf,divide_bg=True)

    sf_mu, sf_jac = sf_model.forward(thetas, mod[None])
    sf_crlb = poisson_crlb_select_axes(sf_mu, sf_jac)
        
    crlb = poisson_crlb_select_axes(mu,jac)
    print(f"SMLM CRLB: {crlb.mean(0)} ")
    print(f"SF CRLB: {sf_crlb.mean(0)} ")
    
    photoncounts = np.logspace(2, 4, 10)

    astig_mle = lm.LM_MLE(psf_model, param_range, iterations=200, lambda_=1e2)
    sf_mle = lm.LM_MLE(sf_model, param_range, iterations=200, lambda_=1e2)
    
    #    sfi_crlb, sfi_prec, sfi_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3, mle=
    #                       psf_model = sfi_model_3D, const_=mod[None],
    #                       skip_axes=[2])
    
    loss = PoissonLikelihoodLoss(sf_model)
    #loss = torch.jit.script(loss)

    gd_estimator = None
   
    if False:
        gd_estimator = BatchAdaptiveGradientDescent(loss, param_range.T,  initial_step=0.1)
    
        def gd_estim(smp, initial, const_):
            return gd_estimator.forward(smp,initial,const_=const_)[0]
    
        
        _, sf_prec_gd, sf_rmsd_gd = estimate_precision(thetas, photoncounts, 
                        psf_model=sf_model, mle=gd_estim, phot_ix=3, 
                        const_=mod[None].repeat((N,1,1)), jit=False)
    
    astig_crlb, astig_prec, astig_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3, mle=astig_mle,
                       psf_model = psf_model)

    sf_crlb, sf_prec, sf_rmsd = estimate_precision(thetas, photoncounts, 
                    psf_model=sf_model, mle=sf_mle, phot_ix=3, const_=mod[None])

    
    plot_axes=[0,1,2,3]
    axes_scaling=[100,100,1000,1]
    axes_units=['nm', 'nm', 'nm', 'photons']

    for i, ax_ix in enumerate(plot_axes):
        fig,ax = plt.subplots()
        #ax.loglog(photoncounts, axes_scaling[i] * astig_prec[:, ax_ix], label='Precision (SF)')
        ax.loglog(photoncounts, axes_scaling[i] * sf_crlb[:, ax_ix], '--b',  label='CRLB (SF)')
        ax.loglog(photoncounts, axes_scaling[i] * sf_rmsd[:, ax_ix], 'ob', ms=4, label='RMSD (SF)')
        if gd_estimator is not None:
            ax.loglog(photoncounts, axes_scaling[i] * sf_rmsd_gd[:, ax_ix], 'ok', ms=4, label='RMSD (SF,GD)')
        if ax_ix != 2:
            ...
            #ax.loglog(photoncounts, axes_scaling[i] * sfi_crlb[:, ax_ix], '--r',  label='CRLB (SFI)')
            #ax.loglog(photoncounts, axes_scaling[i] * sfi_rmsd[:, ax_ix], 'or', ms=4,label='RMSD (SFI)')
        ax.loglog(photoncounts, axes_scaling[i] * astig_crlb[:, ax_ix], '--g',  label='CRLB (Astig.)')
        ax.loglog(photoncounts, axes_scaling[i] * astig_rmsd[:, ax_ix], 'og', ms=4,label='RMSD (Astig.)')
        #ax.loglog(photoncounts, axes_scaling[i] * astig_prec[:, ax_ix], '+g', ms=4,label='Prec.(Astig.)')
        ax.legend()
        ax.set_title(f'Estimation precision for axis {ax_ix}')
        ax.set_xlabel('Photon count [photons]')
        ax.set_ylabel(f'Precision [{axes_units[i]}]')

    plt.show()
    

#%%
if __name__ == '__main__':
    #test_gauss_fixed_sigma()
    test_gauss_astig()

