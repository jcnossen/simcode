#%%
"""
Normally-distributed intensities (NDI) fitter
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Optional, List, Final
import tqdm
import os

from smlmtorch import config_dict
from smlmtorch.smlm.levmar import lm_update

from smlmtorch.smlm.gaussian_psf import gauss_psf_2D_fixed_sigma

def _np(x):
    return x.detach().cpu().numpy()

@torch.jit.script
def ndi_bg_model(params : Tensor, mod : Tensor, ndims:int):
    """
    params: [x,y,z,I0,I1,...,b0,b1...], depending on number of angles
    """
    x = params[:,:ndims]
    k = mod[:,:,:,:ndims]
    depth = mod[:,:,:,3]
    phaseshift = mod[:,:,:,4]

    nax = mod.shape[1]
    phasesteps = mod.shape[2]

    I = params[:,ndims:ndims+nax]
    bg = params[:,ndims+nax:]
    
    phase = (x[:,None]*k[:,:,0]).sum(2,keepdim=True) - phaseshift
    exc = (1+depth*torch.sin(phase)) / phasesteps
    I_modulated = I[:,:,None] * exc + bg[:,:,None]
    
    I_modulated_deriv_xyz = I[:,:,None,None] * k * (
        depth*torch.cos(phase))[:,:,:,None] / phasesteps
    
    # remove z for now
    #I_modulated_deriv_xy = I_modulated_deriv_xyz[...,:2]
    
    I_deriv = torch.eye(nax,device=params.device)[None,:,None] * exc[:,:,:,None]
    bg_deriv = torch.eye(nax,device=params.device)[None,:,None] * (exc[:,:,:,None]*0+1)

    jac = torch.cat([I_modulated_deriv_xyz, I_deriv, bg_deriv], -1)
    
    return I_modulated, jac


@torch.jit.script
def ndi_model(params:Tensor, mod:Tensor, ndims:int):
    """
    params: [x,y,z,I0,I1,...], depending on number of angles
    """
    
    nax = mod.shape[1]

    assert len(mod.shape) == 4
    assert params.shape[1] == ndims+nax
    
    x = params[:,:ndims]
    k = mod[:,:,:,:ndims]
    depth = mod[:,:,:,3]
    phaseshift = mod[:,:,:,4]
    
    phasesteps = mod.shape[2]
    
    phase = (x[:,None]*k[:,:,0]).sum(2,keepdim=True) - phaseshift
    I = params[:,ndims:ndims+nax]
    exc = (1+depth*torch.sin(phase)) / phasesteps
    I_modulated = I[:,:,None] * exc
    
    I_modulated_deriv_xyz = I[:,:,None,None] * k * (
        depth*torch.cos(phase))[:,:,:,None] / phasesteps
    
    # remove z for now
    #I_modulated_deriv_xy = I_modulated_deriv_xyz[...,:2]
    
    #I_deriv = [(np.eye(nax)[i][None,:,None] * exc)[...,None] for i in range(nax)]
    I_deriv = torch.eye(nax,device=params.device)[None,:,None] * exc[:,:,:,None]
    jac = torch.cat([I_modulated_deriv_xyz, I_deriv], -1)
    
    return I_modulated, jac
    
@torch.jit.script
def intensity_bg_crlb(psfs :Tensor, I:Tensor, bg:Tensor) -> Tensor:
    
    jac = torch.stack([psfs, psfs*0+1],-1)
    mu = psfs * I[:,None,None] + bg[:,None,None]
    
    fisher = torch.matmul(jac[...,None], jac[...,None,:])
    fisher = fisher / mu[...,None,None]  
    fisher = fisher.flatten(1, 2).sum(1)

    return torch.sqrt( torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2) )

@torch.jit.script
def _compute_nd_crlb(jac:Tensor, sig:Tensor) -> Tensor:
    """
    Compute CRLB for estimation with normally distributed errors
    """
    fisher = torch.matmul(jac[...,None], jac[...,None,:])
    fisher = fisher / sig[...,None,None]**2
    fisher = fisher.sum((1,2))
    
    return torch.sqrt( torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2) )


def ndi_crlb_5p(params, psfs, mod, ndims=2):
    """
    Computes CRLB assuming isolated PSFs defined by PSF, and background as given in params
    # params:x,y,z,I,bg
    # background is ignored and each modulation axes has an individual spot intensity
    """
    bg = params[:,4]
    
    naxes = mod.shape[1]
        
    # expanded params(ep): x,y,z,I0,I1,I2
    ep = np.zeros((len(params), ndims+naxes))
    ep[:,:ndims] = params[:,:ndims]
    ep[:,ndims:] = params[:,3,None] / naxes# 
    
    I_ev, I_ev_deriv = ndi_model(ep, mod, ndims)
    
    npat = mod.shape[1]*mod.shape[2]
    # compute modulated intensities
    ibg_crlb = intensity_bg_crlb(psfs.repeat(npat,0), 
                                 I_ev.flatten(), bg.repeat(npat)).reshape( 
                                     (len(params), mod.shape[1], mod.shape[2], 2))
                                         
    I_sig = ibg_crlb[:,:,:,0]
    return _compute_nd_crlb(I_ev_deriv, I_sig)

#@torch.jit.script
def ndi_crlb(params, I_sig, mod, ndims:int=2):
    """
    Computes CRLB assuming isolated PSFs defined by PSF, and background as given in params
    # params:x,y,z,I,bg
    # background is ignored and each modulation axes has an individual spot intensity
    """
    nax = mod.shape[1]
    assert params.shape[1] == ndims+nax
    assert len(I_sig.shape) == 3 # [batchsize, nax, nphase]
    
    I_ev, jac = ndi_model(params, mod, ndims)
    return _compute_nd_crlb(jac, I_sig)

@torch.jit.script
def ndi_bg_crlb(params, I_sig, mod, ndims:int=2):
    """
    Computes CRLB assuming isolated PSFs defined by PSF, and background as given in params
    # params:x,y,z,I,bg
    # background is ignored and each modulation axes has an individual spot intensity
    """
    nax = mod.shape[1]
    assert params.shape[1] == ndims+nax*2
    assert len(I_sig.shape) == 3 # [batchsize, nax, nphase]
    
    I_ev, jac = ndi_bg_model(params, mod, ndims)
    return _compute_nd_crlb(jac, I_sig)


#def _clip(a, a_min, a_max):
#    t

def ndi_bg_fit(initial:Tensor, I_smp_mu:Tensor, I_smp_sig:Tensor, mod:Tensor, param_limits :Tensor, 
            ndims:int=2, lambda_:float=10, iterations:int = 100, normalize_scale:bool=True):
    """
    I_mu & I_sig: [batchsize, #axes, #phase steps]
    initial: [batchsize, params: [x,y,z,I0,I1,...,b0,b1...]]
    
    """
    nax = mod.shape[1]
    assert initial.shape[1] == ndims+nax*2

    if not (param_limits.shape[0] == ndims+nax*2 and param_limits.shape[1] == 2):
        raise ValueError("param_limits must be [ndims+nax*2, 2]")

    dev = initial.device
    params = initial*1   
    traces = []
    errtrace = []
    
    #with progbar(total=iterations) as pb:
    for i in range(iterations):
        I_ev, I_ev_deriv = ndi_bg_model(params, mod, ndims)
        jac = I_ev_deriv / I_smp_sig[...,None]
        
        sqerr = (( (I_ev - I_smp_mu) / I_smp_sig )**2).mean((1,2))
        
        # now get into Ax = b form..
        # A = J^T * J + lambda * ident
        # b = J^T * (I_mu - I_ev) / I_sig
        A = torch.matmul(jac[...,None], jac[...,None,:]).sum((1,2))
        b = (jac * ((I_smp_mu - I_ev) / I_smp_sig)[...,None]).sum((1,2))
 
        scale = (A * A).sum(1)
        if normalize_scale:
            scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent
        #assert torch.isnan(scale).sum()==0
        A += lambda_ * scale[:,:,None] * torch.eye(jac.shape[-1],device=dev)[None]
        #A += lambda_ * np.eye(jac.shape[-1])[None]
        
        #A_damped = A + lambda_ * np.eye(jac.shape[-1])[None]
        delta = torch.linalg.solve(A,b)
        params += delta
        
        # coordinate limits

        params[:,:ndims] = torch.clamp(params[:,:ndims], param_limits[None,:ndims,0], param_limits[None,:ndims,1])

        # intensity limit
        params[:,ndims:] = torch.clamp(params[:,ndims:], param_limits[None,3,0], param_limits[None,3,1])
                
        #sqerr = np.mean((I_mu-I_ev)**2)

        errtrace.append(_np(sqerr))
        traces.append(_np(params*1))

    errtrace = np.array(errtrace)        
    traces = np.array(traces).transpose((1,0,2))
    
    crlb = ndi_bg_crlb(params, I_smp_sig, mod, ndims=ndims)
    
    return params,traces,errtrace,crlb



def ndi_fit(initial, I_smp_mu, I_smp_sig, mod, param_limits, 
            ndims=2, lambda_=10, iterations = 100, normalize_scale=True):
    """
    I_mu & I_sig: [batchsize, #axes, #phase steps]
    initial: [batchsize, params: [x,y,z,I0,I1,...]]
    
    """
    nax = mod.shape[1]
    assert initial.shape[1] == ndims+nax

    if not (param_limits.shape[0] == ndims+nax and param_limits.shape[1] == 2):
        raise ValueError("param_limits must be [ndims+nax, 2]")

    params = initial*1   
    
    traces = []
    errtrace = []
    
    #with progbar(total=iterations) as pb:
    for i in range(iterations):
        I_ev, I_ev_deriv = ndi_model(params, mod, ndims)
        jac = I_ev_deriv / I_smp_sig[...,None]
        
        sqerr = (( (I_ev - I_smp_mu) / I_smp_sig )**2).mean((1,2))
        
        # now get into Ax = b form..
        # A = J^T * J + lambda * ident
        # b = J^T * (I_mu - I_ev) / I_sig
        A = torch.matmul(jac[...,None], jac[...,None,:]).sum((1,2))
        b = (jac * ((I_smp_mu - I_ev) / I_smp_sig)[...,None]).sum((1,2))
 
        scale = (A * A).sum(1)
        if normalize_scale:
            scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent
        #assert torch.isnan(scale).sum()==0
        A += lambda_ * scale[:,:,None] * torch.eye(jac.shape[-1],device=params.device)[None]
        #A += lambda_ * np.eye(jac.shape[-1])[None]
        
        #A_damped = A + lambda_ * np.eye(jac.shape[-1])[None]
        delta = torch.linalg.solve(A,b)
        params += delta
        
        # coordinate limits
        params[:,:ndims] = torch.clamp(params[:,:ndims], param_limits[None,:ndims,0], param_limits[None,:ndims,1])

        # intensity limit
        params[:,ndims:] = torch.clamp(params[:,ndims:], param_limits[None,3,0], param_limits[None,3,1])
                
        #sqerr = np.mean((I_mu-I_ev)**2)

        errtrace.append(_np(sqerr))
        traces.append(_np(params*1))

    errtrace = np.array(errtrace)        
    traces = np.array(traces).transpose((1,0,2))

    crlb = ndi_crlb(params, I_smp_sig, mod, ndims=ndims)
    
    return params,traces,errtrace,crlb


def ndi_fit_5p(initial, I_mu, I_sig, mod, param_limits, 
            ndims=2, lambda_=10, iterations = 100, normalize_scale=False):
    """
    I_mu & I_sig: [batchsize, #axes, #phase steps]
    initial: [batchsize, params: [x,y,z,I,bg]]
    """
    nax = mod.shape[1]
    params = np.zeros((len(initial), ndims+nax))
    params[:,:ndims] = initial[:,:ndims]
    params[:,ndims:ndims+nax] = initial[:,3,None]/I_mu.shape[1]

    ## translate param limits..    
    
    if ndims == 2:
        param_limits = param_limits[[0,1,3,3]]
    
    return ndi_fit(params, I_mu, I_sig, mod, param_limits, ndims, lambda_, 
                   iterations, normalize_scale=normalize_scale)

def crlb_range(photons, mod, roisize, sigma):
    expval, jac = gauss_psf_2D_fixed_sigma(theta, roisize, sigma, sigma)

if __name__ == '__main__':

    pixelsize = 65
    pitch = 221/65
    k = 2*np.pi/pitch
    mod = np.array([
               [0, k, 0, 0.95, 0, 1/6],
               [k, 0, 0, 0.95, 0, 1/6],
               [0, k, 0, 0.95, 2*np.pi/3, 1/6],
               [k, 0, 0, 0.95, 2*np.pi/3, 1/6],
               [0, k, 0, 0.95, 4*np.pi/3, 1/6],
               [k, 0, 0, 0.95, 4*np.pi/3, 1/6]
               ])

    pattern_frames = np.array([[0,2,4],[1,3,5]])
    
    roisize = 16
    param_range = torch.tensor([
        [2, roisize-3],
        [2, roisize-3],
        [1, 1e9],
        [1, 1e9],
#        [0, 1e2],
 #       [0, 1e2]
    ])
    
    param_range_bg = torch.cat([param_range, torch.tensor([[0,1e2], [0,1e2]])],0)
      
    bg=5
    numspots = 10000
    ndims = 2
    theta=[[roisize/2, roisize/2, 1,1]]
    theta=np.repeat(theta,numspots,0)
    theta[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))
    
    theta_bg=[[roisize/2, roisize/2, 1,1,bg,bg*2]]
    theta_bg=np.repeat(theta_bg,numspots,0)
    theta_bg[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))

    mod = mod[None,pattern_frames]


    #for i, ph in enumerate(photons):
        
    photons = np.logspace(2, 5, 20)
    
    models = [
        config_dict(model=ndi_bg_model, crlb=ndi_bg_crlb, fitter=ndi_bg_fit,label= 'NDI with bg', gt=theta_bg, param_range=param_range_bg),
        config_dict(model=ndi_model, crlb=ndi_crlb, fitter=ndi_fit, label='NDI', gt=theta, param_range= param_range)
        ]

    all_errs = []
    all_crlbs = []

    dev = torch.device('cuda:0')
    
    mod = torch.tensor(mod, device=dev)

    for mi in range(len(models)):
        errs = []
        crlbs = []
        for i in pb_range(len(photons)):

            m = models[mi]
            
            gt = torch.tensor(m.gt*1, device=dev)
            gt[:,2:4] = photons[i]/2
            
            I_mod, jac = m.model(gt, mod, ndims=ndims)

            I_smp_sig = 10+torch.sqrt(abs(I_mod))    # arbitrary but reasonable noise level
            I_smp_mu = torch.normal(I_mod, I_smp_sig)
            
            initial = gt*1#0.9
        
            pos,traces,errtrace = m.fitter(initial, I_smp_mu, I_smp_sig, mod, ndims=ndims,
                                 param_limits=m.param_range.to(dev), lambda_=1000, iterations=200,
                                 normalize_scale=True)
            
            #break
            crlb_m = m.crlb(gt, I_smp_sig, mod)
            err_m = pos-gt
            #print(f"std: {err.std(0)}")
            #print(f"crlb: {crlb.mean(0)}")
            
            errs.append(_np(err_m))
            crlbs.append(_np(crlb_m))        

        all_errs.append(np.array(errs))
        all_crlbs.append(np.array(crlbs))
    
    #%%
    os.makedirs(os.path.split(__file__)[0]+'/plots/',exist_ok=True)
    axes = [0, 1, 2, 3, 4]
    axes_labels=['x', 'y', 'I0', 'I1', 'bg0']
    axes_unit=['pixels','pixels', 'photons','photons', 'ph/px']
    axes_scale=[1, 1, 1, 1, 1,1]
    for i in range(len(axes)):
        ai = axes[i]
        name = axes_labels[i]

        plt.figure(dpi=150,figsize=(8,6))
        
        for j in range(len(models)):
            m = models[j]
            
            if ai >= all_errs[j].shape[2]:
                continue
            
            prec = all_errs[j].std(1)
            crlb = all_crlbs[j].mean(1)
            line, = plt.gca().plot(photons,axes_scale[i]*prec[:,ai],linewidth=2,label=f'Precision {m.label}')
            plt.plot(photons,axes_scale[i]*crlb[:,ai],label=f'CRLB {m.label}', color=line.get_color(), linestyle='--')
    
        plt.title(f'{name} axis')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Signal intensity [photons]')
        plt.ylabel(f"{name} [{axes_unit[i]}]")
        plt.grid(True)
        plt.legend()
        plt.savefig(f'plots/{name}-crlb-ndi.png')
    
    
#%%
    

    #crlb_ndi = ndi_crlb(gt, psfs[::npat], mod[:,pattern_frames])
    #crlb_g = psf.CRLB(gt)



            