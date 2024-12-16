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

from smlmtorch import struct


def compute_crlb(mu:Tensor, jac:Tensor, *, skip_axes:List[int]=[]):
    """
    Compute crlb from expected value and per pixel derivatives. 
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    if not isinstance(mu, torch.Tensor):
        mu = torch.Tensor(mu)

    if not isinstance(jac, torch.Tensor):
        jac = torch.Tensor(jac)
        
    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[...,axes]

    sample_dims = tuple(np.arange(1,len(mu.shape)))
        
    fisher = torch.matmul(jac[...,None], jac[...,None,:])  # derivative contribution
    fisher = fisher / mu[...,None,None]  # px value contribution
    fisher = fisher.sum(sample_dims)

    crlb = torch.zeros((len(mu),naxes), device=mu.device, dtype=mu.dtype)
    crlb[:,axes] = torch.sqrt( torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2) )
    return crlb


def ndi_bg_model(params, mod, ndims):
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
    
    phase = (x[:,None]*k[:,:,0]).sum(2,keepdims=True) - phaseshift
    exc = (1+depth*np.sin(phase)) / phasesteps
    I_modulated = I[:,:,None] * exc + bg[:,:,None]
    
    I_modulated_deriv_xyz = I[:,:,None,None] * k * (
        depth*np.cos(phase))[:,:,:,None] / phasesteps
    
    # remove z for now
    #I_modulated_deriv_xy = I_modulated_deriv_xyz[...,:2]
    
    I_deriv = np.eye(nax)[None,:,None] * exc[:,:,:,None]
    bg_deriv = np.eye(nax)[None,:,None] * (exc[:,:,:,None]*0+1)

    jac = np.concatenate([I_modulated_deriv_xyz, I_deriv, bg_deriv], -1)
    
    return I_modulated, jac



def ndi_model(params, mod, ndims):
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
    
    phase = (x[:,None]*k[:,:,0]).sum(2,keepdims=True) - phaseshift
    I = params[:,ndims:ndims+nax]
    exc = (1+depth*np.sin(phase)) / phasesteps
    I_modulated = I[:,:,None] * exc
    
    I_modulated_deriv_xyz = I[:,:,None,None] * k * (
        depth*np.cos(phase))[:,:,:,None] / phasesteps
    
    # remove z for now
    #I_modulated_deriv_xy = I_modulated_deriv_xyz[...,:2]
    
    #I_deriv = [(np.eye(nax)[i][None,:,None] * exc)[...,None] for i in range(nax)]
    I_deriv = np.eye(nax)[None,:,None] * exc[:,:,:,None]
    jac = np.concatenate([I_modulated_deriv_xyz, I_deriv], -1)
    
    return I_modulated, jac

def ndi_fixed_intensities_model(params, mod, ndims):
    """
    params: [x,y,z], fixed intensities
    """
    
    nax = mod.shape[1]

    assert len(mod.shape) == 4
    assert params.shape[1] == ndims
    
    x = params[:,:ndims]
    k = mod[:,:,:,:ndims]
    depth = mod[:,:,:,3]
    phaseshift = mod[:,:,:,4]
    
    phasesteps = mod.shape[2]
    
    phase = (x[:,None]*k[:,:,0]).sum(2,keepdims=True) - phaseshift
    exc = (1+depth*np.sin(phase)) / phasesteps
    
    I_modulated = exc
    
    I_modulated_deriv_xyz = k * (
        depth*np.cos(phase))[:,:,:,None] / phasesteps
    
    jac = I_modulated_deriv_xyz
    
    return I_modulated, jac
    
def intensity_bg_crlb(psfs, I, bg):
    
    jac = np.stack([psfs, psfs*0+1],-1)
    mu = psfs * I[:,None,None] + bg[:,None,None]
    
    fisher = np.matmul(jac[...,None], jac[...,None,:])
    fisher = fisher / mu[...,None,None]  
    fisher = fisher.sum((1,2))

    return np.sqrt( np.diagonal(np.linalg.inv(fisher), axis1=1, axis2=2) )

def ndi_crlb_5p(params, psfs, mod, ndims=2):
    """
    Computes CRLB assuming isolated PSFs defined by PSF, and background as given in params
    # params:x,y,z,I,bg
    # background is ignored and each modulation axes has an individual spot intensity
    """
    bg = params[:,4]
    
    nax = mod.shape[1]
        
    # expanded params(ep): x,y,z,I0,I1,I2
    ep = np.zeros((len(params), ndims+nax*2))
    ep[:,:ndims] = params[:,:ndims]
    ep[:,ndims:ndims+nax] = params[:,3,None] / nax 
    ep[:,ndims+nax:] = params[:,4,None]
    
    I_ev, I_ev_deriv = ndi_bg_model(ep, mod, ndims)
    
    npat = mod.shape[1]*mod.shape[2]
    # compute modulated intensities
    ibg_crlb = intensity_bg_crlb(psfs.repeat(npat,0), 
                                 I_ev.flatten(), bg.repeat(npat)).reshape( 
                                     (len(params), mod.shape[1], mod.shape[2], 2))
                                         
    I_sig = ibg_crlb[:,:,:,0]
    
    #jac = I_ev_deriv / I_sig[...,None]
    jac = I_ev_deriv

    A = np.matmul(jac[...,None], jac[...,None,:]) / I_sig[...,None,None]**2
    fisher = A.sum((1,2))
    crlb = np.sqrt( np.diagonal(np.linalg.inv(fisher), axis1=1, axis2=2) )

    return crlb        

def ndi_crlb(params, I_sig, mod, ndims=2):
    """
    Computes CRLB assuming isolated PSFs defined by PSF, and background as given in params
    # params:x,y,z,I,bg
    # background is ignored and each modulation axes has an individual spot intensity
    """
    nax = mod.shape[1]
    assert params.shape[1] == ndims+nax
    assert len(I_sig.shape) == 3 # [batchsize, nax, nphase]
    
    I_ev, jac = ndi_model(params, mod, ndims)

    A = np.matmul(jac[...,None], jac[...,None,:]) / I_sig[...,None,None]**2
    fisher = A.sum((1,2))

    crlb = np.sqrt( np.diagonal(np.linalg.inv(fisher), axis1=1, axis2=2) )

    return crlb        


def ndi_bg_crlb(params, I_sig, mod, ndims=2):
    """
    Computes CRLB assuming isolated PSFs defined by PSF, and background as given in params
    # params:x,y,z,I,bg
    # background is ignored and each modulation axes has an individual spot intensity
    """
    nax = mod.shape[1]
    assert params.shape[1] == ndims+nax*2
    assert len(I_sig.shape) == 3 # [batchsize, nax, nphase]
    
    I_ev, jac = ndi_bg_model(params, mod, ndims)

    A = np.matmul(jac[...,None], jac[...,None,:]) / I_sig[...,None,None]**2
    fisher = A.sum((1,2))
    crlb = np.sqrt( np.diagonal(np.linalg.inv(fisher), axis1=1, axis2=2) )
    return crlb        



def ndi_bg_fit(initial, I_smp_mu, I_smp_sig, mod, param_limits, 
            ndims=2, lambda_=10, iterations = 100, normalize_scale=True):
    """
    I_mu & I_sig: [batchsize, #axes, #phase steps]
    initial: [batchsize, params: [x,y,z,I0,I1,...,b0,b1...]]
    
    """
    nax = mod.shape[1]
    assert initial.shape[1] == ndims+nax*2
    assert param_limits.shape[0] == ndims+nax*2 and param_limits.shape[1] == 2

    params = initial*1   
    sampledims = tuple([i for i in range(1,len(I_smp_mu.shape))])
    
    traces = []
    errtrace = []
    
    #with tqdm.tqdm(total=iterations) as pb:
    for i in range(iterations):
        I_ev, I_ev_deriv = ndi_bg_model(params, mod, ndims)
        jac = I_ev_deriv / I_smp_sig[...,None]
        
        sqerr = (( (I_ev - I_smp_mu) / I_smp_sig )**2).mean((1,2))
        
        # now get into Ax = b form..
        # A = J^T * J + lambda * ident
        # b = J^T * (I_mu - I_ev) / I_sig
        A = np.matmul(jac[...,None], jac[...,None,:]).sum(sampledims)
        b = (jac * ((I_smp_mu - I_ev) / I_smp_sig)[...,None]).sum(sampledims)
 
        scale = (A * A).sum(1)
        if normalize_scale:
            scale /= scale.mean(1, keepdims=True) # normalize so lambda scale is not model dependent
        #assert torch.isnan(scale).sum()==0
        A += lambda_ * scale[:,:,None] * np.eye(jac.shape[-1])[None]
        #A += lambda_ * np.eye(jac.shape[-1])[None]
        
        #A_damped = A + lambda_ * np.eye(jac.shape[-1])[None]
        delta = np.linalg.solve(A,b)
        params += delta
        
        # coordinate limits
        params[:,:ndims] = np.clip(params[:,:ndims], param_limits[None,:ndims,0], param_limits[None,:ndims,1])

        # intensity limit
        params[:,ndims:] = np.clip(params[:,ndims:], param_limits[None,3,0], param_limits[None,3,1])
                
        #sqerr = np.mean((I_mu-I_ev)**2)

        errtrace.append(sqerr)
        traces.append(params)

    errtrace = np.array(errtrace)        
    traces = np.array(traces).transpose((1,0,2))
    
    return params,traces,errtrace



def ndi_fit(initial, I_smp_mu, I_smp_sig, mod, param_limits, 
            ndims=2, lambda_=10, iterations = 100, normalize_scale=True):
    """
    I_mu & I_sig: [batchsize, #axes, #phase steps]
    initial: [batchsize, params: [x,y,z,I0,I1,...,b0,b1...]]
    
    """
    nax = mod.shape[1]
    assert initial.shape[1] == ndims+nax
    assert param_limits.shape[0] == ndims+nax and param_limits.shape[1] == 2

    params = initial*1   
    sampledims = (1,2)
    
    traces = []
    errtrace = []
    
    #with tqdm.tqdm(total=iterations) as pb:
    for i in range(iterations):
        I_ev, I_ev_deriv = ndi_model(params, mod, ndims)
        jac = I_ev_deriv / I_smp_sig[...,None]
        
        sqerr = (( (I_ev - I_smp_mu) / I_smp_sig )**2).mean((1,2))
        
        # now get into Ax = b form..
        # A = J^T * J + lambda * ident
        # b = J^T * (I_mu - I_ev) / I_sig
        A = np.matmul(jac[...,None], jac[...,None,:]).sum(sampledims)
        b = (jac * ((I_smp_mu - I_ev) / I_smp_sig)[...,None]).sum(sampledims)
 
        scale = (A * A).sum(1)
        if normalize_scale:
            scale /= scale.mean(1, keepdims=True) # normalize so lambda scale is not model dependent
        #assert torch.isnan(scale).sum()==0
        A += lambda_ * scale[:,:,None] * np.eye(jac.shape[-1])[None]
        #A += lambda_ * np.eye(jac.shape[-1])[None]
        
        #A_damped = A + lambda_ * np.eye(jac.shape[-1])[None]
        delta = np.linalg.solve(A,b)
        params += delta
        
        # coordinate limits
        params[:,:ndims] = np.clip(params[:,:ndims], param_limits[None,:ndims,0], param_limits[None,:ndims,1])

        # intensity limit
        params[:,ndims:] = np.clip(params[:,ndims:], param_limits[None,3,0], param_limits[None,3,1])
                
        #sqerr = np.mean((I_mu-I_ev)**2)

        errtrace.append(sqerr)
        traces.append(params)

    errtrace = np.array(errtrace)        
    traces = np.array(traces).transpose((1,0,2))
    
    return params,traces,errtrace


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
    param_range = np.array([
        [2, roisize-3],
        [2, roisize-3],
        [1, 1e9],
        [1, 1e9],
#        [0, 1e2],
 #       [0, 1e2]
    ])
    
    param_range_bg = np.concatenate([param_range, [[0,1e2], [0,1e2]]],0)
      
    bg=5
    numspots = 2000
    ndims = 2
    theta=[[roisize/2, roisize/2, 1,1]]
    theta=np.repeat(theta,numspots,0)
    theta[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))
    
    theta_bg=[[roisize/2, roisize/2, 1,1,bg,bg*2]]
    theta_bg=np.repeat(theta_bg,numspots,0)
    theta_bg[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))

    mod = mod[None,pattern_frames]


    #for i, ph in enumerate(photons):
        
    photons = np.logspace(2, 5, 10)
    
    models = [
        struct(model=ndi_bg_model, crlb=ndi_bg_crlb, fitter=ndi_bg_fit,label= 'NDI with bg', gt=theta_bg, param_range=param_range_bg),
        struct(model=ndi_model, crlb=ndi_crlb, fitter=ndi_fit, label='NDI', gt=theta, param_range= param_range)
        ]

    all_errs = []
    all_crlbs = []


    for mi in range(len(models)):
        errs = []
        crlbs = []
        for i in tqdm.trange(len(photons)):

            m = models[mi]
            
            gt = m.gt*1
            gt[:,2:4] = photons[i]/2
            
            I_mod, jac = m.model(gt, mod, ndims=ndims)
            I_smp_sig = 10+np.sqrt(abs(I_mod))    # arbitrary but reasonable noise level
            I_smp_mu = np.random.normal(I_mod, I_smp_sig)
            
            initial = gt*1#0.9
        
            pos,traces,errtrace = m.fitter(initial, I_smp_mu, I_smp_sig, mod, ndims=ndims,
                                 param_limits=m.param_range, lambda_=1, iterations=80,
                                 normalize_scale=True)
            
            #break
            crlb_m = m.crlb(gt, I_smp_sig, mod)
            err_m = pos-gt
            #print(f"std: {err.std(0)}")
            #print(f"crlb: {crlb.mean(0)}")
            
            errs.append(err_m)
            crlbs.append(crlb_m)        

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



            