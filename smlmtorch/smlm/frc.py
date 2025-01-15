# -*- coding: utf-8 -*-

import numpy as np
from fastpsf import Context,GaussianPSFMethods
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import convolve
import torch
import time

def _getfft(xy,photons,imgshape,zoom,ctx:Context,device=None):
    spots = np.zeros((len(xy),5))
    spots[:,[0,1]] = xy * zoom
    spots[:,4] = photons 
    spots[:,[2,3]] = 0.5
    
    w = np.max(imgshape)*zoom
    img = np.zeros((w,w))
    img = GaussianPSFMethods(ctx).Draw(img, spots)
    img = np.array(img, dtype=np.float32)
    
    # Image width / Width of edge region
    wnd = tukey(w, 1/4).astype(np.float32)
    #plt.plot(wnd)
    img = (img * wnd[:,None]) * wnd[None,:]
    
    if device is not None:
        img = torch.tensor(img, device=device)
        f_img = torch.fft.fftshift(torch.fft.fft2(img)).cpu().numpy()
    else:
        f_img = np.fft.fftshift(np.fft.fft2(img))
    return f_img

    #return f_img.cpu().numpy()

def radialsum(sqimg):
    W = len(sqimg)
    Y,X = np.indices(sqimg.shape)
    R = np.sqrt((X-W//2)**2+(Y-W//2)**2)
    R = R.astype(np.int32)
    return np.bincount(R.ravel(), sqimg.ravel()) # / np.bincount(R.ravel())

def radialsum_(img):
    center = np.array([(img.shape[0]-1)//2, (img.shape[1]-1)//2])
    y, x = np.ogrid[ :img.shape[0], :img.shape[1]]
    dists = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    dists = np.round(dists).astype(int)
    unique_dists, unique_indices = np.unique(dists, return_inverse=True)
    radial_sums = np.bincount(unique_indices, weights=img.ravel())
    return radial_sums

def FRC(xy, photons, zoom, imgshape, pixelsize, display=True, smooth=0, mask=None,device=None):
    if type(xy) == list and len(xy) == 2:
        #assert len(photons) == 2
        ...
    else:
        if mask is None:
            mask = np.random.binomial(1, 0.5, len(xy))==1
                   
        set1 = mask
        set2 = np.logical_not(set1)
        xy = [xy[set1],xy[set2]]
        photons = [photons[set1],photons[set2]]

    t0 = time.time()

    with Context() as ctx:
        f1 = _getfft(xy[0],photons[0],imgshape,zoom,ctx,device=device)
        f2 = _getfft(xy[1],photons[1],imgshape,zoom,ctx,device=device)

    x = np.real(f1*np.conj(f2))
    frc_num = radialsum(x)
    frc_denom = np.sqrt(radialsum(np.abs(f1)**2) * radialsum(np.abs(f2)**2))
    
    frc = frc_num / frc_denom
    
    t1 = time.time()
    
    freq = np.fft.fftfreq(len(f1))
    
    frc = frc[:imgshape[0]*zoom//2]
    freq = freq[:imgshape[0]*zoom//2]

    if smooth > 0:
        frc = convolve(frc, np.ones(smooth)/smooth, mode='valid')
        freq = convolve(freq, np.ones(smooth)/smooth, mode='valid')
        
    b = np.where(frc<1/7)[0]
    frc_res =  freq[b[0]] if len(b)>0 else freq[0] 
    
    print(f"Elapsed time: {t1-t0:.1f} s. FRC={pixelsize / (zoom*frc_res):.2f} nm")

    if display: 
        plt.figure()
        plt.plot(freq * zoom / pixelsize, frc)
        plt.axhline(1/7, color='r', linestyle='--')
        plt.title(f'FRC resolution: {pixelsize / (zoom*frc_res):.2f} nm ({1/(zoom*frc_res):.2f} px)')
        plt.xlabel('Frequency [1/nm]')
    
    return pixelsize / (zoom*frc_res), frc, freq*zoom/pixelsize
    
if __name__ == "__main__":
    
    #fn= 'C:/data/simflux/sim4_1/results/sim4_1/g2d-dc-fbp10.hdf5' 
    #fn  = 'C:/data/simflux/sim4_1/results/sim4_1/sf-dc-fbp10.hdf5' 
    #fn='C:/data/drift/gatta RY/3.hdf5'
    
    #fn = 'C:/data/simflux/sim4_1/results/sim4_1/smlm_crlb15.hdf5'
    fn='C:/dev/smlmtorch/scripts/densities3/D1/results/sim_tubules_bg2.5_I500_psf_gauss1.3px/sfhd-ndi+smlm.hdf5'
    
    from smlmtorch import Dataset
    ds = Dataset.load(fn)
        
    #ds[mask].save('test1.hdf5')
    #ds[np.logical_not(mask)].save('test2.hdf5')

    ds=ds[ds.frame%6==0]
    
    frc,frc_res = FRC(ds.pos, ds.photons, 25
                      , ds.imgshape, 
                      pixelsize=108.3,
                      #mask=mask,
                      device=torch.device('cuda:0'))
    