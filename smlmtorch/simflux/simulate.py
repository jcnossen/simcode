import os
import torch
import numpy as np
import smlmtorch.util.multipart_tiff as tif
#import tqdm
from smlmtorch.smlm.gaussian_psf import Gaussian2DFixedSigmaPSF, GaussAstigPSFCalib, Gaussian2DAstigmaticPSF
from fastpsf import Context,CSplineMethods,CSplinePSF,GaussianPSFMethods
import smlmtorch.simflux.pattern_estimator as pe
from smlmtorch import progbar, pb_range

def blinking_spots(numspots, numframes=2000, avg_on_time = 20, on_fraction=0.1, subframe_blink=10):
    on_fraction = min(0.999, on_fraction)
    p_off = 1-on_fraction
    k_off = 1/(avg_on_time * subframe_blink)
    k_on = (k_off - p_off*k_off)/p_off
    
    print(f"p_off={p_off:.3f}, k_on={k_on:.3f}, k_off={k_off:.3f}. avg #on={on_fraction*numspots:0.0f}",flush=True)
    
    blinkstate = np.random.binomial(1, on_fraction, size=numspots)
    
    for f in range(numframes):
        spot_ontimes = np.zeros((numspots),dtype=np.float32)
        for i in range(subframe_blink):
            # todo: this thing gets strange when high on_fraction is used
            turning_on = (1 - blinkstate) * np.random.binomial(1, min(k_on,1), size=numspots)
            remain_on = blinkstate * np.random.binomial(1, 1 - k_off, size=numspots)
            blinkstate = remain_on + turning_on
            spot_ontimes += blinkstate / subframe_blink
        yield spot_ontimes
        


def simulate(tif_fn, mp: pe.ModulationPattern, psf, pos, numframes, em_drift=None, pb_cb=None, apply_poisson=True,
             intensity=1000, bg=5, width=100, device=None, return_movie=False, **blink_params):

    os.makedirs(os.path.split(tif_fn)[0], exist_ok=True)
    roisize = psf.sampleshape[-1]

    if np.isscalar(intensity):
        intensity = np.ones(len(pos))*intensity

    if return_movie:
        movie = np.zeros((numframes, width, width), dtype=np.uint16)

    with tif.MultipartTiffSaver(tif_fn) as writer:

        if device is not None:
            torch.cuda.set_device(device)

        if pb_cb is not None:
            pb = pb_cb(enumerate(blinking_spots(len(pos), numframes, **blink_params)), numframes)
        else:
            pb = progbar(enumerate(blinking_spots(len(pos), numframes, **blink_params)), total=numframes, position=0, leave=True)
        
        nspots=0
        for i, spot_ontimes in pb:
            on_pos = pos[spot_ontimes>0]
            if em_drift is not None:
                on_pos[:,:em_drift.shape[1]] += em_drift[i][None]
            
            if mp is None:
                excitation = 1
            else:
                excitation = mp.compute_excitation(start_frame_ix = [i], frame_window=1, pos = on_pos)[:,0]

            intensities = excitation * intensity[spot_ontimes>0] * spot_ontimes[spot_ontimes>0]
    
            params = np.concatenate((
                on_pos, 
                intensities[:,None], 
                np.zeros((len(on_pos),1), dtype=np.float32) ), 1)
            
            roiposYX = (params[:,[1,0]] - roisize/2).astype(np.int32)
            params[:,:2] -= roiposYX[:,[1,0]]

            img = psf.DrawROIs([width,width], roiposYX, params)
            if apply_poisson:
                img = np.random.poisson(img + bg)
            else:
                img += bg
            if return_movie:
                movie[i] = img

            writer.save(img.astype(np.uint16))
            
            nspots += len(on_pos)
            pb.set_description(f"#spots={nspots}")
            
        
    if return_movie:
        return movie
                   
       
def angles_to_mod(pitch_nm, pixelsize, angle_deg, depth, pattern_frames, z_pitch=None):
    """
    Assign mod array and phase_interp for simulation purposes
    """
    freq = 2*np.pi/np.array(pitch_nm)*pixelsize
    angle = np.deg2rad(angle_deg)
    mod = np.zeros(pattern_frames.size, dtype=pe.ModulationType)

    for i,pf in enumerate(pattern_frames):
        mod['k'][pf,0] = np.cos(angle[i]) * freq[i]
        mod['k'][pf,1] = np.sin(angle[i]) * freq[i]
        mod['k'][pf,2] = 2*np.pi/z_pitch[i] if z_pitch is not None else 0
        mod['phase'][pf] = np.linspace(0,2*np.pi,len(pf),endpoint=False)

    mod['depth'] = depth
    mod['relint'] = 1/pattern_frames.size
    return mod
            