# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
from smlmtorch import config_dict

class PerPixelCameraCalibration:
    def __init__(self, fn):
        with open(fn, "rb") as s:
            d = pickle.load(s)
        
        self.baseline = d['baseline']
        self.darkcurrent = d['darkcurrent']
        self.read_noise = d['read_noise']
        self.thermal_noise = d['thermal_noise']
        self.gain = d['gain']
        self.gain_error = d['gain_error']
        self.aoi = config_dict(d['aoi'])
        
    @staticmethod
    def get_roi_calib(info_yaml_fn, camera_calib_fn, roi_index=0, plot=False):
        """
        This reads out a yaml file that contains the camera AOI, exposure time and ROI,
        if the camera calibration file contains that pixel area, returns the per-pixel calibration data.
        """
        with open(info_yaml_fn, "r") as s:
            info = yaml.safe_load(s)
            info = config_dict.from_dict(info)
            # rois have format: x,y,w,h
            #print(info)

        calib = PerPixelCameraCalibration(camera_calib_fn)
        print(f"Calibration data available for AOI: {calib.aoi}")
        
        offset_y = info.aoi.bottom - calib.aoi.bottom
        offset_x = info.aoi.left - calib.aoi.left

        if len(info ['rois']) == 0:
            roi = [0,0,info.aoi.width,info.aoi.height]
        else:
            roi = info['rois'][roi_index]
            
        roi[0] += offset_x
        roi[1] += offset_y
        
        offset, gain, readnoise = calib.compute_calib_matrices(info['exposure'], roi)
        
        if plot:
            calib.plot_rois(roi)
        
        return offset, gain, readnoise
        
    def compute_calib_matrices(self, exposure_s, roi = None):
        """
        returns baseline, gain, readnoise for given ROI [x,y,w,h]
        """
        
        if roi is None:
            roi_ix = np.s_[:,:]
        else:
            shape = self.gain.shape
            assert roi[1]+roi[3] < shape[0]
            assert roi[0]+roi[2] < shape[1]
            roi_ix = np.s_[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
        baseline = self.baseline[roi_ix] + exposure_s * self.darkcurrent[roi_ix]
        gain = self.gain[roi_ix]
        gain[:,:] = gain.mean()
        readnoise = self.read_noise[roi_ix] + exposure_s * self.thermal_noise[roi_ix]
        readnoise *= gain**2
        
        return baseline, gain, readnoise

        
    def plot_rois(self, roi):
        ix = np.s_[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        plt.figure()
        plt.imshow(self.baseline[ix])
        plt.title('Baseline')
        plt.colorbar()
    
        plt.figure()
        plt.imshow(self.darkcurrent[ix])
        plt.title('Dark current')
        plt.colorbar()
    
        plt.figure()
        plt.imshow(self.read_noise[ix])
        plt.title('Read noise')
        plt.colorbar()
    
        plt.figure()
        plt.imshow(self.thermal_noise[ix])
        plt.title('Thermal noise')    
        plt.colorbar()
        
        plt.figure()
        plt.imshow(self.gain[ix])
        plt.colorbar()
        plt.title('Gain')
    
        plt.figure()
        plt.imshow(self.gain_error[ix])
        plt.colorbar()
        plt.title('Gain Error')
        
        
if __name__ == '__main__':
    info_fn = 'C:/data/sim-decode/june/singlechan/1000mW/1/info.yaml'
    calib_fn = 'C:/data/multicolor/camera_calib.pickle'

    calib = PerPixelCameraCalibration.get_roi_calib(info_fn, calib_fn, plot=True)
    