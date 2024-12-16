""" 
Dataset class to manage localization dataset. Picasso compatible HDF5 and thunderstorm CSV are supported

photonpy - Single molecule localization microscopy library
© Jelmer Cnossen 2018-2021
"""# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from smlmtorch.util.multipart_tiff import MultipartTiffSaver
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from smlmtorch.smlm.frc import FRC
from smlmtorch.util.progbar import progbar,pb_range

from numpy.lib.recfunctions import append_fields
import h5py 
import yaml

class Dataset:
    """
    Keep a localization dataset using a numpy structured array
    """
    def __init__(self, length, dims, imgshape, data=None, config=None, haveSigma=False, extraFields=None, **kwargs):

        assert len(imgshape) == 2
        self.imgshape = np.array(imgshape)
        self.config = config if config is not None else {}

        if data is not None:
            self.data = np.copy(data).view(np.recarray)
        else:
            dtype = self.createDTypes(dims, len(imgshape), includeGaussSigma=haveSigma, extraFields=extraFields)
            self.data = np.recarray(length, dtype=dtype)
            self.data.fill(0)

        self.sigma = np.zeros(dims)
                
        if kwargs is not None:
            self.config = {**self.config, **kwargs}
            
    def addFields(self, fieldDTypes):
        dt = self.data.dtype.descr
        for f in fieldDTypes:
            dt.append(f)

        ndata = np.recarray(len(self), dtype=dt)
        ndata.fill(0)
        
        fieldnames = self.data.dtype.fields.keys()
        for n in fieldnames:
            ndata[n] = self.data[n]
        
        self.data = ndata
        
        if not 'extraFields' in self.config:
            self.config['extraFields']=[]
                
        for f in fieldDTypes:
            self.config['extraFields'].append(f)
                    
    @staticmethod
    def merge(datasets, set_groups=False):
        ds1 = datasets[0]
        n = sum([len(ds) for ds in datasets])
        copy = type(ds1)(n, ds1.dims, ds1.imgshape, config=ds1.config)
        j = 0
        for i,ds in enumerate(datasets):
            copy.data[j:j+len(ds)] = ds.data
            if set_groups:
                copy.data.group[j:j+len(ds)] = i
            j += len(ds)
        return copy
        
    def __add__(self, other):
        return Dataset.merge([self,other])

    def __getitem__(self,idx):
        if type(idx) == str:
            return self.config[idx]
        else:
            indices = idx
            return type(self)(length=0, 
                              dims=self.dims, 
                              imgshape=self.imgshape, 
                              data=self.data[indices], 
                              config=self.config)

    def __setitem__(self, idx, val):
        if type(idx) == str:
            self.config[idx]=val
        else:
            if not isinstance(val, Dataset):
                raise ValueError('dataset[val] __setitem__ operator expects another dataset')
            
            if len(val) != len(self.data[idx]):
                raise ValueError('dataset __setitem__ operator: Lengths do not match, expecting {len(self.data[idx])} received {len(val)}')
            
            self.data[idx] = val.data

    def copy(self):
        return self[:]
    
    def createDTypes(self,dims, imgdims, includeGaussSigma, extraFields=None, extraEstimFields=None):
        """
        Can be overriden to add columns
        """

        if extraFields is not None:
            self.config['extraFields'] = extraFields
        if extraEstimFields is not None:
            self.config['extraEstimFields'] = extraEstimFields

        dtypeEstim = [
            ('pos', np.float32, (dims,)),
            ('photons', np.float32),
            ('bg', np.float32)]
        
        if includeGaussSigma:
            dtypeEstim.append(
                ('sigma', np.float32, (2,))
            )
            
        if extraEstimFields is not None:
            dtypeEstim.extend(extraEstimFields)

        dtypeEstim = np.dtype(dtypeEstim)
        
        dtypeLoc = [
            ('roi_id', np.int32),  # typically used as the original ROI index in an unfiltered dataset
            ('frame', np.int32),
            ('estim', dtypeEstim),
            ('crlb', dtypeEstim),
            ('chisq', np.float32),
            ('group', np.int32),
            ('roipos', np.int32, (imgdims,))
            ]

        if extraFields is not None:
            dtypeLoc.extend(extraFields)
        
        return np.dtype(dtypeLoc)

    def hasPerSpotSigma(self):
        return 'sigma' in self.data.estim.dtype.fields
            
    def filter(self, indices):
        """
        Keep only the selected indices, filtering in-place. An alternative is doing a copy: ds2 = ds[indices]
        """
        prevcount = len(self)
        self.data = self.data[indices]
        
        print(f"Keeping {len(self.data)}/{prevcount} spots")
    
    def filterNaN(self):
        hasnan = self.hasNaN()
        
        print(f"Removing {hasnan.sum()} localizations with NaN")
        self.data = self.data[np.logical_not(hasnan)]
        
    def crlb_filter(self, max_crlb_xy, max_crlb_z=None, min_photons=None, inplace=True):

        sel = np.all(self.crlb.pos[:,:2]<np.array(max_crlb_xy)[None],1)
                
        if max_crlb_z is not None:
            sel = sel & (self.crlb.pos[:,2]<max_crlb_z)
        
        if min_photons is not None:
            sel = sel & (self.photons > min_photons)
        
        if inplace:
            self.filter(sel)
        else:
            copy = self[:]
            copy.filter(sel)
            return copy
    
    def hasNaN(self):
        return  (np.isnan(self.pos).any(1) | 
                  np.isnan(self.photons) | 
                  np.isnan(self.background) |
                  np.isinf(self.photons) |
                  np.isinf(self.background))
    
    @property
    def numFrames(self):
        if len(self) == 0:
            return 0
        
        return np.max(self.data.frame)+1
    
    @property
    def shape(self):
        """ return [number of locs, number of dimensions]"""
        return self.pos.shape
            
    def indicesPerFrame(self):
        frame_indices = self.data.frame
        if len(frame_indices) == 0: 
            numFrames = 0
        else:
            numFrames = np.max(frame_indices)+1
        frames = [[] for i in range(numFrames)]
        for k in range(len(self.data)):
            frames[frame_indices[k]].append(k)
        for f in range(numFrames):
            frames[f] = np.array(frames[f], dtype=int)
        return frames
            
    def __len__(self):
        return len(self.data)
    
    @property
    def dims(self):
        return self.data.estim.pos.shape[1]
    
    @property
    def pos(self):
        return self.data.estim.pos
    
    @pos.setter
    def pos(self, val):
        self.data.estim.pos = val
    
    @property
    def crlb(self):
        return self.data.crlb
    
    @property
    def photons(self):
        return self.data.estim.photons

    @photons.setter
    def photons(self, val):
        self.data.estim.photons = val
    
    @property
    def background(self):
        return self.data.estim.bg

    @background.setter
    def background(self,val):
        self.data.estim.bg = val

    @property
    def bg(self):
        return self.data.estim.bg

    @bg.setter
    def bg(self,val):
        self.data.estim.bg = val
    
    @property
    def frame(self):
        return self.data.frame
    
    @frame.setter
    def frame(self,val):
        self.data.frame = val
    
    @property
    def chisq(self):
        return self.data.chisq
    
    @chisq.setter
    def chisq(self,val):
        self.data.chisq = val
    
    @property
    def roi_id(self):
        return self.data.roi_id
    
    @property
    def group(self):
        return self.data.group
    
    @group.setter
    def group(self,val):
        self.data.group = val
    
    @property
    def local_pos(self):
        """
        Localization position within ROI.
        """
        lpos = self.pos*1
        lpos[:,0] -= self.data.roipos[:,-1]
        lpos[:,1] -= self.data.roipos[:,-2]
        return lpos
        
    def __repr__(self):
        return f'Dataset with {len(self)} {self.dims}D localizations ({self.imgshape[1]}x{self.imgshape[0]} pixel image).'
    
    def estimateDriftMinEntropy(self, framesPerBin=10, initialEstimate=None, estimatePrecision=False, apply=False, dims=None, pixelsize=None, returnCV=False, **kwargs):
        if dims is None:
            dims = self.dims

        if pixelsize is None:
            pixelsize = self['pixelsize']
            
        from dme.dme import dme_estimate

        r = dme_estimate(self.data.estim.pos[:,:dims], 
                   framenum = self.data.frame,
                   estimatePrecision=estimatePrecision,
                   crlb = self.data.crlb.pos[:,:dims],
                   framesperbin=framesPerBin, 
                   initialEstimate=initialEstimate,
                   imgshape=self.imgshape, pixelsize=pixelsize, **kwargs)
        
        if estimatePrecision:
            drift, drift_sets  = r
        else:
            drift = r
            
        if apply:
            self.applyDrift(drift)
            
        if returnCV:
            assert estimatePrecision
            return drift, drift_sets

        return drift
        
    def applyDrift(self, drift):
        if drift.shape[1] != self.dims:
            print(f"Applying {drift.shape[1]}D drift to {self.dims}D localization dataset")
        
        sel = self.data.frame < drift.shape[0]
        if sel.sum() < len(self):
            print(f"Warning: {len(self)-sel.sum()} localizations are outside the drift correction range")
        self.data.estim.pos[:,:drift.shape[1]][sel] -= drift[self.data.frame[sel]]
        self.config['drift'] = drift
        
    @property
    def isDriftCorrected(self):
        return 'drift' in self.config
        
    def undoDrift(self):
        if not self.isDriftCorrected:
            raise ValueError('No drift correction has been done on this dataset')
        
        drift = self['drift']
        self.data.estim.pos[:,:drift.shape[1]] += drift[self.frame]
        
    def _xyI(ds):
        r=np.zeros((len(ds),3))
        r[:,:2] = ds.pos[:,:2]
        r[:,2] = ds.photons
        return r

    
    def estimateDriftRCC(self, framesPerBin=500, zoom=1, dims=2, useCuda=True):
        from dme.rcc import rcc,rcc3D
        from dme.native_api import NativeAPI
        
        with NativeAPI(useCuda=useCuda) as dll:
            if dims == 2:
                return rcc(self.pos, self.frame, int(self.numFrames/framesPerBin), dll=dll,
                             zoom=zoom)[0]
            
            else:
                assert dims==3
                return rcc3D(self.pos, self.frame, int(self.numFrames/framesPerBin), dll=dll,zoom=zoom)

    
    def align(self, other, zoom=1):
        from dme.rcc import rcc

        xyI = np.concatenate([self._xyI(), other._xyI()])
        framenum = np.concatenate([np.zeros(len(self),dtype=np.int32), np.ones(len(other),dtype=np.int32)])
        
        return 2*rcc(xyI, framenum, 2, maxdrift=10,zoom=zoom,RCC=False)[0][1]

    @property
    def fields(self):
        return self.data.dtype.fields

           
    def scale(self, s):
        s=np.array(s)
        self.pos *= s[None,:]
        self.crlb.pos *= s[None,:]
    
    def save(self, fn, **kwargs):
        ext = os.path.splitext(fn)[1]
        if ext == '.hdf5':
            self.saveHDF5(fn, **kwargs)
        elif ext == '.3dlp':
            return self.saveVisp3DLP(fn)
        elif ext == '.csv': # thunderstorm compatible csv
            return self.saveCSVThunderStorm(fn)
        elif ext == '.npy':
            self.saveNumpy(fn)
        else:
            raise ValueError('unknown extension')
        
    def saveNumpy(self,fn):
        with open(fn, "wb") as s:
            np.save(s, self.imgshape)
            np.save(s, self.data)
        
    @classmethod
    def loadNumpy(cls,fn,**kwargs):
        with open(fn, "rb") as s:
            imgshape = np.load(s)
            data = np.load(s)
            data = np.recarray(data.shape,data.dtype,buf=data)
            ds = cls(len(data), data.estim.pos.shape[1], imgshape, data=data,**kwargs)
            ds['locs_path'] = fn
        return ds
            
    def saveVisp3DLP(self,fn):
        # x,y,z,lx,ly,lz,i,f
        data = np.zeros((len(self),8),dtype=np.float32)
        data[:,:3] = self.pos
        data[:,3:6] = self.crlb.pos
        data[:,6] = self.photons
        data[:,7] = self.frame
        
        np.savetxt(fn, data, fmt='%.3f %.3f %.3f %.3f %.3f %.3f %.0f %d')
        

    def saveHDF5(self, fn, saveGroups=False, customFields=[]):
        """
        This saves to picasso-format HDF5 with some extra hacks to store additional data.
        
        TODO: Refactor this mess
        """
        print(f"Saving Picasso-compatible hdf5 to {fn}")
         
        
        #if 'extraFields' in self.config:
        #    for ef in self.config['extraFields']:
        #        fields.append(ef[0])

        fields = customFields#self.config['fields'] if 'extraFields'
        estimFields = self.config['extraEstimFields'] if 'extraEstimFields' in self.config else []
    
        with h5py.File(fn, 'w') as f:
            dtype = [('frame', '<u4'), 
                     ('x', '<f4'), ('y', '<f4'),
                     ('photons', '<f4'), 
                     ('sx', '<f4'), ('sy', '<f4'), 
                     ('bg', '<f4'), 
                     ('lpx', '<f4'), ('lpy', '<f4'), 
                     ('lI', '<f4'), ('lbg', '<f4'), 
                     ('ellipticity', '<f4'), 
                     ('net_gradient', '<f4'),
                     ('roi_index', '<i4'),
                     ('chisq', '<f4')]
            
            if saveGroups:
                dtype.append(('group', '<u4'))

            if 'sigma' in self.data.estim.dtype.fields:
                dtype.append(('lsx', '<f4'))
                dtype.append(('lsy', '<f4'))
                
            for fld in fields:
                dtype.append((fld, self.data.dtype[fld]))
            
            """
            for fld,fld_dtype,shape in estimFields:
                dtype.append((fld + "_estim", self.data.estim.dtype[fld]))
                dtype.append((fld + "_crlb", self.data.crlb.dtype[fld]))
            """
            
            if self.dims==3:
                for fld in [('z', '<f4'), ('lpz', '<f4')]:
                    dtype.append(fld)
            
            locs = f.create_dataset('locs', shape=(len(self),), dtype=dtype)
            locs['frame'] = self.frame
            locs['x'] = self.pos[:,0]
            locs['y'] = self.pos[:,1]
            locs['lpx'] = self.crlb.pos[:,0]
            locs['lpy'] = self.crlb.pos[:,1]
            
            for fld in fields:
                locs[fld] = self.data[fld]

            """
            for fld,fld_dtype,shape in estimFields:
                # weird workaround for https://github.com/h5py/h5py/issues/645
                rows = locs[:]
                rows[fld + "_estim"] = self.data.estim[fld]
                rows[fld + "_crlb"] = self.data.crlb[fld]
                locs[:] = rows
            """

            if self.dims==3:
                locs['z'] = self.pos[:,2]
                locs['lpz'] = self.crlb.pos[:,2]
                        
            locs['photons'] = self.photons
            locs['bg'] = self.background
            if 'sigma' in self.data.estim.dtype.fields:
                locs['sx'] = self.data.estim.sigma[:,0]
                locs['sy'] = self.data.estim.sigma[:,1]
                locs['lsx'] = self.crlb.sigma[:,0]
                locs['lsy'] = self.crlb.sigma[:,1]
            locs['lI'] = self.crlb.photons,
            locs['lbg'] = self.crlb.bg
            locs['net_gradient'] = 0
            locs['chisq'] = self.chisq
            locs['roi_index'] = self.data.roi_id # index into original un-filtered list of detected ROIs
            
            if saveGroups:
                locs['group'] = self.data.group
                                
            info =  {'Byte Order': '<',
                     'Camera': 'Dont know' ,
                     'Data Type': 'uint16',
                     'File': fn,
                     'Frames': int(np.max(self.frame)+1 if len(self.frame)>0 else 0),
                     'Width': int(self.imgshape[1]),
                     'Height': int(self.imgshape[0])
                     }
            
            info_fn = os.path.splitext(fn)[0] + ".yaml" 
            with open(info_fn, "w") as file:
                yaml.dump(info, file, default_flow_style=False) 
                                
    def saveCSVThunderStorm(self, fn):
        """
        Save thunderstorm compatible CSV file
        """
        #"frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","offset [photon]","bkgstd [photon]","chi2","uncertainty [nm]"
        fields = ['frame', 'x', 'y', 'sigma', 'intensity', 'offset', 'uncertainty', 'crlbx', 'crlby']
        if 'extraFields' in self.config:
            for f in self.config['extraFields']:
                fields.append(f[0])
            
        data = np.zeros((len(self),len(fields)))
        data[:,0] = self.frame
        data[:,1:3] = self.pos[:,:2]
        data[:,3] = np.mean(self.data.estim.sigma,1)
        data[:,4] = self.photons
        data[:,5] = self.background
        data[:,6] = np.mean(self.crlb.pos[:,:2],1)
        data[:,7] = self.crlb.pos[:,0]
        data[:,8] = self.crlb.pos[:,1]
        if 'extraFields' in self.config:
            for i,f in enumerate(self.config['extraFields']):
                data[:,7+i] = self.data[f[0]]
        
        header= ','.join([f'"{v}"' for v in fields])
        np.savetxt(fn, data, fmt='%.6f', delimiter=',', header=header, comments='')
    
    def autocrop(self):
        minpos = np.min(self.pos,0)
        maxpos = np.max(self.pos,0)
        print(f"Min: {minpos}, max: {maxpos}")
        return self.crop(minpos,maxpos)
    
    def cropFromEdge(self, dist, silent=False):
        return self.crop([dist,dist],[self.imgshape[1]-dist,self.imgshape[0]-dist])
    
    def crop(self, minpos_xy, maxpos_xy, silent=False):
        minpos = np.array(minpos_xy)
        maxpos = np.array(maxpos_xy)
        which = (self.pos[:,:len(minpos)] >= minpos[None]) & (self.pos[:,:len(maxpos)] <= maxpos[None])
        which = np.all(which,1)
        ds = self[which]
        ds.imgshape = np.ceil((maxpos-minpos)[[1,0]]).astype(int) # imgshape has array index order instead of x,y,z
        ds.pos[:,:len(minpos)] -= minpos[None]
        if not silent:
            print(f"Cropping dataset. New shape: {ds.imgshape}, keeping {np.sum(which)}/{len(self)} spots")
        
        return ds
    
    def distanceToBorder(self):
        
        dist = self.pos[:,0]
        dist = np.minimum(dist, self.imgshape[1]-1-self.pos[:,0])
        dist = np.minimum(dist, self.pos[:,1])
        dist = np.minimum(dist, self.imgshape[0]-1-self.pos[:,1])
        
        return dist
    
    @classmethod
    def loadHDF5(cls, fn, **kwargs):
        
        with h5py.File(fn, 'r') as f:
            locs = f['locs'][:]
                        
            info_fn = os.path.splitext(fn)[0] + ".yaml" 
            with open(info_fn, "r") as file:
                if hasattr(yaml, 'unsafe_load'):
                    obj = yaml.unsafe_load_all(file)
                else:
                    obj = yaml.load_all(file)
                obj=list(obj)[0]
                imgshape=np.array([obj['Height'],obj['Width']])

            if 'z' in locs.dtype.fields:
                dims = 3
            else:
                dims = 2

            haveSigma = 'sx' in locs.dtype.fields
            ds = cls(len(locs), dims, imgshape, haveSigma = haveSigma, **kwargs)
            
            """
            estimFields = ds.config['extraEstimFields'] if 'extraEstimFields' in ds.config else []
            for fld,fld_dtype,shape in estimFields:
                ds.data.estim[fld] = locs[fld + "_estim"]
                ds.data.crlb[fld] = locs[fld + "_crlb"]
            """

            ds.photons[:] = locs['photons']
            ds.background[:] = locs['bg']
            ds.pos[:,0] = locs['x']
            ds.pos[:,1] = locs['y']
            if dims==3: 
                ds.pos[:,2] = locs['z']
                ds.crlb.pos[:,2] = locs['lpz']

            if 'lpx' in locs.dtype.fields:
                ds.crlb.pos[:,0] = locs['lpx']
            ds.crlb.pos[:,1] = locs['lpy']
            if 'lI' in locs.dtype.fields:
                ds.crlb.photons = locs['lI']
            if 'lbg' in locs.dtype.fields:
                ds.crlb.bg = locs['lbg']

            if 'lsx' in locs.dtype.fields: # picasso doesnt save crlb for the sigma fits
                ds.crlb.sigma[:,0] = locs['lsx']
                ds.crlb.sigma[:,1] = locs['lsy']

            if ds.hasPerSpotSigma():
                ds.data.estim.sigma[:,0] = locs['sx']
                ds.data.estim.sigma[:,1] = locs['sy']
            
            ds.frame[:] = locs['frame']
            
            if 'chisq' in locs.dtype.fields:
                ds.data.chisq = locs['chisq']
            
            if 'group' in locs.dtype.fields:
                ds.data.group = locs['group']
            
        ds['locs_path'] = fn
        return ds
    
    @classmethod
    def loadCSVThunderStorm(cls, fn, **kwargs):
        """
        Load CSV, thunderstorm compatible
        """

        data = np.genfromtxt(fn, delimiter=',',skip_header=0,names=True)
        dims = 3 if 'z_nm' in data.dtype.fields else 2
        
        imgshape = [
            int(np.ceil(np.max(data['y_nm']))), 
            int(np.ceil(np.max(data['x_nm'])))
        ]
        
        usedFields = ['x_nm', 'y_nm', 'z_nm', 'intensity_photon', 'offset_photon', 'chi2', 'uncertainty_nm', 'sigma_nm', 'frame']
        extraFields = []
        for f in data.dtype.fields:
            if not f in usedFields:
                extraFields.append((f, data.dtype.fields[f][0]))

        ds = cls(len(data), dims, imgshape, pixelsize=1, haveSigma='sigma_nm' in data.dtype.fields, extraFields=extraFields)
        ds.pos[:,0] = data['x_nm']
        ds.pos[:,1] = data['y_nm']
        if dims==3: ds.pos[:,2] = data['z_nm']
        
        ds.data.estim.sigma = data['sigma_nm'][:,None]
        ds.data.frame = data['frame'].astype(np.int32)
        ds.data.crlb.pos[:,:2] = data['uncertainty_nm'][:,None]
        
        ds.photons[:] = data['intensity_photon']
        ds.background[:] = data['offset_photon']
        ds.chisq[:] = data['chi2']
        
        for f in data.dtype.fields:
            if not f in usedFields:
                ds.data[f] = data[f]

        ds['locs_path'] = fn
                
        return ds
        
        
    @classmethod
    def load(cls, fn, **kwargs):
        ext = os.path.splitext(fn)[1]
        if ext == '.hdf5':
            return cls.loadHDF5(fn, **kwargs)
        elif ext == '.csv':
            return cls.loadCSVThunderStorm(fn, **kwargs)
        elif ext == '.npy':
            return cls.loadNumpy(fn, **kwargs)
        else:
            raise ValueError('unknown extension')
    
    @classmethod
    def fromEstimates(cls, estim, param_names, framenum, imgshape, crlb=None, chisq=None, 
                      roipos=None, addroipos=True, **kwargs):
        
        is3D = 'z' in param_names
        haveSigma = 'sx' in param_names
        if haveSigma:
            sx = param_names.index('sx')
            sy = param_names.index('sy')
        else:
            sx=sy=None
            
        dims = 3 if is3D else 2
        I_idx = param_names.index('I')
        bg_idx = param_names.index('bg')
        
        ds = cls(len(estim), dims, imgshape, haveSigma=haveSigma, **kwargs)
        ds.data.roi_id = np.arange(len(estim))
        
        if estim is not None:
            if addroipos and roipos is not None:
                estim = estim*1
                estim[:,[0,1]] += roipos[:,[1,0]]

            if np.can_cast(estim.dtype, ds.data.estim.dtype):
                ds.data.estim = estim
            else:
                # Assuming X,Y,[Z,]I,bg
                ds.data.estim.pos = estim[:,:dims]
                ds.data.estim.photons = estim[:,I_idx]
                ds.data.estim.bg = estim[:,bg_idx]
                
                if haveSigma:
                    ds.data.estim.sigma = estim[:,[sx,sy]]
                    ds.sigma = np.median(ds.data.estim.sigma,0)
            
        if crlb is not None:
            if np.can_cast(crlb.dtype, ds.data.estim.dtype):
                ds.data.crlb = crlb
            else:
                ds.data.crlb.pos = crlb[:,:dims]
                ds.data.crlb.photons = crlb[:,I_idx]
                ds.data.crlb.bg = crlb[:,bg_idx]

                if haveSigma:
                    ds.data.crlb.sigma = crlb[:,[sx,sy]]
            
        if chisq is not None:
            ds.data.chisq = chisq
        
        if framenum is not None:
            ds.data.frame = framenum
            
        if roipos is not None:
            ds.data.roipos = roipos[:, -2:]
            
        return ds
    
    def info(self):
        m_crlb_x = np.median(self.crlb.pos[:,0])
        m_bg= np.median(self.background)
        m_I=np.median(self.photons)
        return f"#Spots: {len(self)}. Imgsize:{self.imgshape[0]}x{self.imgshape[1]} pixels. Median CRLB X: {m_crlb_x:.2f} [pixels], bg:{m_bg:.1f}. I:{m_I:.1f}"
    
    def frc(self, zoom = 20, split_window=30, mask=None, **kwargs):
        """
        Compute FRC by splitting up based on frame numbers.
        This means drift correction needs to be done beforehand.
        """
       
        if mask is None and split_window is not None:
            mask = (self.frame // split_window) % 2 == 0
        
        return FRC(self.pos[:,:2], np.ones_like(self.photons), zoom=zoom, mask=mask,
                   imgshape=self.imgshape, pixelsize=self['pixelsize'], **kwargs)
    
    def selectByFrame(self, frames):
        if np.isscalar(frames):
            frames=[frames]
        
        ipf = self.indicesPerFrame()
        ix = np.concatenate([ipf[f] for f in frames])
        return self[ix]
    
    def joinByROI(self, other):
        nrois = max(self.roi_id.max(), other.roi_id.max())+1
        
        id_map = np.ones(nrois,dtype=np.int32)*-1
        id_map[self.roi_id] = np.arange(len(self))

        id_map2 = np.ones(nrois,dtype=np.int32)*-1
        id_map2[other.roi_id] = np.arange(len(other))
        
        which_locs = (id_map>=0) & (id_map2>=0)
        return id_map[which_locs], id_map2[which_locs]
    
    def render(self, zoom=1, sigma=1, clipmax=1, clip_percentile=None, intensities=False, transpose=False):
        imgshape = np.array(self.imgshape)*zoom
        
        from fastpsf import Context, GaussianPSFMethods
        with Context() as ctx:
            img = np.zeros(imgshape,dtype=np.float64)
            if sigma is None:
                sigma = np.mean(self.crlb.pos[:2])
            
            spots = np.zeros((len(self), 5), dtype=np.float32)
            spots[:, 0] = self.pos[:,0] * zoom
            spots[:, 1] = self.pos[:,1] * zoom
            spots[:, 2] = sigma
            spots[:, 3] = sigma
            spots[:, 4] = self.photons if intensities else 1

            img = GaussianPSFMethods(ctx).Draw(img, spots)

            if transpose:
                img = img.T
            
            if clip_percentile is not None:
                clipmax = np.percentile(img.flatten(), clip_percentile)
            else:
                clipmax *= img.max()
                        
            return np.minimum(img, clipmax)
        

    def renderFigure(self, scalebar_nm=1000, pixelsize=None, 
                     axes=None, scalebar_fontsize=14, title=None, cmap='inferno', 
                     scalebar_color='white', **kwargs):
    
        img = self.render(**kwargs)
            
        if axes is None:
            fig, axes = plt.subplots()
            
        axes.imshow(img, cmap=cmap)
    
        if scalebar_nm is not None:
            if pixelsize is None:
                pixelsize = self['pixelsize']
                
            scalebar_size = scalebar_nm / (self.imgshape[1] * pixelsize)
            
            axes.add_patch(Rectangle(xy=(0.05,0.05), width=scalebar_size, height=0.03, transform=axes.transAxes, 
                                     color=scalebar_color))
            axes.text(0.05, 0.08, f'{scalebar_nm} nm', transform=axes.transAxes, color= scalebar_color, 
                      va='bottom', fontsize=scalebar_fontsize)
        
        axes.axis('off')
        
        if title is not None:
            axes.set_title(title)
            

    def subsetInFrames(self, slice_or_start, end=None):
        if end is not None:
            s = slice(slice_or_start, end)
        else:
            s = slice_or_start
            
        ipf = self.indicesPerFrame()
        sel = np.concatenate([ix for ix in ipf[s]])
        return self[sel]

    def frameConnect(self, distance_px, ndims=2):
        """
        predicate(positions1, positions2): returns a list of booleans indicating if spots are close enough
        """
        ipf = self.indicesPerFrame()
        
        from smlmtorch.util.locs_util import find_pairs
        nframes = self.numFrames
        
        current = self.data[ipf[0]]
        results = []
        startframes = []
        
        for i in pb_range(nframes-1):
            
            next_ = self.data[ipf[i+1]]
            pairs = find_pairs(current.estim.pos[:,:2], 
                               next_.estim.pos[:,:2], distance_px)
            
            # for the frame i locs, if unpaired, add them to the new results list
            unpaired = np.zeros(len(current))==0
            unpaired[pairs[:,0]] = False
            
            to_add = current[unpaired]
            startframes.append(to_add.frame*1)
            to_add.frame = i
            results.append(to_add)
            merged = self._mergeLocalizations(current[pairs[:,0]], next_[pairs[:,1]])
            
            current = next_.copy()
            current[pairs[:,1]] = merged
            
            # current should contain the ipf[i+1] data now, with merged info
            #print(len(current))

        startframes.append(current.frame*1)
        current.frame = nframes-1
        results.append(current)

        total = sum([len(r) for r in results])
        #result_ds = type(self)(total, self.dims, self.imgshape, config=self.config)
        data = np.zeros(total, dtype=self.data.dtype).view(np.recarray)
 
        pos = 0
        for i,r in enumerate(results):
            data[pos:pos+len(r)] = r
            pos += len(r)
        startframes = np.concatenate(startframes)
            
        ds = type(self)(total, self.dims, self.imgshape, data=data, config=self.config)
        return ds,startframes
    
    def _mergeLocalizations(self, loc1, loc2):
        fi1 = loc1.crlb.pos**-2
        fi2 = loc2.crlb.pos**-2
        total_fi = fi1+fi2
        
        r = loc1.copy()

        r.estim.pos = (loc1.estim.pos*fi1 + loc2.estim.pos*fi2) / total_fi
        r.crlb.pos = total_fi ** -0.5
        r.estim.photons = loc1.estim.photons + loc2.estim.photons
        r.crlb.photons = (loc1.crlb.photons**2+loc2.crlb.photons**2) ** 0.5
        r.estim.bg = 0.5 * (loc1.estim.bg + loc2.estim.bg)
        return r

    def link(self, other_ds, distance_px, ndims=2):
        """
        predicate(positions1, positions2): returns a list of booleans indicating if spots are close enough
        """
        ipf = [self.indicesPerFrame(), other_ds.indicesPerFrame()]
        pos = [self.pos, other_ds.pos]
        nframes = min([len(i) for i in ipf])
        
        from smlmtorch.util.locs_util import find_pairs
        nframes = self.numFrames
        
        pair_list = []        
        for i in pb_range(nframes):
            pairs = find_pairs(*[pos[j][ipf[j][i]][:,:2] for j in range(2)])
            
            pairs[:,0] = ipf[0][i][pairs[:,0]]
            pairs[:,1] = ipf[1][i][pairs[:,1]]
            
            pair_list.append(pairs)
            
        pairs = np.concatenate(pair_list)
        
        print(f"#Pairs: {len(pairs)}.")
        return pairs
