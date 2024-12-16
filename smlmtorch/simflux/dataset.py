# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:00:53 2021

@author: jelmer
"""

from smlmtorch.smlm.dataset import Dataset
import numpy as np

class SFDataset (Dataset):
    """
    Dataset that holds an array of intensity values per spot
    """
    def __init__(self, length, dims, imgshape, data=None, numPatterns=None, config=None, **kwargs):

        if config is None:
            config = {}

        if data is not None:
            numPatterns = data.estim.ibg.shape[1]

        # This looks kinda convoluted but the goal is that Dataset.merge and __getitem__ are able to create SFDatasets
        if numPatterns is not None:
            config['numPatterns'] = numPatterns

            config['extraEstimFields'] = [
                ('ibg', np.float32, (numPatterns, 2))
            ]

        super().__init__(length, dims, imgshape, data=data, config=config, **kwargs)
    
    def createDTypes(self,dims, imgdims, includeGaussSigma, extraFields=None):
        return super().createDTypes(dims, imgdims, includeGaussSigma, 
                                    extraEstimFields=self.config['extraEstimFields'])

    @property
    def ibg(self):
        return self.data.estim.ibg
    
    @property
    def IBg(self):
        return self.ibg
    
    @property
    def ibg_crlb(self):
        return self.data.crlb.ibg
    
    @ibg_crlb.setter
    def ibg_crlb(self, v):
        self.data.crlb.ibg = v
    
    @ibg.setter
    def setter(self,v):
        self.data.ibg = v
        
    def hasNaN(self):
        return super().hasNaN() | np.isnan(self.ibg).any((1,2))
    
    