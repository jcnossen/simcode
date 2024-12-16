# -*- coding: utf-8 -*-
"""
Dictionary that allows indexing like struct.member

Created on Thu May 26 21:03:08 2022

@author: jelmer
"""
from collections.abc import Mapping
import numpy as np
import torch
import yaml

class config_dict(Mapping):
    def __init__(self,  d=None, **kwargs):
        if d is not None:
            self.__dict__.update(d)
        self.__dict__.update(kwargs)
        for k,v in self.items():
            if type(v) == dict:
                self[k] = config_dict(v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        if not (key in self.__dict__):
            return
        return self.__dict__[key]
    
    def __setitem__(self, key, val):
        self.__dict__[key] = val
        
    def pop(self, key):
        self.__dict__.pop(key)

    def remove(self, key):
        self.__dict__.pop(key)

    def clone(self):
        r = {}
        for k,v in self.items():
            if type(v) == dict:
                r[k] = config_dict(v).clone()
            else:
                r[k] = v
        return config_dict(r)

    def __repr__(self):
        return str(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __iter__(self):
        return self.__dict__.__iter__()
        
    def to_dict(self):
        """
        Convert all numpy arrays and tensors to lists. 
        """
        r = {}
        for k,v in self.items():
            if type(v) == dict:
                v = config_dict(v)
                r[k] = v.to_dict()
            if type(v) == config_dict:
                r[k] = v.to_dict()
            elif type(v) == np.ndarray:
                r[k] = v.tolist()
            elif type(v) == torch.Tensor:
                r[k] = v.detach().tolist()
            elif type(v) == tuple:
                r[k] = list(v)
            else:
                r[k] = v
        return r
    
    @staticmethod
    def from_dict(v):
        s = config_dict(v)
        for k,v in s.items():
            if type(v) == dict: 
                s[k] = config_dict(v)
        return s

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            return config_dict.from_dict(yaml.safe_load(f))

if __name__ == '__main__':
    b = config_dict(x=1)

    print(b.x)
    print(b.test)
    
    print(config_dict.from_dict({ 'x':{'y':2} }))
    