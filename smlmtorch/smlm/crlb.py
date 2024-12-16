import torch
from torch import Tensor
from typing import List, Tuple, Union, Optional, Final
import numpy as np
import matplotlib.pyplot as plt
import math

from abc import ABC, abstractmethod


@torch.jit.script
def poisson_crlb(mu:Tensor, jac:Tensor):
    """
    Compute crlb from expected value and per pixel derivatives, where pixel noise is assumed have Poisson distribution
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    fisher = torch.matmul(jac[...,None], jac[...,None,:])
    fisher = fisher / mu[...,None,None]  
    fisher = fisher.flatten(1, len(mu.shape)-1).sum(1)

    return torch.sqrt( torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2) )


def poisson_crlb_select_axes(mu:Tensor, jac:Tensor, *, skip_axes:List[int]=[]):
    
    if not isinstance(mu, torch.Tensor):
        mu = torch.Tensor(mu)

    if not isinstance(jac, torch.Tensor):
        jac = torch.Tensor(jac)

    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[...,axes]

    crlb = torch.zeros((len(mu),naxes), device=mu.device, dtype=mu.dtype)
    crlb[:,axes] = poisson_crlb(mu, jac)
    return crlb

