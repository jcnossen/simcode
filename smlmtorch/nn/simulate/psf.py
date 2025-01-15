"""
Abstract PSF model. Gradient calculation is implemented using 
torch autograd vmap and jacfwd, so the models don't need to explicitly implement
the derivatives.

Author: Jelmer Cnossen
"""
import numpy as np
import torch
from smlmtorch.nn.simulate.gaussian_model import Gaussian2DModel
from torch.func import jacrev, jacfwd
import matplotlib.pyplot as plt
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional
from smlmtorch.smlm.gaussian_psf import gauss_psf_2D_fixed_sigma

class PSFBase(torch.nn.Module):
    def __init__(self, shape):

        super().__init__()
        self.shape = shape

    @abstractmethod
    def forward(self, params):
        pass

    def forward_non_batched(self, params):
        return self.forward(params[None])[0]

    def derivatives(self, params):
        def non_batched(p):
            return self.forward_non_batched(p).flatten()

        def calc_deriv(p): 
            ev = non_batched(p)
            return ev, jacfwd(non_batched)(p)

        ev, deriv = torch.vmap(calc_deriv)(params)
        return (ev.view(len(ev), *self.shape), 
                    deriv.view(len(params), *self.shape, -1))

    def fisher(self, params, read_noise:Optional[Tensor]=None):
        """
        Compute fisher matrix. read_noise is optional map of per-pixel readnoise
        """
        def non_batched(p):
            return self.forward(p[None])[0].flatten()
        
        """
        Read noise contribution is computed according to

        Huang, F., Hartwich, T., Rivera-Molina, F. et al. Video-rate nanoscopy using sCMOS camera–specific single-molecule localization algorithms. 
        Nat Methods 10, 653–658 (2013). https://doi.org/10.1038/nmeth.2488

        Basically, just add the readnoise to the expected value of the pixel in the denumerator
        """

        def calc_fisher(p): 
            ev = non_batched(p)
            if read_noise is not None:
                ev += read_noise
            jac = jacfwd(non_batched)(p)
        
            fisher = torch.matmul(jac[...,None], jac[...,None,:])
            fisher = fisher / ev[...,None,None]
            fisher = fisher.sum(0)
            return ev, fisher

        ev, fisher = torch.vmap(calc_fisher)(params)
        return ev.view(len(ev), *self.shape), fisher
    
    def crlb(self, params):
        ev, fisher = self.fisher(params)
        return ev, torch.sqrt( torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2) )


class Gaussian2DPSF(PSFBase):
    def __init__(self, sigma, shape):
        super().__init__(shape)

        self.sigma = sigma
        self.imgshape = shape

        if isinstance(sigma, (int, float)):
            self.sigma = [float(sigma), float(sigma)]

        self.model = Gaussian2DModel(shape, self.sigma[0], self.sigma[1])

    def forward(self, params):
        return self.model(params[:,[0,1,3,4]]) # ignore z

    """
    def fisher(self, params, read_noise):
        assert self.imgshape[0] == self.imgshape[1], "Only square images supported"
        ev, jac = gauss_psf_2D_fixed_sigma(params[:,[0,1,3,4]], self.imgshape[0], self.sigma[0], self.sigma[1])
        fisher = torch.matmul(jac[...,None], jac[...,None,:])
        fisher = fisher / ev[...,None,None]  
        fisher = fisher.flatten(1, len(ev.shape)-1).sum(1)
        return ev, fisher
    """

    def crlb(self, params, read_noise : Optional[Tensor] = None):
        # Some hacks to go from 3D to 2D
        ev, fisher = self.fisher(params, read_noise)
        fisher = fisher[:, [0,1,3,4]][:, :, [0,1,3,4]]
        crlb = params*0
        crlb[:, [0,1,3,4]] = torch.sqrt( torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2) )
        return ev, crlb



class CubicSplinePSF(PSFBase):
    def __init__(self, calib_file, img_shape):
        super().__init__(img_shape)

        assert img_shape[0] == img_shape[1], "Only square images supported"
        self.roisize = img_shape[0]
        
        from fastpsf import Context, CSplineCalibration, CSplineMethods

        self.ctx = Context()
        self.calib = CSplineCalibration.from_file(calib_file)
        self.psf_inst = CSplineMethods(self.ctx).CreatePSF_XYZIBg(self.roisize, self.calib)

    def forward(self, params):
        #return self.model(params[:,[0,1,3,4]]) # ignore z
        if len(params)>1024*16:
            result = []
            for i in range(0,len(params),1024*16):
                result.append(self.forward(params[i:i+1024*16]))
            result = torch.cat(result,0).to(params.device)
        else:
            ev = self.psf_inst.ExpectedValue(params.detach().cpu().numpy())
            return torch.tensor(ev).to(params.device)

    def crlb(self, params, read_noise : Optional[Tensor] = None):
        p = params.detach().cpu().numpy()
        ev = self.psf_inst.ExpectedValue(p)
        crlb = self.psf_inst.CRLB(p)
        return torch.from_numpy(ev), torch.from_numpy(crlb)


if __name__ == '__main__':
    psf = Gaussian2DPSF(1.5, (10,10))

    params = torch.tensor([[4.5, 4.5, 0, 200.0, 2.0]])

    ev, fisher = psf.fisher(params)
    print(f"fisher shape: {fisher.shape}")

    ev, crlb = psf.crlb(params)
    print(f"crlb shape: {crlb.shape}")
    print(f'crlb: {crlb}')

    ev, deriv = psf.derivatives(params)
    print(f"ev shape: {ev.shape}")
    print(f"derivatives shape: {deriv.shape}")

    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(ev[0].detach().numpy())
    ax[0,1].imshow(deriv[0,:,:,0].detach().numpy())
    ax[1,0].imshow(deriv[0,:,:,1].detach().numpy())
    ax[1,1].imshow(deriv[0,:,:,2].detach().numpy())
    plt.show()

