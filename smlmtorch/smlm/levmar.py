import torch
from typing import List, Tuple, Union, Optional, Final
from torch import Tensor
from .crlb import poisson_crlb




#@torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [batchsize, numsamples]
    jac: [batchsize, numsamples, numparams]
    smp: [batchsize, numsamples]
    """
    #assert np.array_equal(smp.shape, mu.shape)
    sampledims = [i for i in range(1,len(smp.shape))]
                                   
    invmu = 1.0 / torch.clip(mu, min=1e-9)
    af = smp*invmu**2

    jacm = torch.matmul(jac[...,None], jac[...,None,:])  
    alpha = jacm * af[...,None,None]  
    alpha = alpha.sum(sampledims)

    beta = (jac * (smp*invmu-1)[...,None]).sum(sampledims)
    return alpha, beta

LM_SCALE_INVARIANT : Final[int] = 1
LM_SCALE_INVARIANT_NORM : Final[int]= 2
LM_SCALE_IDENTITY : Final[int] = 0

#@torch.jit.script
def lm_update(cur, alpha, beta, smp, lambda_:float, param_range_min_max:Tensor, scale_mode: int=2):
    """
    Separate some of the calculations to speed up with jit script
    """
    K = cur.shape[-1]

    if scale_mode == 2 or scale_mode == 1:
        # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A) 
        scale = (alpha * alpha).sum(1)
        if scale_mode == 2:
            scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent
        #assert torch.isnan(scale).sum()==0
        alpha += lambda_ * scale[:,:,None] * torch.eye(K,device=smp.device)[None]
    elif scale_mode == 0:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K,device=smp.device)[None]
    else:
        raise ValueError('invalid scale mode')
            
    steps = torch.linalg.solve(alpha, beta)

    assert torch.isnan(cur).sum()==0
    assert torch.isnan(steps).sum()==0

    cur = cur + steps
    cur = torch.maximum(cur, param_range_min_max[None,:,0])
    cur = torch.minimum(cur, param_range_min_max[None,:,1])
    return cur    

class LM_MLE(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations:int, lambda_:float, 
                 scale_mode: int=LM_SCALE_INVARIANT_NORM):
        """
        model: 
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()
        
        self.model = model
        self.iterations = int(iterations)
        self.scale_mode = scale_mode

        #if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)
    
    def forward(self, smp, initial, const_:Optional[Tensor]=None):
        cur = initial*1
        
        for i in range(self.iterations):
            mu,jac = self.model(cur, const_)
            alpha, beta = lm_alphabeta(mu, jac, smp)
            cur = lm_update(cur, alpha, beta, smp, self.lambda_, self.param_range_min_max, self.scale_mode)

        return cur
    
    
        


