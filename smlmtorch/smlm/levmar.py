import torch
from typing import List, Tuple, Union, Optional, Final
from torch import Tensor
from .crlb import poisson_crlb


@torch.jit.script
def poisson_nll(mu: Tensor, smp: Tensor) -> Tensor:
    """Poisson negative log-likelihood, summed over all sample dims.
    Constant (log smp!) terms are dropped. Returns [batchsize]."""
    mu_c = torch.clip(mu, min=1e-9)
    nll = mu_c - smp * torch.log(mu_c)
    return nll.flatten(1).sum(1)




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


class LM_MLE_Adaptive(torch.nn.Module):
    """Levenberg-Marquardt MLE with per-sample step acceptance and adaptive damping.

    Differences from LM_MLE:
      1. Every candidate step is evaluated. If the Poisson NLL decreases, the
         step is accepted per-sample and lambda shrinks; otherwise the step is
         rejected and lambda grows.
      2. Damping is classical Marquardt (lambda * diag(J^T W J)), not the
         (alpha * alpha).sum(1) heuristic.
      3. NaN / non-finite trial losses are masked, not asserted.

    Parameters
    ----------
    model : callable(params, const_) -> (mu, jac)
    param_range_min_max : Tensor [K, 2]
    iterations : int
    lambda_init : float
        Initial damping.  In this Marquardt-style formulation the effective
        damping is lambda * diag(alpha), so 1.0 is roughly "half gradient
        descent, half Gauss-Newton".
    lambda_up, lambda_down : float
        Multiplicative update to lambda on a rejected / accepted step.
    lambda_min, lambda_max : float
        Damping clamps.
    """

    def __init__(self, model, param_range_min_max, iterations: int,
                 lambda_init: float = 1.0,
                 lambda_up: float = 3.0,
                 lambda_down: float = 0.4,
                 lambda_min: float = 1e-4,
                 lambda_max: float = 1e10):
        super().__init__()
        self.model = model
        self.iterations = int(iterations)
        self.lambda_init = float(lambda_init)
        self.lambda_up = float(lambda_up)
        self.lambda_down = float(lambda_down)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.param_range_min_max = param_range_min_max

    def forward(self, smp, initial, const_: Optional[Tensor] = None):
        cur = initial.clone()
        B, K = cur.shape
        dev = cur.device
        dt = cur.dtype

        lam = torch.full((B,), self.lambda_init, device=dev, dtype=dt)
        eye = torch.eye(K, device=dev, dtype=dt)
        pmin = self.param_range_min_max[None, :, 0]
        pmax = self.param_range_min_max[None, :, 1]

        mu, jac = self.model(cur, const_)
        loss = poisson_nll(mu, smp)

        for _ in range(self.iterations):
            alpha, beta = lm_alphabeta(mu, jac, smp)

            # Classical Marquardt: lambda times the diagonal of alpha
            # (which is a positive weighted J^T J), applied per-sample.
            diag_a = torch.diagonal(alpha, dim1=1, dim2=2)
            alpha_d = alpha + (lam[:, None] * diag_a)[..., None] * eye[None]

            # Solve; guard against singular / NaN steps.
            step = torch.linalg.solve(alpha_d, beta)
            step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)

            trial = torch.clamp(cur + step, pmin, pmax)
            mu_new, jac_new = self.model(trial, const_)
            loss_new = poisson_nll(mu_new, smp)

            accept = torch.isfinite(loss_new) & (loss_new < loss)

            # Broadcast the [B] accept mask to each tensor's shape.
            m_p = accept[:, None]
            m_s = accept.view(-1, *([1] * (mu.ndim - 1)))
            m_j = accept.view(-1, *([1] * (jac.ndim - 1)))

            cur = torch.where(m_p, trial, cur)
            loss = torch.where(accept, loss_new, loss)
            mu = torch.where(m_s, mu_new, mu)
            jac = torch.where(m_j, jac_new, jac)

            lam = torch.where(accept, lam * self.lambda_down, lam * self.lambda_up)
            lam = lam.clamp(self.lambda_min, self.lambda_max)

        return cur


