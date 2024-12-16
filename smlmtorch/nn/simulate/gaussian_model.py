#%%
import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from torch.func import jacrev, jacfwd
from math import sqrt
from torch import vmap
from functools import partial
from torch import erf
import time
import tqdm
import torch.jit
from typing import List, Tuple, Union, Optional, Final



def gaussian2D(params, width: int, height: int, sigma_x: float, sigma_y: float):
    device = params.device
    batch_size = params.shape[0]

    # Unpack parameters
    x0 = params[:, 0]
    y0 = params[:, 1]
    intensity = params[:, 2]
    background = params[:, 3]

    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)

    # Use [None] instead of meshgrid
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xx = xx[None]
    yy = yy[None]
    
    # Calculate error function
    erf_x1 = erf((xx - x0[:,None,None] + 0.5) / (sqrt(2) * sigma_x))
    erf_x2 = erf((xx - x0[:,None,None] - 0.5) / (sqrt(2) * sigma_x))
    erf_y1 = erf((yy - y0[:,None,None] + 0.5) / (sqrt(2) * sigma_y))
    erf_y2 = erf((yy - y0[:,None,None] - 0.5) / (sqrt(2) * sigma_y))

    # Integrate Gaussian over pixel
    gaussian = 0.25 * intensity[:,None,None] * (erf_x1 - erf_x2) * (erf_y1 - erf_y2) + background[:,None,None]

    return gaussian


# Class to hold the roisize and sigma values
class Gaussian2DModel(torch.nn.Module):
    def __init__(self, shape, sigma_x, sigma_y):
        super().__init__()
        
        self.h, self.w = shape
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def forward(self, params):
        return gaussian2D(params, self.w, self.h, self.sigma_x, self.sigma_y)

@torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [numsamples]
    jac: [numsamples, numparams]
    smp: [numsamples]
    """
    #assert np.array_equal(smp.shape, mu.shape)
    sampledims = [i for i in range(len(smp.shape))]
                                   
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

@torch.jit.script
def lm_update(cur, alpha, beta, smp, lambda_:float, param_range_min_max:Tensor, scale_mode: int=2):
    """
    Separate some of the calculations to speed up with jit script
    """
    K = cur.shape[-1]

    if scale_mode == 2 or scale_mode == 1:
        # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A) 
        scale = (alpha * alpha).sum(0)
        if scale_mode == 2:
            scale /= scale.mean(0, keepdim=True) # normalize so lambda scale is not model dependent
        #assert torch.isnan(scale).sum()==0
        alpha += lambda_ * scale * torch.eye(K,device=smp.device)
    elif scale_mode == 0:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K,device=smp.device)
    else:
        raise ValueError('invalid scale mode')
            
    steps = torch.linalg.solve(alpha, beta)

    #assert torch.isnan(cur).sum()==0
    #assert torch.isnan(steps).sum()==0

    cur = cur + steps
    cur = torch.maximum(cur, param_range_min_max[:,0])
    cur = torch.minimum(cur, param_range_min_max[:,1])
    return cur    

class LM_MLE(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations:int, lambda_:float, 
                 scale_mode: int=LM_SCALE_INVARIANT_NORM):
        super().__init__()
        
        self.model = model
        self.iterations = int(iterations)
        self.scale_mode = scale_mode

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)
    
    def forward(self, smp, initial):
        cur = initial*1
        
        for i in range(self.iterations):
            mu = self.model.forward(cur)
            jac = jacfwd(self.model.forward)(cur)

            alpha, beta = lm_alphabeta(mu, jac, smp)
            cur = lm_update(cur, alpha, beta, smp, self.lambda_, self.param_range_min_max, self.scale_mode)

        return cur

if __name__ == '__main__':

    K = 5
    roisize = 10
    sigma_x = 1.5
    sigma_y = 1.5
    M = 1000

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Gaussian2DModel object with the given roisize and sigma values
    gaussian2D_model = Gaussian2DModel([roisize, roisize], sigma_x, sigma_y).to(dev)

    # Create a tensor for the parameters with requires_grad=True and shape (M, K)
    # create points in middle of ROI
    params = torch.zeros(M, K, requires_grad=True).to(dev)
    params[:,0] = roisize/2
    params[:,1] = roisize/2
    # Z coord is ignored in this model
    params[:,3] = 100.0
    params[:,4] = 1.0

    #params[:,0] += torch.linspace(-2,2,M).to(dev)
    #ev = gaussian2D_model(params)
    #from pytweezer import image_view, array_plot
    #array_plot(ev[:,roisize//2,:])

    # add noise to x and y
    params[:,0] += torch.randn(M, device=dev)*0.1
    params[:,1] += torch.randn(M, device=dev)*0.1

    def compute_ev_and_jacobian(model, params):
        ev = model.forward(params)
        return ev, jacfwd(model.forward)(params)

    param_range_min_max = torch.tensor([
        [1,roisize-1],[1,roisize-1],
        [0,1e8],[0,100]], device=dev)

    mu = vmap(lambda p: gaussian2D_model.forward(p))(params)
    smp = torch.poisson(mu)
    #from pytweezer import image_view
    #image_view(smp)

    lm = LM_MLE(gaussian2D_model, param_range_min_max, iterations=20, lambda_=100, scale_mode=LM_SCALE_INVARIANT_NORM)

    def calc_alpha_beta(params, smp):
        mu,jac = compute_ev_and_jacobian(gaussian2D_model, params)
        alpha, beta = lm_alphabeta(mu, jac, smp)
        return alpha, beta


    #alpha, beta = vmap(calc_alpha_beta)(params, smp)

    estim = lm.forward(smp[0], params[0])

    #print("Reshaped Jacobian matrix:", jacobian_matrix)

    runs = 100
    # measure speed of computation
    t0 = time.time()
    for i in tqdm.trange(runs):
        #estim = vmap(lm.forward)(smp, params)
        alpha, beta = vmap(calc_alpha_beta)(params, smp)
    t1 = time.time()
    print("Imgs/sec:", runs * M / (t1 - t0))


