"""

3D rotation + scaling

[ x, y, z, 1, 0, 0, 0, 0, 0, 0 ,0, 0]       [x']
[ 0, 0, 0, 0, x, y, z, 0, 0, 0, 0, 0] * T = [y']
[ 0, 0, 0, 0, 0, 0, 0, 1, x, y, z, 1]       [z'] 

"""
import numpy as np
import matplotlib.pyplot as plt


def estimate_affine_transform(pts, pts_t):
    dims = pts.shape[1]
    A = np.zeros((len(pts)*dims, (dims+1)*dims))
    for i in range(dims):
        A[i::dims,(dims+1)*i:(dims+1)*(i+1)-1] = pts
        A[i::dims,(dims+1)*i+dims] = 1
    b = pts_t.flatten()
    T, residual, rank, s = np.linalg.lstsq(A,b,rcond=-1)
    return T.reshape((dims,dims+1))
    

def find_pairs(pos1, pos2, max_dist):
    d = np.sqrt(((pts[None]-pts[:,None])**2).sum(-1))
    ix1,ix2 = np.where(d<max_dist)
    return ix1, ix2    
    

if __name__ == '__main__':
    
    D = 2
    N = 1000
    pts = np.random.uniform(size=(N,D))*20
    
    A = np.eye(D+1)
    A[:2,:2] += np.random.normal(0, 0.01, size=(2,2))
    A[:2,-1] = 0.4
    
    pts_exp = np.concatenate([pts, np.ones(N)[:,None]],-1)
    #pts_t = A @ pts_exp @ A
    #pts_t = (A[None] * pts_exp[:,None]).sum(2)
    measured = ((A @ pts_exp.T).T)[:,:D]
    measured += np.random.normal(size=measured.shape, scale=0.1)

    A_est = np.eye(D+1)

    for i in range(5):
        # apply current mapping
        src_exp = np.concatenate([pts, np.ones(N)[:,None]],-1)
        src_m = ((A_est @ pts_exp.T).T)[:,:D]
       
        pairs = find_pairs(src_m, measured, 0.1)
        
        A_est = estimate_affine_transform(src_m[pairs[0]], measured[pairs[1]])

        pts_exp = np.concatenate([pts, np.ones(N)[:,None]],-1)
        pts_m = ((A_est @ pts_exp.T).T)[:,:D]
        
        err = ((src_m[pairs[0]]-measured[pairs[1]])**2).mean()
        print(f"err={err:.2f}. #pairs={len(pairs[0])}")
        
        plt.figure()
        plt.scatter(pts[:,0], pts[:,1], s=3,label='Original')
        #plt.scatter(pts_t[:,0], pts_t[:,1], s=3,label='Transformed')
        plt.scatter(src_m[pairs[0],0], src_m[pairs[0],1], marker='o', label='Mapped')
        plt.scatter(measured[pairs[1],0], measured[pairs[1],1], s=2, c='k', marker='o', label='Transformed')
        plt.legend()

    
        
    
#%%    
    A_est = estimate_affine_transform(pts, pts_t)
    
    print(A)
    print(A_est)

        
    A_est_exp = np.zeros((D+1,D+1))
    A_est_exp[:D,:D+1] = A_est
    A_est_exp[-1,-1] = 1
    A_inv = np.linalg.pinv(A_est_exp)
    
    print(A_inv)
        
    #%%
    pts_m = ((A_est @ pts_exp.T).T)[:,:D]

    plt.figure()
    plt.scatter(pts[:,0], pts[:,1], s=3,label='Original')
    plt.scatter(pts_t[:,0], pts_t[:,1], s=3,label='Transformed')
    plt.scatter(pts_m[:,0], pts_m[:,1], s=3,label='Mapped')
    plt.legend()
    
    err_org = ((pts_t-pts)**2).sum()
    err_mapped = ((pts_t-pts_m)**2).sum()

    print(err_org)
    print(err_mapped)
    