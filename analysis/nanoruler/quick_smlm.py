"""Quick per-frame 2D-Gaussian SMLM pass on the (reordered) nanoruler movie: only for setup facts (density, photons per
frame, background, PSF width), for the isolated-track extraction of the in-situ recipe and for the camera-map masks.
Detection: Gaussian-smoothed (sigma 1.5 px) frame minus a per-pixel running median (501 frames subsampled), local maxima
above thr x robust noise. Fit: Poisson MLE (Levenberg-Marquardt, torch) of x, y, I, bg, sigma on an R x R ROI; sigma free.
Not a method component and not a baseline of the comparison.
Output: Picasso-style hdf5 'locs' (frame = index in the reordered movie, x, y px with pixel centres at integers + 0... see
below, photons, bg, sx, lpx) ; coordinates: pixel (i, j) covers [j, j + 1) x [i, i + 1)?  NO: here x = column index of the
pixel centre (pixel centres at integer coordinates), as in the simcode pipeline and insitu scripts."""
import argparse, sys, os, numpy as np, torch, tifffile, h5py, torch.nn.functional as Fn
sys.path.insert(0, '/home/jc/simcode2')
from simcode.util.proctitle import set_title
set_title('nr-smlm')
ap = argparse.ArgumentParser()
ap.add_argument('--movie', default='/home/jc/simcode2/out/nanoruler/raw/sim4_1_reord.tif')
ap.add_argument('--gain', type=float, default=0.45); ap.add_argument('--offset', type=float, default=100.0)
ap.add_argument('--f0', type=int, default=0); ap.add_argument('--f1', type=int, default=0)
ap.add_argument('--thr', type=float, default=5.0); ap.add_argument('--roi', type=int, default=11)
ap.add_argument('--sigma0', type=float, default=1.83); ap.add_argument('--fix_sigma', type=int, default=0)
ap.add_argument('--sum', type=int, default=1, help='fit the moving sum of this many frames (window start = frame)')
ap.add_argument('--out', default='/home/jc/simcode2/out/nanoruler/quick_smlm.hdf5')
a = ap.parse_args()
dev = 'cuda'
t = tifffile.TiffFile(a.movie); N = len(t.pages) if not a.f1 else a.f1
R = a.roi; h = R // 2
g1 = torch.exp(-torch.arange(-4, 5, device=dev).float() ** 2 / (2 * 1.5 ** 2)); g1 /= g1.sum()
yy, xx = torch.meshgrid(torch.arange(R, device=dev).float() - h, torch.arange(R, device=dev).float() - h, indexing='ij')

def model(p):
    x, y, I, bg, s = [p[:, i, None, None] for i in range(5)]
    def ef(u, c):
        a_ = (u - c + 0.5) / (s * 2 ** 0.5); b_ = (u - c - 0.5) / (s * 2 ** 0.5)
        return 0.5 * (torch.erf(a_) - torch.erf(b_))
    return I * ef(xx[None], x) * ef(yy[None], y) + bg

def fit(roi):
    B = len(roi)
    bg0 = roi.flatten(1).quantile(0.2, 1).clamp(min=0.05); I0 = (roi - bg0[:, None, None]).sum((1, 2)).clamp(min=20)
    
    p = torch.stack([torch.zeros(B, device=dev), torch.zeros(B, device=dev), I0, bg0, torch.full((B,), a.sigma0, device=dev)], 1)
    lam = torch.full((B,), 1.0, device=dev); np_ = 4 if a.fix_sigma else 5
    def nll(p): mu = model(p).clamp(min=1e-3); return (mu - roi * torch.log(mu)).sum((1, 2))
    cur = nll(p)
    for it in range(30):
        pp = p.clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(lambda q: model(q).sum(0), pp, vectorize=True)  # [R,R,B,5]
        J = J.permute(2, 0, 1, 3).reshape(B, R * R, 5)[:, :, :np_]
        mu = model(p).clamp(min=1e-3).reshape(B, -1); d = roi.reshape(B, -1)
        gr = (J * ((d / mu) - 1)[..., None]).sum(1); H = torch.einsum('bpi,bpj,bp->bij', J, J, 1 / mu)
        Hd = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2)) + 1e-9 * torch.eye(np_, device=dev)
        st = torch.linalg.solve(Hd, gr[..., None])[..., 0]
        q = p.clone(); q[:, :np_] += st
        q[:, 0:2] = q[:, 0:2].clamp(-h + 1, h - 1); q[:, 2] = q[:, 2].clamp(min=1); q[:, 3] = q[:, 3].clamp(min=1e-2); q[:, 4] = q[:, 4].clamp(0.8, 4)
        nw = nll(q); ok = nw < cur
        p = torch.where(ok[:, None], q, p); cur = torch.where(ok, nw, cur); lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-4, 1e6)
    mu = model(p).clamp(min=1e-3).reshape(B, -1)
    pp = p.clone()
    J = torch.autograd.functional.jacobian(lambda q: model(q).sum(0), pp, vectorize=True).permute(2, 0, 1, 3).reshape(B, R * R, 5)[:, :, :np_]
    H = torch.einsum('bpi,bpj,bp->bij', J, J, 1 / mu)
    cr = torch.diagonal(torch.linalg.pinv(H), dim1=1, dim2=2).clamp(min=0).sqrt()
    chi = (((roi.reshape(B, -1) - mu) ** 2) / mu).sum(1) / (R * R)
    return p, cr, chi

rows = []
CH = 600
for c0 in range(a.f0, N, CH):
    c1 = min(c0 + CH + a.sum - 1, N)
    if c1 - c0 < a.sum: break
    mov = torch.tensor(np.stack([t.pages[i].asarray() for i in range(c0, c1)]).astype(np.float32), device=dev)
    ph = (mov - a.offset) * a.gain
    if a.sum > 1:
        cs = torch.cat([torch.zeros_like(ph[:1]), ph.cumsum(0)]); ph = cs[a.sum:] - cs[:-a.sum]
    med = ph[::6].median(0).values                    # static background incl. per-pixel offset deviations
    d = ph - med
    sm = Fn.conv2d(Fn.conv2d(d[:, None], g1[None, None, :, None], padding=(4, 0)), g1[None, None, None, :], padding=(0, 4))[:, 0]
    noise = 1.4826 * (sm - sm.median()).abs().median()
    mx = Fn.max_pool2d(sm[:, None], 7, 1, 3)[:, 0]
    pk = (sm == mx) & (sm > a.thr * noise)
    pk[:, :h + 1] = False; pk[:, -h - 1:] = False; pk[:, :, :h + 1] = False; pk[:, :, -h - 1:] = False
    f, iy, ix = pk.nonzero(as_tuple=True)
    if len(f) == 0: continue
    oy, ox = torch.meshgrid(torch.arange(-h, h + 1, device=dev), torch.arange(-h, h + 1, device=dev), indexing='ij')
    roi = ph[f[:, None, None], iy[:, None, None] + oy[None], ix[:, None, None] + ox[None]].clamp(min=0)   # includes static bg
    P_ = []; C_ = []; X_ = []
    for s in range(0, len(f), 4096):
        p, cr, chi = fit(roi[s:s + 4096]); P_.append(p); C_.append(cr); X_.append(chi)
    p = torch.cat(P_); cr = torch.cat(C_); chi = torch.cat(X_)
    rows.append(np.stack([(f + c0).cpu().numpy(), (ix + p[:, 0]).cpu().numpy(), (iy + p[:, 1]).cpu().numpy(), p[:, 2].cpu().numpy(), p[:, 3].cpu().numpy(),
                          p[:, 4].cpu().numpy(), cr[:, 0].cpu().numpy(), cr[:, 1].cpu().numpy(), chi.cpu().numpy(), sm[f, iy, ix].cpu().numpy() / float(noise)], 1))
    print(c0, len(f), 'noise %.3f' % float(noise), flush=True)
A = np.concatenate(rows)
dt = [('frame', 'u4'), ('x', 'f4'), ('y', 'f4'), ('photons', 'f4'), ('bg', 'f4'), ('sx', 'f4'), ('lpx', 'f4'), ('lpy', 'f4'), ('chi2', 'f4'), ('snr', 'f4')]
L = np.zeros(len(A), dt)
for i, (n, _) in enumerate(dt): L[n] = A[:, i]
L['sy'] if False else None
with h5py.File(a.out, 'w') as hf: hf.create_dataset('locs', data=L)
print('locs', len(L), 'per frame %.2f' % (len(L) / (N - a.f0)))
