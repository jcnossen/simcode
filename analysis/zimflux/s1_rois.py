"""S1 step 1: camera calibration, spot detection and ROI extraction on the jan13 paper movie (classical ZIMFLUX, paper 3.6.1).
- Photons = (ADU - offset_map) * gain_map; maps from Calibration/dark_1 (offset = mean) and bright_2 (gain = mean / var,
  e-/ADU, as photonpy create_calib_obj). Read-noise variance (photons^2) per pixel saved for the likelihood (v = rn^2).
- Cycle c = frames 3c .. 3c+2 (slot = frame % 3, aligned to frame 0; pattern_jan13). Detection on the cycle sum: local
  maxima (11 x 11) of a Gaussian (sigma 1.5 px) smoothed image whose box statistic (7 x 7 sum minus 49 x the 25 x 25 mean)
  exceeds THR photons; 16 x 16 ROIs (centre between pixels 7 and 8 -> corner = peak - 8 + 1 ... see CORNER), 3 frames each.
usage: python s1_rois.py <gpu> [thr=200] -> out/zimflux/s1/rois.npz"""
import sys, os, numpy as np, tifffile, torch, torch.nn.functional as Fn
dev = torch.device(f'cuda:{sys.argv[1]}'); THR = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
D = '/mnt/data/dmdsf/simcode2_out/zimflux/surfdrive/Zimflux/'
OUT = '/home/jc/simcode2/out/zimflux/s1/'; os.makedirs(OUT, exist_ok=True)
P = 16
dark = tifffile.imread(D + 'Calibration/dark_1_MMStack_Default.ome.tif').astype(np.float64)
light = tifffile.imread(D + 'Calibration/bright_2_MMStack_Default.ome.tif').astype(np.float64)
off = dark.mean(0); sig = light - off; gain = sig.mean(0) / sig.var(0)
rnvar = dark.var(0) * gain ** 2
np.savez(OUT + 'camera.npz', offset=off, gain=gain, rnvar=rnvar)
offt = torch.tensor(off, device=dev, dtype=torch.float32); gt = torch.tensor(gain, device=dev, dtype=torch.float32)
files = [D + '3Dnanorulers/nanorulers_simflux_10ms_jan13_MMStack_Default.ome.tif',
         D + '3Dnanorulers/nanorulers_simflux_10ms_jan13_MMStack_Default_1.ome.tif']
mm = [tifffile.memmap(f) for f in files]; nf = [m.shape[0] for m in mm]; NF = sum(nf); NC = NF // 3
def frames(a, b):
    out = []
    for i, m in enumerate(mm):
        s0 = sum(nf[:i]); lo, hi = max(a, s0), min(b, s0 + nf[i])
        if lo < hi: out.append(np.asarray(m[lo - s0:hi - s0]).astype(np.float32))
    return np.concatenate(out)
xs = torch.arange(-6, 7, device=dev, dtype=torch.float32); gk = torch.exp(-xs ** 2 / (2 * 1.5 ** 2)); gk = gk / gk.sum()
rois, corners, cyc, stat = [], [], [], []
CH = 600   # cycles per chunk
for c0 in range(0, NC, CH):
    c1 = min(NC, c0 + CH)
    F = torch.tensor(frames(3 * c0, 3 * c1), device=dev)
    F = (F - offt) * gt
    n = c1 - c0; H, W = F.shape[1:]
    F = F.view(n, 3, H, W); S = F.sum(1, keepdim=True)
    sm = Fn.conv2d(Fn.conv2d(S, gk.view(1, 1, 1, -1), padding=(0, 6)), gk.view(1, 1, -1, 1), padding=(6, 0))
    box = Fn.avg_pool2d(S, 7, 1, 3, count_include_pad=False) * 49
    loc = Fn.avg_pool2d(S, 25, 1, 12, count_include_pad=False) * 49
    Dst = box - loc
    pk = (sm == Fn.max_pool2d(sm, 11, 1, 5)) & (Dst > THR)
    pk[..., :8, :] = False; pk[..., H - 8:, :] = False; pk[..., :8] = False; pk[..., W - 8:] = False
    t, _, y, x = torch.nonzero(pk, as_tuple=True)
    cy, cx = y - 8, x - 8   # ROI rows cy .. cy+15: the peak pixel is index 8, i.e. +0.5 px from the ROI centre (7.5)
    iy = cy[:, None] + torch.arange(P, device=dev); ix = cx[:, None] + torch.arange(P, device=dev)
    R = F[t[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], iy[:, None, :, None], ix[:, None, None, :]]
    rois.append(R.half().cpu()); corners.append(torch.stack([cy, cx], 1).cpu()); cyc.append((t + c0).cpu()); stat.append(Dst[t, 0, y, x].cpu())
    print(c0, len(t), flush=True)
rois = torch.cat(rois).numpy(); corners = torch.cat(corners).numpy(); cyc = torch.cat(cyc).numpy(); stat = torch.cat(stat).numpy()
np.savez(OUT + 'rois.npz', rois=rois, corner=corners, cycle=cyc, boxstat=stat, thr=THR, ncycles=NC)
print('spots', len(cyc), 'per cycle', len(cyc) / NC, 'gain median', np.median(gain), 'rnvar median', np.median(rnvar))
