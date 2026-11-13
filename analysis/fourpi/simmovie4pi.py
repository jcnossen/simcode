"""Full simulated 4Pi-SIMFLUX movie (not candidate patches), in the real data conventions, with ground truth.

Frame: 4 interference channels of H x W px (default 50 x 50, 128 nm/px), stored like the raw camera frame: the four
crops side by side [T, H, 4 W] uint16 ADU, channels 2 and 3 mirrored left-right (as on the sensor; the reader
un-mirrors them exactly like dc.crops). Camera: offset 100 ADU, gain 0.23 e/ADU, read noise 4.9 ADU.
Optics: the bead-fitted coherent 4Pi PSF (out/4pi/psf4pi/NPC_f_dxy15.json, 21 px LUT, channel phases / transmissions
included), per-channel registration ch0 -> ch c = affine (shift + small linear term, magnitudes like the NPC in-situ
map), per-angle 4Pi offset delta = -10.4 deg (MT).
Pattern: step = frame % 6 (frames 0-2 angle 0 along x, 3-5 angle 1 along y), k and step phases of the MT pattern
estimate (out/4pi/smlm/diag/pattern_MT.json), depth m = 0.9.
4Pi phase field: Phi0(x, y, t) = smooth quadratic map (~150 deg peak-to-peak over the FOV) + time drift (linear 80 deg
over the movie + a 40 deg sinusoid with a 1500-frame period), so the in-situ field estimate is exercised.
Sample: 4 microtubule-like hollow filaments (label radius 20 nm) crossing at different heights (one tilted through
-0.25..+0.25 um) and 6 Nup96-like pores (8 corners x 2 copies, radius 53.7 nm, two rings 50 nm apart in z).
Kinetics: DNA-PAINT, each site has binding events (Poisson in time), exponential on-time (mean `ton` frames), per-event
photon rate log-normal (median 450 e/frame summed over the channels, sd log 0.5), duty cycle set so that the mean
density of ON emitters is `dens` per um^2 per frame. Background per channel 0.42 / 1.06 / 0.54 / 1.17 e/px/frame
(MT, measured) with a smooth +-20 % structure.
Output: out/4pi/simmovie/<tag>/movie.npy, gt_sites.npz, gt_events.npz, gt_frames.npz (per frame and emitter:
x, y px ch0, z um, photons), meta.json
Usage: CUDA_VISIBLE_DEVICES=2 python analysis/fourpi/simmovie4pi.py --tag d1 --dens 1.0 --frames 4000
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
from simcode.fourpi.lut4pi import LUT4Pi

PX = 0.128
GAIN, OFF, RN_ADU = 0.23, 100.0, 4.9
BG = (0.42, 1.06, 0.54, 1.17)
DELTA = math.radians(-10.4)
# ch0 -> ch c: x_c = A_c @ (x, y) + b_c about the FOV centre (px); like the NPC in-situ map (shifts 0.06-0.38 px)
REG = {1: ([[1.0, 0.002], [-0.002, 1.0]], [-0.058, 0.053]),
       2: ([[1.001, -0.003], [0.003, 0.999]], [0.307, -0.005]),
       3: ([[0.999, 0.002], [-0.001, 1.001]], [0.378, 0.091])}


def reg_map(c, x, y, H, W):
    if c == 0: return x, y
    A, b = np.array(REG[c][0]), np.array(REG[c][1])
    cx, cy = (W - 1) / 2, (H - 1) / 2
    u, v = x - cx, y - cy
    return A[0, 0] * u + A[0, 1] * v + cx + b[0], A[1, 0] * u + A[1, 1] * v + cy + b[1]


def phi0_field(x, y, t, H, W, T):
    u, v = (x - W / 2) / (W / 2), (y - H / 2) / (H / 2)
    m = np.radians(60.0) * u + np.radians(-40.0) * v + np.radians(50.0) * (u * u - 0.5 * v * v) + np.radians(25.0) * u * v
    d = np.radians(80.0) * t / T + np.radians(40.0) * np.sin(2 * np.pi * t / 1500.0)
    return m + d


def make_sites(H, W, rng):
    S = []
    L = W * PX  # um
    # microtubules: lines across the FOV, hollow: label at radius 20 nm around the axis
    fil = [((0.3, 1.0), (6.1, 5.6), -0.15, -0.15), ((0.5, 5.9), (5.9, 0.6), 0.10, 0.10),
           ((3.2, 0.2), (2.6, 6.2), -0.25, 0.25), ((0.2, 3.6), (6.2, 2.9), 0.0, 0.0)]
    for (x0, y0), (x1, y1), z0, z1 in fil:
        ln = math.hypot(x1 - x0, y1 - y0)
        n = int(ln / 0.004)                      # one site per 4 nm of filament
        s = rng.uniform(0, 1, n)
        ex, ey = (x1 - x0) / ln, (y1 - y0) / ln
        th = rng.uniform(0, 2 * np.pi, n)
        rr = 0.020
        # cross-section in the plane spanned by the in-plane normal (-ey, ex) and z
        px_ = x0 + s * (x1 - x0) + rr * np.cos(th) * (-ey)
        py_ = y0 + s * (y1 - y0) + rr * np.cos(th) * ex
        pz_ = z0 + s * (z1 - z0) + rr * np.sin(th)
        S.append(np.stack([px_, py_, pz_, np.zeros(n)], 1))
    # NPCs: 8 corners x 2 copies (12 nm apart) per ring, two rings 50 nm apart in z
    pores = [(1.2, 2.0, -0.05), (4.8, 1.3, 0.10), (2.3, 4.9, 0.20), (5.2, 4.3, -0.10), (1.0, 5.4, 0.05), (4.0, 3.3, 0.15)]
    for cx, cy, cz in pores:
        rot = rng.uniform(0, np.pi / 4)
        for ring, dz in ((0, -0.025), (1, 0.025)):
            for k in range(8):
                for cp in (-1, 1):
                    a = rot + 2 * np.pi * k / 8 + cp * 0.006 / 0.0537 + ring * 0.12
                    S.append(np.array([[cx + 0.0537 * np.cos(a), cy + 0.0537 * np.sin(a), cz + dz, 1.0]]))
    S = np.concatenate(S)
    S[:, :2] /= PX                               # -> px
    return S                                     # [n, 4]: x px, y px, z um, kind (0 MT, 1 NPC)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='d1'); p.add_argument('--dens', type=float, default=1.0)
    p.add_argument('--frames', type=int, default=4000); p.add_argument('--H', type=int, default=50)
    p.add_argument('--W', type=int, default=50); p.add_argument('--ton', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ksum', type=int, default=1, help='K-summed conditions (real dense data by summing K windows): K x background, sqrt(K) read noise; the offset stays one OFF (= the sum with (K - 1) offsets removed)')
    a = p.parse_args()
    from simcode.util.proctitle import auto; auto('4pi-sim')
    out = f'{P4}/simmovie/{a.tag}'; os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    H, W, T = a.H, a.W, a.frames
    S = make_sites(H, W, rng)
    area = H * W * PX * PX
    n_on = a.dens * area                                   # mean ON emitters per frame
    duty = n_on / len(S)
    rate = duty / a.ton                                    # events per site per frame
    nev = rng.poisson(rate * (T + 3 * a.ton), len(S))
    site = np.repeat(np.arange(len(S)), nev)
    t0 = rng.uniform(-3 * a.ton, T, len(site)); dur = rng.exponential(a.ton, len(site))
    F = 450.0 * np.exp(0.5 * rng.standard_normal(len(site)))
    keep = (t0 + dur > 0) & (t0 < T)
    site, t0, dur, F = site[keep], t0[keep], dur[keep], F[keep]
    # per-frame emitter list
    fr, ev, q = [], [], []
    for e in range(len(site)):
        f0, f1 = int(max(0, math.floor(t0[e]))), int(min(T - 1, math.floor(t0[e] + dur[e])))
        for f in range(f0, f1 + 1):
            qq = min(f + 1, t0[e] + dur[e]) - max(f, t0[e])
            if qq > 0.02: fr.append(f); ev.append(e); q.append(qq)
    fr, ev, q = np.array(fr), np.array(ev), np.array(q)
    x, y, z = S[site[ev], 0], S[site[ev], 1], S[site[ev], 2]
    pat = json.load(open(f'{P4}/smlm/diag/pattern_MT.json'))
    step = fr % 6; ang = step // 3
    k = np.array([pat['angle0']['k'], pat['angle1']['k']]); psi = np.radians(np.array([pat['angle0']['psi_deg'], pat['angle1']['psi_deg']]))
    m = 0.9
    P = 1 + m * np.cos(np.einsum('nd,nd->n', k[ang], np.c_[x, y]) - psi[ang, step % 3])
    Nph = F[ev] * q * P                                     # photons of this emitter in this frame (4 channels)
    Phi = phi0_field(x, y, fr, H, W, T) + np.where(ang == 1, DELTA, 0.0)
    print(f'sites {len(S)}, events {len(site)}, emitter-frames {len(fr)} ({len(fr) / T:.1f} per frame = {len(fr) / T / area:.2f} /um^2/frame)', flush=True)
    # render
    dev = 'cuda'
    st = json.load(open(f'{P4}/psf4pi/NPC_f_dxy15.json'))['psf']
    R = 21; h = R // 2
    lut = LUT4Pi(st, P=R, dev=dev)
    img = torch.zeros(T, 4, H + 2 * h, W + 2 * h, device=dev)
    o = torch.arange(R, device=dev)
    tb = time.time()
    for c in range(4):
        xc, yc = reg_map(c, x, y, H, W)
        for i in range(0, len(fr), 8192):
            sl = slice(i, i + 8192)
            ix, iy = np.round(xc[sl]).astype(int), np.round(yc[sl]).astype(int)
            pos = torch.tensor(np.c_[xc[sl] - ix, yc[sl] - iy, z[sl]], device=dev, dtype=torch.float32)
            g = lut.evaluate_ch(pos, torch.tensor(Phi[sl], device=dev, dtype=torch.float32), c)       # [n,R,R]
            g = g * torch.tensor(Nph[sl], device=dev, dtype=torch.float32)[:, None, None]
            fi = torch.tensor(fr[sl], device=dev)
            Y = (torch.tensor(iy, device=dev)[:, None, None] + o[None, :, None]).expand(-1, R, R)
            X = (torch.tensor(ix, device=dev)[:, None, None] + o[None, None, :]).expand(-1, R, R)
            ok = (Y >= 0) & (Y < H + 2 * h) & (X >= 0) & (X < W + 2 * h)
            # index into the padded frame (emitter at px ix -> padded ix + h; patch corner at ix + h - h)
            img.index_put_((fi[:, None, None].expand_as(Y)[ok], torch.full_like(Y[ok], c), Y[ok], X[ok]), g[ok], accumulate=True)
    img = img[:, :, h:h + H, h:h + W]
    yy, xx = torch.meshgrid(torch.arange(H, device=dev) / H, torch.arange(W, device=dev) / W, indexing='ij')
    bgs = 1 + 0.2 * torch.sin(2 * math.pi * (xx * 0.8 + 0.3)) * torch.cos(2 * math.pi * (yy * 0.6))
    lam = img + a.ksum * torch.tensor(BG, device=dev)[None, :, None, None] * bgs[None, None]
    g = torch.Generator(device=dev).manual_seed(a.seed)
    adu = torch.poisson(lam, generator=g) / GAIN + OFF + RN_ADU * math.sqrt(a.ksum) * torch.randn(lam.shape, device=dev, generator=g)
    adu = adu.round().clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)       # [T,4,H,W]
    raw = np.concatenate([adu[:, 0], adu[:, 1], adu[:, 2, :, ::-1], adu[:, 3, :, ::-1]], 2)   # sensor layout, ch2/3 mirrored
    np.save(f'{out}/movie.npy', raw)
    np.savez(f'{out}/gt_sites.npz', sites=S)
    np.savez(f'{out}/gt_events.npz', site=site, t0=t0, dur=dur, F=F)
    np.savez(f'{out}/gt_frames.npz', frame=fr, event=ev, q=q, x=x, y=y, z=z, N=Nph, Phi=Phi)
    meta = dict(tag=a.tag, H=H, W=W, T=T, px_um=PX, gain=GAIN, offset=OFF, rn_adu=RN_ADU * math.sqrt(a.ksum), bg=[b * a.ksum for b in BG], ksum=a.ksum, delta_deg=-10.4, m=m,
                k=k.tolist(), psi_deg=np.degrees(psi).tolist(), reg=REG, ton=a.ton, dens_target=a.dens,
                dens_actual=len(fr) / T / area, n_sites=len(S), n_events=int(len(site)), psf='NPC_f_dxy15',
                render_s=time.time() - tb)
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
    print(meta, flush=True)


if __name__ == '__main__':
    main()
