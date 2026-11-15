"""4Pi-SIMFLUX high-density showcase: simulated movie of a small 3D microtubule network (hollow, antibody-labelled tubes).

Everything except the sample is simmovie4pi.py (imported, unchanged): bead-fitted coherent 4Pi PSF NPC_f_dxy15, real
channel layout / registration / camera, SIM pattern of the MT data (m 0.9), 4Pi phase field with drift, DNA-PAINT
kinetics (ton 10 frames, per-event photons log-normal median 450 e/frame, sd log 0.5), measured background.
Sample (60 x 60 px = 7.68 um FOV): 6 gently curved filaments (quadratic Bezier in x-y), label sites on a cylinder of
radius `--r` (default 25 nm = 50 nm between the antibody label layers; 12.5 nm = bare tube) around the 3D axis, one site
per 2 nm of axis, random angle. Heights: crossings at dz 270 / 150 / 100 nm, one filament tilted -0.25 -> +0.25 um,
one at a small angle to it 100 nm higher (long x-y overlap, separated only in z), one descending. Axis |z| <= 0.25 um,
labels |z| <= 0.275 um: inside the range where the heads were trained (+-0.45 um) and the CRLB is flat (|z| < 0.3 um).
Rendering is chunked in time (GPU memory independent of the movie length).
Output: out/4pi/simmovie/<tag>/ like simmovie4pi.py (+ gt_axis.npz: dense axis samples x, y px, z um, tangent, filament).
Usage: CUDA_VISIBLE_DEVICES=1 python analysis/fourpi/showcase4pi_movie.py --tag sc_d1 --dens 1.0 --frames 6000
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.config import root; P4 = root('fourpi')
from simmovie4pi import reg_map, phi0_field, PX, GAIN, OFF, RN_ADU, BG, DELTA, REG
from simcode.fourpi.lut4pi import LUT4Pi

# (p0, control, p2) in um, (z at p0, z at p2) in um (z linear along the curve parameter)
FILS = [((0.0, 2.0), (3.8, 1.6), (7.68, 2.6), (-0.15, -0.15)),     # A: low, nearly horizontal
        ((2.2, 0.0), (2.6, 3.8), (2.0, 7.68), (0.12, 0.12)),       # B: high, crosses A (dz 270 nm)
        ((0.0, 6.6), (3.8, 4.4), (7.68, 0.6), (-0.25, 0.25)),      # C: tilted through the whole range
        ((5.4, 0.0), (5.0, 3.8), (5.8, 7.68), (0.0, 0.0)),         # D: mid, crosses A (dz 150 nm)
        ((0.0, 6.0), (3.8, 4.6), (7.68, 1.6), (0.10, 0.10)),       # E: small angle to C, ~100 nm above it at the overlap
        ((0.0, 5.6), (3.8, 6.3), (7.68, 4.4), (0.20, -0.10))]      # F: descending, crosses B, C, D, E


def bezier(p0, p1, p2, s):
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    s = s[:, None]
    P = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2
    dP = 2 * (1 - s) * (p1 - p0) + 2 * s * (p2 - p1)
    return P, dP


def axis_samples(step=0.002):
    """dense axis points of every filament: x, y, z (um), unit 3D tangent, filament index"""
    out = []
    for i, (p0, p1, p2, (z0, z1)) in enumerate(FILS):
        s = np.linspace(0, 1, 20001)
        P, dP = bezier(p0, p1, p2, s)
        dz = np.full(len(s), z1 - z0)
        seg = np.sqrt(((dP / 20000.0) ** 2).sum(1) + (dz / 20000.0) ** 2)
        L = np.concatenate([[0], np.cumsum(seg[:-1])])
        sq = np.interp(np.arange(0, L[-1], step), L, s)
        P, dP = bezier(p0, p1, p2, sq)
        z = z0 + sq * (z1 - z0)
        t = np.c_[dP, np.full(len(sq), z1 - z0)]; t /= np.linalg.norm(t, axis=1, keepdims=True)
        out.append(np.c_[P, z, t, np.full(len(sq), i)])
    return np.concatenate(out)                    # [n, 7]


def make_sites(rng, r=0.025, step=0.002, jit=0.0):
    A = axis_samples(step)
    t = A[:, 3:6]
    zh = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(t, zh); n1 /= np.linalg.norm(n1, axis=1, keepdims=True)   # in-plane normal
    n2 = np.cross(t, n1)                                                     # mostly z
    th = rng.uniform(0, 2 * np.pi, len(A))
    p = A[:, :3] + r * (np.cos(th)[:, None] * n1 + np.sin(th)[:, None] * n2)
    if jit > 0: p = p + jit * rng.standard_normal(p.shape)
    S = np.c_[p, A[:, 6]]
    S[:, :2] /= PX
    return S, A                                   # S [n, 4]: x px, y px, z um, filament


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='sc_d1'); p.add_argument('--dens', type=float, default=1.0)
    p.add_argument('--frames', type=int, default=6000); p.add_argument('--H', type=int, default=60)
    p.add_argument('--W', type=int, default=60); p.add_argument('--ton', type=float, default=10.0)
    p.add_argument('--r', type=float, default=0.025, help='label radius um (0.025 antibody, 0.0125 bare tube)')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--fchunk', type=int, default=1500)
    a = p.parse_args()
    from simcode.util.proctitle import set_title; set_title('4pi-scmovie')
    out = f'{P4}/simmovie/{a.tag}'; os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/meta.json'):
        print('exists', out); return
    rng = np.random.default_rng(a.seed)
    H, W, T = a.H, a.W, a.frames
    S, AX = make_sites(rng, a.r)
    area = H * W * PX * PX
    n_on = a.dens * area
    duty = n_on / len(S); rate = duty / a.ton
    nev = rng.poisson(rate * (T + 3 * a.ton), len(S))
    site = np.repeat(np.arange(len(S)), nev)
    t0 = rng.uniform(-3 * a.ton, T, len(site)); dur = rng.exponential(a.ton, len(site))
    F = 450.0 * np.exp(0.5 * rng.standard_normal(len(site)))
    keep = (t0 + dur > 0) & (t0 < T)
    site, t0, dur, F = site[keep], t0[keep], dur[keep], F[keep]
    # per-frame emitter list (vectorised)
    f0 = np.maximum(0, np.floor(t0)).astype(int); f1 = np.minimum(T - 1, np.floor(t0 + dur)).astype(int)
    nf = f1 - f0 + 1
    ev = np.repeat(np.arange(len(site)), nf)
    fr = f0[ev] + (np.arange(len(ev)) - np.repeat(np.cumsum(nf) - nf, nf))
    q = np.minimum(fr + 1, t0[ev] + dur[ev]) - np.maximum(fr, t0[ev])
    k_ = q > 0.02
    fr, ev, q = fr[k_], ev[k_], q[k_]
    o_ = np.argsort(fr, kind='stable'); fr, ev, q = fr[o_], ev[o_], q[o_]
    x, y, z = S[site[ev], 0], S[site[ev], 1], S[site[ev], 2]
    pat = json.load(open(f'{P4}/smlm/diag/pattern_MT.json'))
    step = fr % 6; ang = step // 3
    k = np.array([pat['angle0']['k'], pat['angle1']['k']]); psi = np.radians(np.array([pat['angle0']['psi_deg'], pat['angle1']['psi_deg']]))
    m = 0.9
    Pt = 1 + m * np.cos(np.einsum('nd,nd->n', k[ang], np.c_[x, y]) - psi[ang, step % 3])
    Nph = F[ev] * q * Pt
    Phi = phi0_field(x, y, fr, H, W, T) + np.where(ang == 1, DELTA, 0.0)
    print(f'sites {len(S)}, events {len(site)}, emitter-frames {len(fr)} ({len(fr) / T:.1f} per frame = {len(fr) / T / area:.2f} /um^2/frame)', flush=True)
    dev = 'cuda'
    st = json.load(open(f'{P4}/psf4pi/NPC_f_dxy15.json'))['psf']
    R = 21; h = R // 2
    lut = LUT4Pi(st, P=R, dev=dev)
    o = torch.arange(R, device=dev)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev) / H, torch.arange(W, device=dev) / W, indexing='ij')
    bgs = 1 + 0.2 * torch.sin(2 * math.pi * (xx * 0.8 + 0.3)) * torch.cos(2 * math.pi * (yy * 0.6))
    raw = np.lib.format.open_memmap(f'{out}/movie_tmp.npy', mode='w+', dtype=np.uint16, shape=(T, H, 4 * W))
    tb = time.time()
    bnd = np.searchsorted(fr, np.arange(0, T + a.fchunk, a.fchunk))
    for ci, c0 in enumerate(range(0, T, a.fchunk)):
        c1 = min(T, c0 + a.fchunk); Tc = c1 - c0
        e0, e1 = bnd[ci], bnd[ci + 1]
        img = torch.zeros(Tc, 4, H + 2 * h, W + 2 * h, device=dev)
        for c in range(4):
            xc, yc = reg_map(c, x[e0:e1], y[e0:e1], H, W)
            for i in range(0, e1 - e0, 8192):
                sl = slice(i, i + 8192); g_ = slice(e0 + i, min(e1, e0 + i + 8192))
                ix, iy = np.round(xc[sl]).astype(int), np.round(yc[sl]).astype(int)
                pos = torch.tensor(np.c_[xc[sl] - ix, yc[sl] - iy, z[g_]], device=dev, dtype=torch.float32)
                g = lut.evaluate_ch(pos, torch.tensor(Phi[g_], device=dev, dtype=torch.float32), c)
                g = g * torch.tensor(Nph[g_], device=dev, dtype=torch.float32)[:, None, None]
                fi = torch.tensor(fr[g_] - c0, device=dev)
                Y = (torch.tensor(iy, device=dev)[:, None, None] + o[None, :, None]).expand(-1, R, R)
                X = (torch.tensor(ix, device=dev)[:, None, None] + o[None, None, :]).expand(-1, R, R)
                ok = (Y >= 0) & (Y < H + 2 * h) & (X >= 0) & (X < W + 2 * h)
                img.index_put_((fi[:, None, None].expand_as(Y)[ok], torch.full_like(Y[ok], c), Y[ok], X[ok]), g[ok], accumulate=True)
        img = img[:, :, h:h + H, h:h + W]
        lam = img + torch.tensor(BG, device=dev)[None, :, None, None] * bgs[None, None]
        gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + ci)
        adu = torch.poisson(lam, generator=gen) / GAIN + OFF + RN_ADU * torch.randn(lam.shape, device=dev, generator=gen)
        adu = adu.round().clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
        raw[c0:c1] = np.concatenate([adu[:, 0], adu[:, 1], adu[:, 2, :, ::-1], adu[:, 3, :, ::-1]], 2)
        del img, lam
    raw.flush(); del raw
    os.replace(f'{out}/movie_tmp.npy', f'{out}/movie.npy')
    np.savez(f'{out}/gt_sites.npz', sites=S)
    np.savez(f'{out}/gt_axis.npz', axis=AX, fils=np.array([[*f[0], *f[1], *f[2], *f[3]] for f in FILS]), r=a.r)
    np.savez(f'{out}/gt_events.npz', site=site, t0=t0, dur=dur, F=F)
    np.savez(f'{out}/gt_frames.npz', frame=fr, event=ev, q=q, x=x, y=y, z=z, N=Nph, Phi=Phi)
    meta = dict(tag=a.tag, H=H, W=W, T=T, px_um=PX, gain=GAIN, offset=OFF, rn_adu=RN_ADU, bg=BG, delta_deg=-10.4, m=m,
                k=k.tolist(), psi_deg=np.degrees(psi).tolist(), reg=REG, ton=a.ton, dens_target=a.dens,
                dens_actual=len(fr) / T / area, n_sites=len(S), n_events=int(len(site)), psf='NPC_f_dxy15',
                sample='mt3d', label_radius_um=a.r, render_s=time.time() - tb)
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
    print(meta, flush=True)


if __name__ == '__main__':
    main()
