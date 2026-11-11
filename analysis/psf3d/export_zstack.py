"""(2026-11-11 in-situ 3D PSF) Export the calibrated PSF of a calib json as plain z-stacks for other pipelines (e.g. a cubic-
spline PSF): the angle-averaged PSF ('mean' = the exact average of the angle 0 / angle 1 PSFs, what SMLM / SIMFLUX / DECODE use)
and, for a per-angle calib, angle0 / angle1 (pattern angle = frame % 6 // 3).
Per PSF, two arrays (float32, photons-normalised like the simulator: sum over the full image plane = 1 for an aberration-free
emitter at z = 0, so the sum over the ROI is < 1):
  <name>        [nz, R, R]          pixel-integrated PSF, emitter at the centre of the centre pixel, R = 31
  <name>_os     [nz, R*os, R*os]    the same PSF at os = 4 x finer emitter shifts: element [iz, i, j] = pixel-integrated value of the
                                    pixel whose centre is (j/os - c, i/os - c) px from the emitter (c = (R - 1) / 2 = 15), i.e. the
                                    shift-invariant pixel-integrated PSF g(u, v) on a 1/os px grid (the PSFLUT knot grid)
z: emitter height above the coverslip, z_um = z0 + dz * arange(nz), z0 = -0.03, dz = 0.01, up to 0.23 um (fit range [0, 0.2] plus
3 knots of spline padding); focus fixed at the calib's focus_um.  Pixel size pixelsize_um (0.10833).  x = column, y = row.
usage: python analysis/psf3d/export_zstack.py <calib.json> <out.npz>"""
import sys, json
import numpy as np, torch
from simcode2.psf.vectorial import VectorialPSF, set_psf_extras, has_angle_psf

calib, outp = sys.argv[1:3]
dev = 'cuda'
c = json.load(open(calib))
R, OS = 31, 4
z0, dz, nz = -0.03, 0.01, 27
zs = z0 + dz * np.arange(nz)
psf = VectorialPSF(R, pixelsize=c['pixelsize_um'], NA=c['NA'], n_imm=c['n_imm'], n_med=c['n_med'], wavelength=c['wavelength_um'],
                   oversample=3, device=dev)
with torch.no_grad():
    psf.zcoef.copy_(torch.tensor(c['zernike_waves'], device=dev)); psf.apod.copy_(torch.tensor(c['apod'], device=dev))
out = dict(z_um=zs.astype(np.float32), pixelsize_um=np.float32(c['pixelsize_um']), dz_um=np.float32(dz), z0_um=np.float32(z0),
           focus_um=np.float32(c['focus_um']), os=np.int32(OS), roi=np.int32(R), calib=calib,
           note=__doc__)
angles = ([0, 1] if has_angle_psf(c) else []) + [None]
a = (np.arange(OS) / OS)
with torch.no_grad():
    for ang in angles:
        set_psf_extras(psf, c, angle=ang)
        nm = 'mean' if ang is None else f'angle{ang}'
        if ang is None and 'angle0' in out:   # exact average of the two axis PSFs
            out['mean'] = 0.5 * (out['angle0'] + out['angle1']); out['mean_os'] = 0.5 * (out['angle0_os'] + out['angle1_os'])
            print(nm, 'sum over ROI at z 0 / 0.1:', out[nm][3].sum(), out[nm][13].sum()); continue
        st = torch.stack([psf(torch.tensor([[0.0, 0.0, float(z)]], device=dev), c['focus_um'])[0] for z in zs])
        out[nm] = st.cpu().numpy().astype(np.float32)
        # fine grid: emitter shifted by -(a_x, a_y) -> pixel (i, j) samples g at (j - c0 + a_x, i - c0 + a_y)
        big = np.zeros((nz, R * OS, R * OS), np.float32)
        for iy, ay in enumerate(a):
            pos = torch.tensor([[-ax, -ay, 0.0] for ax in a], device=dev).repeat(nz, 1)
            pos[:, 2] = torch.tensor(np.repeat(zs, OS), device=dev, dtype=torch.float32)
            im = psf(pos, c['focus_um']).reshape(nz, OS, R, R).cpu().numpy()     # [z, ax, i, j]
            for ix in range(OS): big[:, iy::OS, ix::OS] = im[:, ix]
        out[nm + '_os'] = big
        print(nm, 'sum over ROI at z 0 / 0.1:', out[nm][3].sum(), out[nm][13].sum())
np.savez_compressed(outp, **out)
print('wrote', outp)
