"""Compare an automatic-pipeline in-situ calibration (simcode.pipeline stage d) with a reference calibration json of the same movie:
Zernikes, focus, blur, photoselection tensor, PSF images per pattern angle (rms difference relative to the peak, z 0-0.15 um),
camera maps (correlation and rms of the offset deviation and read-noise maps), and the CRLB of x / y (single frame, 1000
photons, bg 6 ph/px, z = 0.05 um; per angle and averaged).
usage: python analysis/auto/cmp_calib.py <auto calib.json> <reference calib.json> [out.json] [reference emitter z median um]"""
import json, os, sys
import numpy as np, torch
from simcode import config
from simcode import psf as spsf
from simcode.calibration import cam_maps


def model(path, angle, R=15, dev='cuda'):
    return spsf.PSF.load(path, roisize=R, oversample=3, device=dev, angle=angle)


def images(c, angle, z=(0.0, 0.05, 0.1, 0.15)):
    m = model(c['_path'], angle)
    with torch.no_grad():
        return m(torch.tensor([[0.0, 0.0, v] for v in z], device='cuda'), torch.full((len(z),), c['focus_um'], device='cuda')).cpu().numpy()


def crlb(c, angle, I=1000.0, bg=6.0, z=0.05):
    m = model(c['_path'], angle)
    th = torch.tensor([0.1, -0.1, I, bg], device='cuda', dtype=torch.float32)
    def mu(t):
        p = m(torch.stack([t[0], t[1], torch.tensor(z, device='cuda')])[None], torch.tensor([c['focus_um']], device='cuda'))[0]
        return t[2] * p + t[3]
    J = torch.autograd.functional.jacobian(mu, th).reshape(-1, 4)
    v = mu(th).reshape(-1).detach()
    Fi = (J / v[:, None]).T @ J
    return (torch.linalg.inv(Fi).diagonal()[:2].sqrt() * c['pixelsize_um'] * 1e3).tolist()


def main(a, b, outp=None, zmed_ref=None):
    A, B = dict(json.load(open(a)), _path=a), dict(json.load(open(b)), _path=b)
    zA = A.get('emitter_z_median_um', 0.05); zB = float(zmed_ref) if zmed_ref else B.get('emitter_z_median_um', 0.05)
    r = {}
    za, zb = np.array(A['zernike_waves']), np.array(B['zernike_waves'])
    r['zernike_diff'] = dict(rms=float(np.sqrt(np.mean((za - zb) ** 2))), max=float(np.abs(za - zb).max()),
                             noll_max=int(A['zernike_noll'][int(np.abs(za - zb).argmax())]), auto=za.round(4).tolist(), ref=zb.round(4).tolist())
    r['focus_nm'] = dict(auto=A['focus_um'] * 1e3, ref=B['focus_um'] * 1e3, diff=(A['focus_um'] - B['focus_um']) * 1e3)
    r['blur_px2'] = dict(auto=A.get('psf_blur_px2'), ref=B.get('psf_blur_px2'))
    for k in ('psf_dipole_M_angle', 'psf_dipole_M'):
        if k in A or k in B:
            r[k] = dict(auto=A.get(k), ref=B.get(k), maxdiff=float(np.abs(np.array(A[k]) - np.array(B[k])).max()) if k in A and k in B else None)
    r['gamma_zeta'] = dict(auto=[A.get('psf3d', {}).get('gamma'), A.get('psf3d', {}).get('zeta')], ref=[B.get('psf3d', {}).get('gamma'), B.get('psf3d', {}).get('zeta')])
    r['psf_images'] = {}
    for ang in (0, 1, None):
        ia, ib = images(A, ang), images(B, ang)
        r['psf_images'][str(ang)] = dict(rms_rel_peak=float(np.sqrt(np.mean((ia - ib) ** 2)) / ib.max()),
                                         max_rel_peak=float(np.abs(ia - ib).max() / ib.max()),
                                         per_z_rms_rel_peak=[float(np.sqrt(np.mean((x - y) ** 2)) / y.max()) for x, y in zip(ia, ib)])
    # matched emitter z (focus / z gauge): auto at z + (zmed_auto - zmed_ref) vs ref at z, z = zmed_ref + (-0.04 .. +0.06) um
    zz = [zB + d for d in (-0.04, -0.02, 0.0, 0.02, 0.04, 0.06) if zB + d >= 0]
    r['psf_images_matched_z'] = dict(zmed_auto=zA, zmed_ref=zB)
    for ang in (0, 1, None):
        ia, ib = images(A, ang, [z + zA - zB for z in zz]), images(B, ang, zz)
        r['psf_images_matched_z'][str(ang)] = dict(rms_rel_peak=float(np.sqrt(np.mean((ia - ib) ** 2)) / ib.max()),
                                                   per_z=[float(np.sqrt(np.mean((x - y) ** 2)) / y.max()) for x, y in zip(ia, ib)], z_ref=zz)
    r['crlb_nm_xy_matched_z'] = {str(ang): dict(auto=crlb(A, ang, z=zA), ref=crlb(B, ang, z=zB)) for ang in (0, 1, None)}
    for ang, v in r['crlb_nm_xy_matched_z'].items(): v['ratio'] = [x / y for x, y in zip(v['auto'], v['ref'])]
    r['crlb_nm_xy'] = {str(ang): dict(auto=crlb(A, ang), ref=crlb(B, ang)) for ang in (0, 1, None)}
    for k in ('crlb_nm_xy',):
        for ang, v in r[k].items(): v['ratio'] = [x / y for x, y in zip(v['auto'], v['ref'])]
    ma, mb = np.load(cam_maps(a, A['camera'])), np.load(cam_maps(b, B['camera']))
    r['camera'] = dict(gain=[A['camera']['gain_e_per_adu'], B['camera']['gain_e_per_adu']], offset=[A['camera']['offset_adu'], B['camera']['offset_adu']])
    for k in ('offset_adu', 'readnoise_e'):
        x, y = ma[k].astype(np.float64), mb[k].astype(np.float64)
        r['camera'][k] = dict(corr=float(np.corrcoef(x.ravel(), y.ravel())[0, 1]), rms_diff=float(np.sqrt(np.mean((x - y) ** 2))),
                              sd_ref=float(y.std()), median_auto=float(np.median(x)), median_ref=float(np.median(y)))
    print(json.dumps(r, indent=1))
    if outp: json.dump(r, open(outp, 'w'), indent=1)
    return r


if __name__ == '__main__':
    main(*sys.argv[1:5])
