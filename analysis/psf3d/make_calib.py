"""(2026-11-11 in-situ 3D PSF) Assemble a calib json from a fit_insitu2 _psf.pt (+ opt-in extras) and an existing calib of the
same movie (camera block + maps, pattern block, optics constants are copied from it). Writes a NEW file.
PSF fields written: zernike_waves, focus_um, NA, apod (emitter_z_median_um, tirf_depth_um stay those of the base calib); extras -> psf_blur_px2 (global blur),
psf_dipole_M_angle (per-angle photoselection tensors, when the fit has them), saf, and a 'psf3d' provenance block.
With a 6th argument 'avg': the AVERAGE of the two axis PSFs as a pattern-free PSF (psf_dipole_M = mean of the per-angle
tensors, exact since the image is linear in M; a per-angle blur is averaged as a covariance, approximate), no per-angle keys.
usage: python analysis/psf3d/make_calib.py <fit_psf.pt> <fit_psf.json> <base calib.json> <out calib.json> <source note> [avg]"""
import sys, json, os
import torch

fit_pt, fit_js, base, outp, note = sys.argv[1:6]; AVG = len(sys.argv) > 6 and sys.argv[6] == 'avg' 
S = torch.load(fit_pt, weights_only=False); J = json.load(open(fit_js)); c = json.load(open(base))
ex = S.get('extras', {})
assert os.path.abspath(outp) != os.path.abspath(base) and not os.path.exists(outp), 'write a NEW calib file'
c['source'] = note
c['zernike_waves'] = [float(v) for v in S['zcoef'].tolist()]
c['focus_um'] = float(S['focus']); c['NA'] = float(S['NA']); c['apod'] = [float(v) for v in S['apod'].tolist()]
for k in ('psf_blur_px2', 'psf_dipole_M_angle', 'psf_blur_angle_px2', 'saf', 'psf_dipole_M', 'psf_dipole_w'):
    c.pop(k, None)
if 'blur_px2' in ex:
    if len(ex['blur_px2']) == 1: c['psf_blur_px2'] = ex['blur_px2'][0]
    else: c['psf_blur_angle_px2'] = ex['blur_px2']
if 'dipole_M' in ex: c['psf_dipole_M_angle'] = ex['dipole_M']
if 'dipole_w' in ex:
    if len(ex['dipole_w']) == 1: c['psf_dipole_w'] = ex['dipole_w'][0]
    else: c['psf_dipole_M_angle'] = [[[w[0] / 3, 0, 0], [0, w[1] / 3, 0], [0, 0, w[2] / 3]] for w in ex['dipole_w']]
if 'saf' in ex: c['saf'] = ex['saf']
if AVG:
    if 'psf_dipole_M_angle' in c:
        M = c.pop('psf_dipole_M_angle'); c['psf_dipole_M'] = [[(M[0][i][j] + M[1][i][j]) / 2 for j in range(3)] for i in range(3)]
    if 'psf_blur_angle_px2' in c:
        Bc = c.pop('psf_blur_angle_px2'); c['psf_blur_px2'] = [(Bc[0][k] + Bc[1][k]) / 2 for k in range(3)]
c['psf3d'] = dict(fit=fit_pt, mode=dict(blur=J.get('blur_mode', 'none'), dipole=J.get('dipole_mode', 'none'), fit_saf=J.get('fit_saf', 0)),
                  gamma=ex.get('gamma'), zeta=ex.get('zeta'), e_dir=ex.get('e_dir'), resid_var=J.get('resid_var'),
                  note='image-plane blur covariance px^2 [sxx, syy, sxy] (VectorialPSF.blur); psf_dipole_M_angle = emission-dipole '
                       'tensor per pattern angle (frame % 6 // 3), M_a = (1 - gamma) diag((1 - zeta)/2, (1 - zeta)/2, zeta) + gamma e_a e_a^T; '
                       'see docs/PSF_POLARIZATION.md' + ('; THIS FILE = the average of the two axis PSFs (pattern-free consumers)' if AVG else ''))
json.dump(c, open(outp, 'w'), indent=1)
print('wrote', outp)
