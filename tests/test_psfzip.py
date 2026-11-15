"""
The paper PSFs as psf.zip (models/**/*.psf.zip, analysis/psfzip/convert_paper_psfs.py) give exactly what their json gives.
CPU (env SIMCODE_TEST_DEVICE to change); needs only this checkout's models/ (json, camera maps, zips; no networks).
Run: python tests/test_psfzip.py (or pytest tests/test_psfzip.py)
  - read_calib(zip) == the calib json (camera.maps = ZIP_MAPS); the zip's blocks alone agree with the json up to the gaps
    the format has (calib_gaps: camera scalars of a camera with maps = map medians, float32-rounded values, notes)
  - camera_maps(zip) == the json's npz maps (offset_adu, readnoise_e: same values and dtype)
  - PSF.load(zip) renders bit-identically to PSF.load(json) (roisize 11 / oversample 3 and the stored defaults, angle
    average), the solve spline (simcode.psf.spline) has identical knots, the Simulator's LUT and pattern are identical,
    the zip's zstack.tif = the json PSF rendered at its planes
  - model folders: m1809 / m25 / g13 hold byte copies of the insitu zips; models.json / model.json point at the zips
  - bead fit zip: its Zernikes / NA / n_med = the fit json
  - 4Pi: fourpi_state(zip) == the state json; torch_psf FourPiPSF of the zip renders bit-identically to
    simcode.fourpi.psf4pi; the spline zip (if present) has the knots of simcode.fourpi.lut4pi.LUT4Pi on the same device
"""
import io, json, os, sys, zipfile
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DEV = os.environ.get('SIMCODE_TEST_DEVICE', 'cpu')
M = lambda f: os.path.join(REPO, 'models', f)
CALIBS = ['insitu/calib_1809_2_psf_v2', 'insitu/calib_2p5nM_m2_psf_v2', 'insitu/calib_20nM_m2_psf_v2', 'insitu/calib_gauss1.3',
          'm1809/calib_1809_2_psf_v2', 'm25/calib_2p5nM_m2_psf_v2', 'g13/calib_gauss1.3']
# keys where a zip's blocks alone (exact=False) may differ from its json (the format's gaps, see simcode.psf.calib_gaps)
GAP_OK = {'camera.maps', 'camera.note', 'camera.replaces', 'camera.type', 'camera.excess_noise', 'camera.offset_adu',
          'camera.readnoise_e', 'camera.sensor_shift_vs_0.5nM', 'pattern.note', 'pattern_note', 'n_med', 'NA', 'n_imm',
          'wavelength_um', 'npupil', 'oversample', 'roisize', 'saf', 'z_range_um'}


def _pairs():
    """(json, zip); a model-folder zip is compared with the insitu json (the folder's json copy refers to camera maps that
    are only in insitu/: models/g13/calib_gauss1.3.json -> camera_maps_1809_2.npz)"""
    return [(M('insitu/' + os.path.basename(f) + '.json'), M(f + '.psf.zip')) for f in CALIBS]


def _positions():
    g = torch.Generator().manual_seed(5)
    return torch.cat([torch.rand(64, 2, generator=g) - 0.5, 0.4 * torch.rand(64, 1, generator=g)], 1).to(DEV)


def test_read_calib():
    from simcode import psf as spsf
    for js, zp in _pairs():
        J = json.load(open(js)); C = spsf.read_calib(zp)
        if J.get('camera', {}).get('maps'): J['camera']['maps'] = spsf.ZIP_MAPS
        assert C == J, (zp, {k for k in set(C) | set(J) if C.get(k) != J.get(k)})
        gaps = spsf.calib_gaps(zp)
        assert set(gaps) <= GAP_OK, (zp, set(gaps) - GAP_OK)
        B = spsf.read_calib(zp, exact=False)
        assert B['camera']['gain_e_per_adu'] == J['camera']['gain_e_per_adu']
        for k in ('emitter_z_quantiles_um', 'emitter_z_range_um', 'emitter_z_median_um', 'tirf_depth_um', 'focus_um',
                  'zernike_waves', 'zernike_noll', 'pixelsize_um', 'psf_gauss_sigma_um'):
            assert B.get(k) == J.get(k), (zp, k)
        assert {k: v for k, v in B['pattern'].items() if k != 'note'} == {k: v for k, v in J['pattern'].items() if k != 'note'}


def test_camera_maps():
    from simcode import psf as spsf
    for js, zp in _pairs():
        a, b = spsf.camera_maps(js), spsf.camera_maps(zp)
        for k in ('offset_adu', 'readnoise_e'):
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), (zp, k)
        assert spsf.camera_maps_path(zp) == os.path.abspath(zp)


def test_psf_render_spline_simulator():
    from simcode import psf as spsf
    from simcode.simulate.simulator import Simulator
    pos = _positions()
    for js, zp in _pairs():
        for kw in (dict(roisize=11, oversample=3), dict()):
            a, b = spsf.PSF.load(js, device=DEV, **kw), spsf.PSF.load(zp, device=DEV, **kw)
            assert type(a) is type(b)
            with torch.no_grad():
                assert torch.equal(spsf.render(a, pos), spsf.render(b, pos)), (zp, kw)
        with torch.no_grad():
            ka = spsf.spline(spsf.PSF.load(js, roisize=11, oversample=3, device=DEV), 0.2).knots
            kb = spsf.spline(spsf.PSF.load(zp, roisize=11, oversample=3, device=DEV), 0.2).knots
        assert torch.equal(ka, kb), zp
        if 'gauss' in zp: continue
        sa, sb = (Simulator(p, H=16, W=16, T=6, device=DEV, psf_roi=15, psf_lut_z=11, psf_lut_sub=8) for p in (js, zp))
        assert torch.equal(sa.lut, sb.lut) and torch.equal(sa.k0, sb.k0) and torch.equal(sa.depth0, sb.depth0), zp
        assert torch.equal(sa.rn_map, sb.rn_map) and (sa.gain, sa.offset, sa.rn) == (sb.gain, sb.offset, sb.rn), zp


def test_zstack_member():
    import tifffile
    from simcode import psf as spsf
    for js, zp in _pairs()[:4]:
        with zipfile.ZipFile(zp) as z:
            meta = json.loads(z.read('psf.json')); st = tifffile.imread(io.BytesIO(z.read('zstack.tif')))
        p = spsf.PSF.load(js, device=DEV)
        zs = torch.tensor(meta['zstack']['z_um'], dtype=torch.float32)
        pos = torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], 1).to(DEV)
        with torch.no_grad():
            ref = p.render(pos).cpu().numpy()
        assert st.size == ref.size and np.array_equal(st.reshape(ref.shape), ref), zp      # (tifffile squeezes a single plane)


def test_model_folders():
    import hashlib
    h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
    for f in ('calib_1809_2_psf_v2', 'calib_2p5nM_m2_psf_v2', 'calib_gauss1.3'):
        m = {'calib_1809_2_psf_v2': 'm1809', 'calib_2p5nM_m2_psf_v2': 'm25', 'calib_gauss1.3': 'g13'}[f]
        assert h(M(f'insitu/{f}.psf.zip')) == h(M(f'{m}/{f}.psf.zip'))
        assert json.load(open(M(f'{m}/model.json')))[m]['psf'] == f'{f}.psf.zip'
    S = json.load(open(os.path.join(REPO, 'simcode', 'models.json')))
    for m in ('m1809', 'm25', 'g13'):
        assert S[m]['psf'].endswith('.psf.zip') and os.path.isfile(M(S[m]['psf']))
    for d in S['_production']['datasets'].values():
        assert d['calib'].endswith('.psf.zip') and os.path.isfile(M(d['calib']))


def test_beads():
    from simcode import psf as spsf
    B = json.load(open(M('insitu/beads/psf_fit_sign-1_na149_ball_drift.json')))
    p = spsf.PSF.load(M('insitu/beads/psf_fit_sign-1_na149_ball_drift.psf.zip'), device=DEV)
    assert [int(j) for j in p.zidx] == sorted(int(k) for k in B['zernike'])
    assert torch.equal(p.zcoef.detach().cpu(), torch.tensor([B['zernike'][str(j)] for j in sorted(int(k) for k in B['zernike'])]))
    assert np.float32(p.NA) == np.float32(B['NA']) and np.float32(p.n_med) == np.float32(B['n_med'])


def test_fourpi():
    from simcode import psf as spsf, config
    from simcode.fourpi.psf4pi import psf_from_state
    js = os.path.join(config.root('fourpi'), 'psf4pi', 'NPC_f_dxy15.json')
    if not os.path.isfile(js):
        print('skip 4Pi: no', js); return
    st = json.load(open(js))['psf']
    zp = M('fourpi/NPC_f_dxy15.psf.zip')
    assert spsf.fourpi_state(zp) == st and spsf.fourpi_state('NPC_f_dxy15') == st
    p = spsf.PSF.load(zp, device=DEV); q = psf_from_state(st, device=DEV).eval()
    pos = _positions(); ph = torch.linspace(-3, 3, len(pos), device=DEV)
    with torch.no_grad():
        assert torch.equal(p(pos, phi_extra=ph), q(pos, phi_extra=ph))
    sp = M('fourpi/NPC_f_dxy15_spline.psf.zip')
    if os.path.isfile(sp):
        from simcode.fourpi.lut4pi import LUT4Pi
        S = spsf.PSF.load(sp, device=DEV); L = LUT4Pi(st, P=15, dev=DEV)
        K0 = torch.stack([torch.stack([L.luts[i][c].knots for c in range(4)]) for i in range(3)])
        assert torch.equal(S.knots, K0)
        assert spsf.fourpi_state(sp) == st
        with torch.no_grad():
            assert torch.equal(S.evaluate(pos, ph), L.evaluate(pos, ph))


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k, flush=True)
