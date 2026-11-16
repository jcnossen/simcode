"""
The paper PSFs as psf.zip (models/**/*.psf.zip, analysis/psfzip/convert_paper_psfs.py) give exactly what their json gives.
CPU (env SIMCODE_TEST_DEVICE to change); needs only this checkout's models/ (json, camera maps, zips; no networks), and for the
4Pi state json the 4Pi work folder (skipped without it).
Run: python tests/test_psfzip.py [<4Pi work folder>] (or pytest tests/test_psfzip.py [--work4pi <4Pi work folder>])
  - read_calib(zip) (the zip's blocks) gives every key of the calib json with the json's value (camera.maps = ZIP_MAPS),
    plus only the defaults torch-psf writes explicitly (EXTRA_OK)
  - camera_maps(zip) == the json's npz (every entry: offset_adu, readnoise_e and the extra entries; values and dtype)
  - PSF.load(zip) renders bit-identically to PSF.load(json) (roisize 11 / oversample 3 and the stored defaults, angle
    average), the solve spline (simcode.psf.spline) has identical knots, the Simulator's LUT and pattern are identical,
    the zip's zstack.tif = the json PSF rendered at its planes
  - model folders: m1809 / m25 / g13 hold byte copies of the insitu zips; models.json / model.json point at the zips
  - bead fit zip: its Zernikes / NA / n_med = the fit json
  - 4Pi: fourpi_state(zip) == the state json; torch_psf FourPiPSF of the zip renders bit-identically to
    simcode.fourpi.coherent_psf, its 'raw' parameters are those of coherent_psf; the spline zip (spline_knots='rebuild') rebuilds the
    knots of simcode.fourpi.spline_table.LUT4Pi on the same device (GPU when there is one)
"""
import io, json, os, sys, zipfile
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DEV = os.environ.get('SIMCODE_TEST_DEVICE', 'cpu')
M = lambda f: os.path.join(REPO, 'models', f)
CALIBS = ['insitu/calib_1809_2_psf_v2', 'insitu/calib_2p5nM_m2_psf_v2', 'insitu/calib_20nM_m2_psf_v2', 'insitu/calib_gauss1.3',
          'm1809/calib_1809_2_psf_v2', 'm25/calib_2p5nM_m2_psf_v2', 'g13/calib_gauss1.3']
# keys a zip's blocks add to its json (torch-psf writes the values the json left to the defaults: the vectorial sampling
# and the Camera fields); every json key reads back with the json's value
EXTRA_OK = {'camera.type': 'scmos', 'camera.excess_noise': 1.0, 'npupil': 64, 'oversample': 3, 'roisize': 15,
            'saf': [0.0, 0.0, 0.0, 0.0], 'z_range_um': [0.0, 1.0]}


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
        for k, v in J.items():
            if isinstance(v, dict):
                assert all(C[k][kk] == vv for kk, vv in v.items()), (zp, k)
            else:
                assert C[k] == v, (zp, k)
        assert set(C) - set(J) <= set(EXTRA_OK) and set(C['camera']) - set(J['camera']) <= {'type', 'excess_noise'}, zp
        assert all(C[k] == EXTRA_OK[k] for k in set(C) - set(J)), zp          # the defaults the zip states
        assert all(C['camera'][k] == EXTRA_OK['camera.' + k] for k in set(C['camera']) - set(J['camera'])), zp


def test_camera_maps():
    from simcode import psf as spsf
    for js, zp in _pairs():
        a, b = spsf.camera_maps(js), spsf.camera_maps(zp)
        assert set(a) == set(b), zp
        for k in a:
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]), (zp, k)
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
        p = spsf.PSF.load(js, device='cpu')            # the zips were written on the CPU (convert_paper_psfs.py)
        zs = torch.tensor(meta['zstack']['z_um'], dtype=torch.float32)
        pos = torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], 1)
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
    assert np.float32(float(p.NA)) == np.float32(B['NA']) and np.float32(float(p.n_med)) == np.float32(B['n_med'])


def test_fourpi(work4pi):
    from simcode import psf as spsf
    from simcode.fourpi.coherent_psf import psf_from_state
    js = os.path.join(work4pi, 'psf4pi', 'NPC_f_dxy15.json') if work4pi else 'the 4Pi work folder (--work4pi)'
    if not os.path.isfile(js):
        print('skip 4Pi: no', js); return
    st = json.load(open(js))['psf']
    zp = M('fourpi/NPC_f_dxy15.psf.zip')
    assert spsf.fourpi_state(zp) == st and spsf.fourpi_state('NPC_f_dxy15') == st
    with zipfile.ZipFile(zp) as z:
        raw = json.loads(z.read('psf.json'))['source']['model']['fourpi']['raw']
    q0 = psf_from_state(st, device='cpu')
    for k, v in raw.items():          # the zip's exact parameters = those SIMCODE builds from the state
        assert torch.equal(torch.as_tensor(v, dtype=torch.float32).reshape(getattr(q0, k).shape), getattr(q0, k).detach()), k
    p = spsf.PSF.load(zp, device=DEV); q = psf_from_state(st, device=DEV).eval()
    pos = _positions(); ph = torch.linspace(-3, 3, len(pos), device=DEV)
    with torch.no_grad():
        assert torch.equal(p(pos, phi_extra=ph), q(pos, phi_extra=ph))
    sp = M('fourpi/NPC_f_dxy15_spline.psf.zip')
    if os.path.isfile(sp):
        from simcode.fourpi.spline_table import LUT4Pi
        with zipfile.ZipFile(sp) as z:
            rebuild = 'rebuild' in json.loads(z.read('psf.json'))['channels'][0]
        # spline_knots='rebuild': torch_psf tabulates the source on the GPU when there is one (else the CPU); on the GPU
        # the knots equal LUT4Pi on the GPU (the paper's 4Pi LUT) bit for bit
        dv = ('cuda' if torch.cuda.is_available() else 'cpu') if rebuild else DEV
        S = spsf.PSF.load(sp, device=dv); L = LUT4Pi(st, P=15, dev=dv); pos, ph = pos.to(dv), ph.to(dv)
        K0 = torch.stack([torch.stack([L.luts[i][c].knots for c in range(4)]) for i in range(3)])
        assert torch.equal(S.knots, K0)
        assert spsf.fourpi_state(sp) == st
        with torch.no_grad():
            assert torch.equal(S.evaluate(pos, ph), L.evaluate(pos, ph))


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            (f(sys.argv[1] if len(sys.argv) > 1 else None) if k == 'test_fourpi' else f()); print('ok', k, flush=True)
