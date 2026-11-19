"""
The PSFs of the example models as psf.zip (models/<model>/*.psf.zip) give exactly what their json gives.
CPU (env SIMCODE_TEST_DEVICE to change); needs only this checkout's models/ (json, camera maps, zips; no networks), and for the
4Pi state json the 4Pi work folder and the 4Pi PSF zip models/fourpi/NPC_f_dxy15.psf.zip (paper_figures branch; skipped without them).
Run: python tests/test_psfzip.py [<4Pi work folder>] (or pytest tests/test_psfzip.py [--work4pi <4Pi work folder>])
  - read_calib(zip) (the zip's blocks) gives every key of the calib json with the json's value (camera.maps = ZIP_MAPS),
    plus only the defaults torch-psf writes explicitly (EXTRA_OK)
  - camera_maps(zip) == the json's npz (every entry: offset_adu, readnoise_e and the extra entries; values and dtype)
  - PSF.load(zip) renders bit-identically to PSF.load(json) (roisize 11 / oversample 3 and the stored defaults, angle
    average), the PSF renderer (simcode.psf.calib_psf, the integrated spline) has identical coefficients, the Simulator's PSF
    and pattern are identical, the zip's zstack.tif = the json PSF rendered at its planes
  - write_calib_zip (the integrated spline in the zip): read_calib and camera_maps give what the vectorial zip gives, calib_psf
    reads the spline from the zip = the one built from the json
  - model folders: models.json / model.json point at the zips
  - 4Pi: fourpi_state(zip) == the state json; torch_psf FourPiPSF of the zip renders bit-identically to the one SIMCODE
    builds from the state json (simcode.psf.fourpi_psf), its 'raw' parameters are those of fourpi_psf; the spline zip
    (spline_knots='rebuild') rebuilds the knots of SIMCODE's 4Pi table (simcode.psf.fourpi_spline(state, 15)) on the same
    device (GPU when there is one)
"""
import io, json, os, sys, zipfile
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DEV = os.environ.get('SIMCODE_TEST_DEVICE', 'cpu')
M = lambda f: os.path.join(REPO, 'models', f)
# (json, zip): the m1809 folder's json copy; the g13 folder's json copy refers to camera maps that are only in insitu/
# (camera_maps_1809_2.npz), so its zip is compared with models/insitu/calib_gauss1.3.json
PAIRS = [('m1809/calib_1809_2_psf_v2.json', 'm1809/calib_1809_2_psf_v2.psf.zip'), ('insitu/calib_gauss1.3.json', 'g13/calib_gauss1.3.psf.zip')]
# keys a zip's blocks add to its json (torch-psf writes the values the json left to the defaults: the vectorial sampling
# and the Camera fields); every json key reads back with the json's value
EXTRA_OK = {'camera.type': 'scmos', 'camera.excess_noise': 1.0, 'npupil': 64, 'oversample': 3, 'roisize': 15,
            'saf': [0.0, 0.0, 0.0, 0.0], 'z_range_um': [0.0, 1.0]}


def _pairs():
    """(json, zip) of each example model (PAIRS)"""
    return [(M(j), M(z)) for j, z in PAIRS]


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
                assert torch.equal(a.render(pos), b.render(pos)), (zp, kw)
        if 'gauss' in zp: continue
        ka, kb = spsf.calib_psf(js, device=DEV), spsf.calib_psf(zp, device=DEV)
        assert torch.equal(ka.coefs, kb.coefs), zp
        sa, sb = (Simulator(p, H=16, W=16, T=6, device=DEV, psf_roi=15) for p in (js, zp))
        with torch.no_grad():
            assert torch.equal(sa.psf.render(pos), sb.psf.render(pos)), zp
        assert torch.equal(sa.k0, sb.k0) and torch.equal(sa.depth0, sb.depth0), zp
        assert torch.equal(sa.rn_map, sb.rn_map) and (sa.gain, sa.offset, sa.rn) == (sb.gain, sb.offset, sb.rn), zp


def test_write_calib_zip_spline():
    """write_calib_zip writes the integrated spline (channel 0, the vectorial model as source, psf.zip 0.3.0); the calibration
    reads back as from the vectorial zip; calib_psf reads the coefficients = those built from the json"""
    import tempfile, pathlib
    from simcode import psf as spsf
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    js = M('m1809/calib_1809_2_psf_v2.json')
    out = str(tmp_path / 'c.psf.zip'); out0 = str(tmp_path / 'c0.psf.zip')
    J = json.load(open(js)); J['camera']['maps'] = os.path.abspath(os.path.join(os.path.dirname(js), J['camera']['maps']))
    jj = str(tmp_path / 'c.json'); json.dump(J, open(jj, 'w'))
    spsf.write_calib_zip(jj, out, device=DEV); spsf.write_calib_zip(jj, out0, spline=False, device=DEV)
    with zipfile.ZipFile(out) as z:
        meta = json.loads(z.read('psf.json'))
    assert meta['version'] == '0.3.0' and meta['channels'][0]['type'] == 'integrated_spline'
    assert meta['channels'][0]['source']['type'] == 'vectorial'
    assert spsf.read_calib(out) == spsf.read_calib(out0)
    a, b = spsf.camera_maps(out), spsf.camera_maps(out0)
    assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
    s1, s2 = spsf.calib_psf(out, 11, device=DEV), spsf.calib_psf(jj, 11, device=DEV)
    assert torch.equal(s1.coefs, s2.coefs) and s1.roisize == 11
    v = spsf.vectorial_psf(out, 11, device=DEV)
    assert isinstance(v, spsf.VectorialPSF)
    with torch.no_grad():
        pos = _positions()
        assert torch.equal(v.render(pos), spsf.vectorial_psf(jj, 11, device=DEV).render(pos))


def test_zstack_member():
    import tifffile
    from simcode import psf as spsf
    for js, zp in _pairs():
        with zipfile.ZipFile(zp) as z:
            meta = json.loads(z.read('psf.json')); st = tifffile.imread(io.BytesIO(z.read('zstack.tif')))
        p = spsf.PSF.load(js, device='cpu')            # the zips were written on the CPU (convert_paper_psfs.py)
        zs = torch.tensor(meta['zstack']['z_um'], dtype=torch.float32)
        pos = torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], 1)
        with torch.no_grad():
            ref = p.render(pos).cpu().numpy()
        assert st.size == ref.size and np.array_equal(st.reshape(ref.shape), ref), zp      # (tifffile squeezes a single plane)


def test_model_folders():
    for m, f in (('m1809', 'calib_1809_2_psf_v2'), ('g13', 'calib_gauss1.3')):
        assert json.load(open(M(f'{m}/model.json')))[m]['psf'] == f'{f}.psf.zip'
    S = json.load(open(os.path.join(REPO, 'simcode', 'models.json')))
    for m in ('m1809', 'g13'):
        assert S[m]['psf'].endswith('.psf.zip') and os.path.isfile(M(S[m]['psf']))
    for d in S['_production']['datasets'].values():
        assert d['calib'].endswith('.psf.zip') and os.path.isfile(M(d['calib']))


def test_fourpi(work4pi):
    from simcode import psf as spsf
    js = os.path.join(work4pi, 'psf4pi', 'NPC_f_dxy15.json') if work4pi else 'the 4Pi work folder (--work4pi)'
    zp = M('fourpi/NPC_f_dxy15.psf.zip')
    if not os.path.isfile(js) or not os.path.isfile(zp):
        print('skip 4Pi: no', js, 'or', zp); return
    st = json.load(open(js))['psf']
    assert spsf.fourpi_state(zp) == st and spsf.fourpi_state('NPC_f_dxy15') == st
    with zipfile.ZipFile(zp) as z:
        raw = json.loads(z.read('psf.json'))['source']['model']['fourpi']['raw']
    q0 = spsf.fourpi_psf(st, device='cpu')
    for k, v in raw.items():          # the zip's exact parameters = those SIMCODE builds from the state
        assert torch.equal(torch.as_tensor(v, dtype=torch.float32).reshape(getattr(q0, k).shape), getattr(q0, k).detach()), k
    p = spsf.PSF.load(zp, device=DEV); q = spsf.fourpi_psf(st, device=DEV).eval()
    pos = _positions(); ph = torch.linspace(-3, 3, len(pos), device=DEV)
    with torch.no_grad():
        assert torch.equal(p(pos, phi_extra=ph), q(pos, phi_extra=ph))
    sp = M('fourpi/NPC_f_dxy15_spline.psf.zip')
    if os.path.isfile(sp):
        with zipfile.ZipFile(sp) as z:
            rebuild = 'rebuild' in json.loads(z.read('psf.json'))['channels'][0]
        # spline_knots='rebuild': torch_psf tabulates the source on the GPU when there is one (else the CPU); on the GPU
        # the knots equal SIMCODE's 4Pi table on the GPU (the paper's 4Pi LUT) bit for bit
        dv = ('cuda' if torch.cuda.is_available() else 'cpu') if rebuild else DEV
        S = spsf.PSF.load(sp, device=dv); L = spsf.fourpi_spline(st, 15, device=dv); pos, ph = pos.to(dv), ph.to(dv)
        assert torch.equal(S.knots, L.knots)
        assert spsf.fourpi_state(sp) == st
        with torch.no_grad():
            assert torch.equal(S.evaluate(pos, ph), L.evaluate(pos, ph))


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            (f(sys.argv[1] if len(sys.argv) > 1 else None) if k == 'test_fourpi' else f()); print('ok', k, flush=True)
