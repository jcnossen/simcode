"""
psf.pixel_integration of the psf stage (simcode.pipeline.psf_fit): the fit with pixel_integration 'exact' (torch-psf
VectorialPSF pixel_integration 'spectral') stores it in _psf.pt / _psf.json, model_from_pt and heldout() use it (or an
override), and calib_dict writes it; 'box' (deprecated since the default became 'exact', 2026-11-19) leaves the files as before. Small synthetic tracks (CPU or GPU).
Needs a torch-psf with VectorialPSF.pixel_integration (skipped otherwise).
Run: python tests/test_psf_pixint.py (or pytest tests/test_psf_pixint.py)
"""
import json, os, sys, tempfile
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from simcode.pipeline import psf_fit as P
from simcode import psf as spsf

OPT = dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.70, n_imm=1.518, n_med=1.33, tirf_depth_um=0.082)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _has_pixint():
    return hasattr(spsf.VectorialPSF(5, npupil=16, oversample=2, device='cpu'), 'pixel_integration')


def synth(B=48, T=6, R=11, photons=3000.0, seed=0):
    """tracks.npz-like dict rendered with the exact pixel integral (ADU, gain 0.5, offset 100)"""
    g = torch.Generator().manual_seed(seed)
    vec = spsf.VectorialPSF.from_optics(spsf.Optics.from_dict(OPT), R, npupil=32, oversample=2, device='cpu',
                                        pixel_integration='exact').set_zernike({6: 0.04})
    xy = torch.rand(B, 2, generator=g) - 0.5
    z = 0.02 + 0.12 * torch.rand(B, generator=g)
    with torch.no_grad():
        img = vec(torch.cat([xy, z[:, None]], 1), torch.full((B,), 0.1))
    mu = photons * img[:, None].expand(-1, T, -1, -1) + 10.0
    ph = torch.poisson(mu, generator=g)
    rs = np.random.RandomState(seed)
    return dict(rois=(ph.numpy() / 0.5 + 100).round().astype(np.uint16), mask=np.ones((B, T), bool),
                frames=(np.arange(T)[None] + 50 * np.arange(B)[:, None]), corner=rs.randint(0, 400, (B, 2)),
                xy_init=(xy.numpy() + (R - 1) / 2).astype(np.float32))


def test_pixel_integration_fit_files_heldout():
    if not _has_pixint():
        print('skipped: torch-psf without VectorialPSF.pixel_integration'); return
    D = synth()
    with tempfile.TemporaryDirectory() as d:
        kw = dict(optics=OPT, gain=0.5, nsamples=48, iters=40, zmax=0.2, readnoise_e=1.0, offset_adu=100.0, dev=DEV,
                  log=lambda *a: None, init_zcoef=[0.0, 0.04] + [0.0] * 15)
        rb = P.fit(D, os.path.join(d, 'box'), pixel_integration='box', **kw)
        re = P.fit(D, os.path.join(d, 'ex'), **kw)                                  # the default: exact
        assert rb['pixel_integration'] == 'box' and re['pixel_integration'] == 'spectral'
        assert torch.load(os.path.join(d, 'ex_psf.pt'), weights_only=False)['pixel_integration'] == 'spectral'
        assert json.load(open(os.path.join(d, 'ex_psf.json')))['pixel_integration'] == 'spectral'
        F, _ = P.model_from_pt(os.path.join(d, 'ex_psf.pt'), 11, DEV)
        assert F.model.pixel_integration == 'spectral' and all(v.pixel_integration == 'spectral' for v in F.angle_models())
        Fb, _ = P.model_from_pt(os.path.join(d, 'box_psf.pt'), 11, DEV)
        assert Fb.model.pixel_integration == 'box'
        assert P.model_from_pt(os.path.join(d, 'box_psf.pt'), 11, DEV, 'exact')[0].model.pixel_integration == 'spectral'
        H = P.heldout(D, {'box': os.path.join(d, 'box_psf.pt'), 'ex': os.path.join(d, 'ex_psf.pt'),
                          'box_as_ex': (os.path.join(d, 'box_psf.pt'), 'exact')}, '', nsamples=48, iters=60, dev=DEV,
                      log=lambda *a: None)
        assert all(np.isfinite(h['nll']).all() for h in H.values())
        # the same frozen parameters rendered exactly fit the exactly rendered data better
        assert P.paired(H, 'box', 'box_as_ex')['mean'] < 0
        # the calibration carries the setting (only when not 'box')
        from simcode.pipeline import psf_stage
        class DS: c = dict(movie='m', name='n', optics=OPT, psf=dict(blur='none', photosel=0))
        pat = dict(k=[[1.0, 2.0], [-2.0, 1.0]], mod=[0.5] * 6, phase=[0.0] * 6, relint=[1.0] * 6)
        cam = dict(gain=0.5, offset=100.0, readnoise_e=1.0, gain_source='', offset_source='', readnoise_source='')
        ce = psf_stage.calib_dict(DS(), os.path.join(d, 'ex_psf.json'), os.path.join(d, 'ex_psf.pt'), pat, cam, '', True)
        cb = psf_stage.calib_dict(DS(), os.path.join(d, 'box_psf.json'), os.path.join(d, 'box_psf.pt'), pat, cam, '', True)
        assert ce['pixel_integration'] == 'spectral' and 'pixel_integration' not in cb
        assert spsf.PSF.load(ce, roisize=11, device='cpu').pixel_integration == 'spectral'


def test_config_and_stage_settings():
    """psf.pixel_integration defaults to 'exact' (2026-11-19), 'box' is accepted with a deprecation warning, invalid values are
    refused; at 'box' (the fits before the key) it is not part of the psf stage settings, so a work folder recorded before the
    key is up to date at 'box' and stale under the default 'exact'"""
    from types import SimpleNamespace
    from simcode.pipeline import acquisition_config as AC
    from simcode.pipeline import localization_pipeline as LP
    import warnings
    assert AC.DEFAULTS['psf']['pixel_integration'] == 'exact' and P.FIT_PIXEL_INTEGRATION == 'exact'
    assert LP.LATE_DEFAULTS['psf.pixel_integration'] == 'box'
    try:
        AC.load(dict(name='t', movie='x.tif', out='o', psf=dict(pixel_integration='spectral')))
        raise AssertionError('pixel_integration spectral accepted')
    except ValueError:
        pass
    settings = lambda c: LP.LocalizationPipeline._settings(SimpleNamespace(c=c, setup=SimpleNamespace(defaults=lambda: {})), 'psf')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cb = AC.load(dict(name='t', movie='x.tif', out='o', psf=dict(pixel_integration='box')))
    assert any('pixel_integration' in str(x.message) for x in w)
    S = settings(cb)
    assert 'pixel_integration' not in S['psf']
    old = json.loads(json.dumps(cb)); old['psf'].pop('pixel_integration')         # a config from before the key (box fits)
    assert settings(old) == S
    ce = AC.load(dict(name='t', movie='x.tif', out='o'))                            # the default
    assert ce['psf']['pixel_integration'] == 'exact'
    assert settings(ce)['psf']['pixel_integration'] == 'exact' and settings(ce) != S   # box work folders are stale


if __name__ == '__main__':
    test_pixel_integration_fit_files_heldout(); test_config_and_stage_settings(); print('ok')
