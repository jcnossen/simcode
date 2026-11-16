"""CPU tests of the multi-channel layer: the 4Pi acquisition configs (setup fourpi), PolyMap (maps without feats_norm =
the crop-centre normalisation of the earlier 4Pi maps; json round trip), ChannelFront (registered sum and per-channel
patches with an identity map), and the stages of the set-ups in LocalizationPipeline.
Run: python tests/test_channels.py  (or pytest tests)"""
import json, os, sys, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
import numpy as np


def test_fourpi_configs():
    from simcode.setups.fourpi import acquisition, FourPiSetup
    S = FourPiSetup()
    for cell, delta in (('MT', -10.4), ('NPC', -13.9)):
        c = acquisition(cell)
        assert c['setup'] == 'fourpi' and c['name'] == f'fourpi_{cell}' and c['fourpi']['delta_deg'] == delta
        L = S.layout(c)
        assert (L.C, L.H, L.W) == (4, 200, 360) and [ch.mirror for ch in L.channels] == ['none', 'none', 'lr', 'lr']
        assert [ch.crop[0] for ch in L.channels] == [217, 652, 1420, 1855]
        assert S.stages(c) == ['registration', 'process', 'report']
    raw = np.random.default_rng(0).integers(0, 9999, (3, 200, 2304)).astype(np.uint16)
    x = S.layout(acquisition('MT')).split(raw)
    assert x.shape == (3, 4, 200, 360) and x.dtype == np.float32
    assert np.array_equal(x[:, 1], raw[:, :, 652:1012]) and np.array_equal(x[:, 3], raw[:, :, 1855:2215][..., ::-1])
    print('  ok 4Pi acquisition configs (channels, delta, stages, split)')


def test_polymap():
    from simcode.channels.registration import PolyMap
    rng = np.random.default_rng(1)
    nf = 6
    cf = {c: np.r_[[[1.0, 0.0]], rng.normal(0, 0.01, (nf - 1, 2))] for c in (1, 2, 3)}
    for c in cf: cf[c][0] = [180.0 + c, 100.0 - c]; cf[c][1] = [0.0, 100.0]; cf[c][3] = [180.0, 0.0]
    d = dict(order=2, fwd={str(c): v.tolist() for c, v in cf.items()}, inv={str(c): v.tolist() for c, v in cf.items()})
    pm = PolyMap.from_dict(d, size=(360, 200))
    assert pm.norm == (180.0, 100.0, 180.0, 100.0)
    x, y = rng.uniform(0, 360, 50), rng.uniform(0, 200, 50)
    u, v = (x - 180.0) / 180.0, (y - 100.0) / 100.0
    F = np.stack([u ** i * v ** j for i in range(3) for j in range(3 - i)], -1)       # the feats of the earlier fourpi/front.py
    assert np.array_equal(pm.fwd(2, x, y)[0], (F @ cf[2])[:, 0])
    with tempfile.TemporaryDirectory() as t:
        pm.save(f'{t}/m.json', source='test')
        q = PolyMap.load(f'{t}/m.json')
        assert q.norm == pm.norm and json.load(open(f'{t}/m.json'))['source'] == 'test'
        assert np.array_equal(q.inv(1, x, y)[1], pm.inv(1, x, y)[1])
    try:
        PolyMap.from_dict(d); raise AssertionError('a map without feats_norm needs the crop size')
    except ValueError:
        pass
    print('  ok PolyMap (earlier maps: crop-centre normalisation; json round trip)')


def test_channel_front_identity_map():
    import torch
    from simcode.channels.registration import PolyMap
    from simcode.channels.front import ChannelFront
    H, W, C, P = 40, 60, 2, 9
    I = np.zeros((6, 2)); I[0] = [W / 2, H / 2]; I[3, 0] = W / 2; I[1, 1] = H / 2      # identity (feats 1, v, v^2, u, uv, u^2)
    pm = PolyMap({1: I}, {1: I}, 2, PolyMap.default_norm(W, H))
    xs, ys = pm.fwd(1, np.array([3.0, 50.0]), np.array([7.0, 30.0]))
    assert np.allclose(xs, [3, 50]) and np.allclose(ys, [7, 30])
    fr = ChannelFront(pm, 100.0, 0.5, C, H, W, P=P, dev='cpu')
    a = np.random.default_rng(2).uniform(100, 200, (12, C, H, W)).astype(np.float32)
    E = fr.to_e(a)
    assert torch.allclose(E, (torch.tensor(a) - 100.0) * 0.5)
    s = fr.registered_sum(E)
    assert (s - (E[:, 0] + E[:, 1])).abs().max() < 0.05, float((s - (E[:, 0] + E[:, 1])).abs().max())   # float32 grid round trip
    w, cy, cx = torch.tensor([0, 1]), torch.tensor([10, 20]), torch.tensor([15, 30])
    img, doff, ok = fr.patches(E, w, cy, cx)
    assert img.shape == (2, 6, C, P, P) and ok.all() and doff.abs().max() < 1e-4
    assert torch.equal(img[1, 2, 1], E[6 + 2, 1, 20 - P // 2:20 + P // 2 + 1, 30 - P // 2:30 + P // 2 + 1])
    print('  ok ChannelFront (camera conversion, registered sum, patches) with an identity map')


def test_pipeline_stages():
    from simcode.setups.fourpi import acquisition
    from simcode.pipeline.localization_pipeline import LocalizationPipeline
    from simcode.util.config_dict import ConfigDict
    with tempfile.TemporaryDirectory() as t:
        c = ConfigDict(acquisition('MT')).updated(out=t, name='test_fourpi_stages')
        p = LocalizationPipeline(c, log=lambda *a: None)
        assert p.stages == ['registration', 'process', 'report'] and p.setup.name == 'fourpi'
        for bad in ('psf', 'spots'):
            try:
                p.run(only=[bad]); raise AssertionError(f'{bad} is not a 4Pi stage')
            except ValueError:
                pass
    link = os.path.join(REPO, 'out', 'auto', 'test_fourpi_stages')
    if os.path.islink(link): os.remove(link)
    print('  ok pipeline stages of the fourpi set-up')


if __name__ == '__main__':
    for f in (test_fourpi_configs, test_polymap, test_channel_front_identity_map, test_pipeline_stages):
        print(f.__name__, flush=True); f()
    print('ALL OK')
