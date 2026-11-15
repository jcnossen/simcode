"""
Emitter heights from the calibration (2026-11-15 rule: a calibration with fitted spot heights, emitter_z_quantiles_um, sets the
height distribution of the simulator and the training batches; no flag). GPU (the simulator is CUDA-only); needs only the
calibration jsons / camera maps in models/insitu. Run: python tests/test_heights.py (or pytest tests)
  - height_quantiles: nodes = the calibration's quantiles + linear tails to probability 0 / 1; None without quantiles
  - draw_z of a v2 calibration reproduces its quantiles (inverse CDF), the LUT covers the whole range (below 0 too),
    tirf_factor = 1; the read-noise map is the calibration's camera map
  - a calibration without quantiles keeps the presets' draw: lognormal (same random draws as the explicit formula), LUT
    0 .. zmax, the TIRF factor, the preset's read-noise map
  - Simulator LUT with zmin < 0: rendering at a LUT plane gives that plane
  - TrainingBatches on a v2 calibration: GT heights follow the calibration
"""
import json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = 'insitu/calib_2p5nM_m2_psf_v2.json'
OLD = 'insitu/calib_1809_2_psf3d_avg.json'


def _cal(name):
    """the file in this checkout's models/ (the calibration jsons and camera maps are in git; the configured models root may
    be another checkout's)"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', name)


def test_height_quantiles_nodes():
    from simcode.simulate.acquisition import height_quantiles
    assert height_quantiles({}) is None and height_quantiles({'emitter_z_quantiles_um': {}}) is None
    P, Z = height_quantiles({'emitter_z_quantiles_um': {'q05': 0.0, 'q25': 0.1, 'q50': 0.2, 'q75': 0.3, 'q95': 0.5}})
    assert torch.allclose(P.cpu(), torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]))
    # tails: slope of the outer segments (0.1 / 0.2 per unit probability below, 0.2 / 0.2 above)
    assert torch.allclose(Z.cpu(), torch.tensor([-0.025, 0.0, 0.1, 0.2, 0.3, 0.5, 0.55]), atol=1e-6)
    try:
        height_quantiles({'emitter_z_quantiles_um': {'q25': 0.3, 'q75': 0.1}}); assert False
    except ValueError:
        pass


def test_v2_calibration_heights():
    from simcode.simulate.acquisition import AcquisitionSim
    c = json.load(open(_cal(V2)))
    q = c['emitter_z_quantiles_um']
    rs = AcquisitionSim('2.5nM', 32, 32, 6, npsf=1, calib=_cal(V2), zmax=0.3, z_unif=0.5)
    assert rs.zq is not None
    torch.manual_seed(0)
    z = rs.draw_z(400000).cpu().numpy()
    for k, v in q.items():
        assert abs(np.quantile(z, float(k[1:]) / 100) - v) < 0.004, (k, v, np.quantile(z, float(k[1:]) / 100))
    assert rs.zmin <= z.min() and z.max() <= rs.zmax + 1e-6 and rs.zmin < 0 and rs.zmax >= max(q.values())
    assert rs.sim.lut.shape[0] == int(round((rs.zmax - rs.zmin) / 0.01)) + 1
    assert torch.all(rs.tirf_factor(torch.tensor([-0.1, 0.0, 0.5], device='cuda')) == 1)
    maps = np.load(os.path.join(os.path.dirname(_cal(V2)), c['camera']['maps']))['readnoise_e']
    assert np.allclose(rs.rn_map.cpu().numpy(), maps)
    assert rs.sim.gain == 0.4256


def test_presets_unchanged():
    from simcode.simulate.acquisition import AcquisitionSim
    rs = AcquisitionSim('0.5nM', 32, 32, 6, npsf=1, calib=_cal(OLD), zmax=0.3)
    assert rs.zq is None and rs.zmin == 0.0 and rs.zmax == 0.3 and rs.sim.lut.shape[0] == 31
    torch.manual_seed(3); z = rs.draw_z(1000)
    torch.manual_seed(3); z0 = (rs.p['z_med'] * torch.exp(rs.p['z_sd'] * torch.randn(1000, device='cuda'))).clamp(0.0, 0.3)
    assert torch.equal(z, z0)
    zz = torch.tensor([0.0, 0.1], device='cuda')
    assert torch.allclose(rs.tirf_factor(zz), torch.exp(-(zz - rs.p['z_med']) / rs.sim.tirf))
    # the calibration's camera maps (camera_maps_1809_2.npz) replace the preset's (camera_maps_0.5nM.npz)
    assert np.allclose(rs.rn_map.cpu().numpy(), np.load(_cal('insitu/camera_maps_1809_2.npz'))['readnoise_e'])


def test_lut_negative_heights():
    from simcode.simulate.simulator import Simulator
    sim = Simulator(_cal(V2), H=16, W=16, T=6, psf_roi=15, zmin=-0.2, zmax=0.4, psf_lut_z=61)
    assert abs(float(sim.zs[0]) + 0.2) < 1e-6 and abs(float(sim.zs[-1]) - 0.4) < 1e-6
    for k in (0, 7, 20, 60):
        zk = sim.zs[k:k + 1]
        r = sim.render_psf(torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda'), zk)
        assert torch.allclose(r[0], sim.lut[k, 16, 16], atol=1e-6), k
    # below 0 the PSF differs from the one at 0 (the LUT is not clamped there)
    assert (sim.lut[0, 16, 16] - sim.lut[20, 16, 16]).abs().max() > 1e-4


def test_training_batches_heights():
    from simcode.simulate.batches import TrainingBatches
    torch.manual_seed(1)
    gen = TrainingBatches(32, 32, 6, seed=0, calib=[_cal(V2)])
    h = gen.heights()[0]
    assert h != 'presets' and h['lut_um'][0] < 0
    zs = torch.cat([gen.detector_batch(8)['gt']['z'] for _ in range(4)]).cpu().numpy()
    q = json.load(open(_cal(V2)))['emitter_z_quantiles_um']
    assert abs(np.median(zs) - q['q50']) < 0.05 and zs.min() < 0


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print(k, 'ok')
    print('ALL OK')
