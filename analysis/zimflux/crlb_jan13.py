"""S0 CRLB on the paper movie (jan13): calibrated vectorial PSF of the zimflux code (vectorpsf/config.sim_params(depth=60):
NA 1.49, n_med 1.33, n_imm 1.52, lambda 715 nm, 65 nm px, Zernikes from delft_beads_feb23_15nm.npy with Z(4,0) = -0.2 nm),
imaging depth z_d = 60 nm, the measured pattern (inventory/pattern_jan13.json; p_ax 481 nm from the paper), measured
photons / background. Bounds as crlb_s0.py: astig / zimflux / fact (+ fact_joint check, prior_only).
Photon calibration: the movie statistic is the 7 x 7 box sum minus 49 x the 25 x 25 box mean (pattern_movie.py); the
model gives the fraction of the ROI photons that this statistic captures, so N_ROI per frame = box / fraction.
usage: python crlb_jan13.py [gpu] -> out/zimflux/crlb/crlb_jan13.json"""
import sys, json, numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
import crlb_s0 as C
from simcode.psf import VectorialPSF
ZD = 0.060
ab = np.load('/home/jc/simcode2/out/zimflux/ref/zimflux/vectorpsf/delft_beads_feb23_15nm.npy')
orders = [(2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0), (4, -2), (4, 2), (5, -1), (5, 1), (6, 0)]
ab = ab.copy(); ab[6] = -0.2                                          # config.sim_params: abberations[-6, 2] = -0.2 (nm)
ZERN = {C.NOLL[o]: float(a) / 715.0 for o, a in zip(orders, ab)}
pat = json.load(open('/home/jc/simcode2/out/zimflux/inventory/pattern_jan13.json'))
o = max(pat['orientations'], key=lambda o: len(o['slots']))
K = pat['cycle']; slots = o['slots']
PLAT = o['pitch_nm'] / 1000; ANG = o['angle_deg']; M = o['m_per_segment_median']
PAX = 0.481
rel = np.array(pat['slot_mean_intensity_rel']); ETA = (rel / rel.sum()).tolist()
st = pat['stats']; BG = float(np.median([s['bg_ph_px_median'] for s in st]))
box = pat['segment_box7_photons_per_frame_p10_50_90']
print('pattern', K, slots, PLAT, ANG, M, ETA, 'bg', BG, 'box p10/50/90', box, flush=True)

# fraction of the ROI photons in the box statistic (25 px ROI, emitter centred), mean over z_e 0..150 nm
f = C.focus_for_depth(ZD)
p25 = VectorialPSF(25, pixel_um=C.PX, NA=1.49, n_imm=1.52, n_med=1.33, wavelength_um=C.LAM, zernike_indices=sorted(ZERN),
                   npupil=64, oversample=3, device=C.dev)
with torch.no_grad():
    p25.zcoef[:] = torch.tensor([ZERN[j] for j in sorted(ZERN)], device=C.dev)
    ze = torch.linspace(0, 0.15, 7, device=C.dev); pos = torch.zeros(7, 3, device=C.dev); pos[:, 2] = ze
    h = p25(pos, torch.full((7,), f, device=C.dev)).double()
    p16 = h[:, 4:20, 4:20].sum((1, 2))                                   # the 16 x 16 ROI (sum normalisation of crlb_s0)
    stat = h[:, 9:16, 9:16].sum((1, 2)) - 49 * h.mean((1, 2))
frac = (stat / p16).cpu().numpy(); print('box statistic / ROI photons per z_e', frac.round(3), flush=True)
FR = float(frac.mean())
Nf = [b / FR for b in box]                                           # ROI photons per frame (p10 / 50 / 90)

def pattern(p0):   # slot s: phase from the measured slot phases (relative), eta from the slot intensities
    ph = np.deg2rad(o['slot_phases_deg'])
    q = 2 * np.pi / PLAT; kz = 2 * np.pi / PAX; a = np.deg2rad(ANG)
    return [(q * np.cos(a), q * np.sin(a), kz, p0 + ph[i] - ph[0], ETA[s], M) for i, s in enumerate(slots)]

R = {'pattern': {'K': K, 'plat_um': PLAT, 'angle_deg': ANG, 'm': M, 'eta': ETA, 'pax_um': PAX,
                 'slot_phases_deg': o['slot_phases_deg']},
     'bg_ph_px_frame': BG, 'box_fraction': frac.tolist(), 'N_roi_per_frame_p10_50_90': Nf, 'zernikes_waves': ZERN,
     'focus_um': f, 'runs': {}}
zr = np.arange(-60, 91, 10)                                          # z_e = 0..150 nm
for lab, n in zip(('p10', 'p50', 'p90'), Nf):
    r = C.run(ZERN, ZD, zr, K * n, BG, pattern, f'jan13 N {lab}')    # N = photons per K-frame localization
    r['mean_0_100nm'] = {k: np.mean(np.array(v)[(zr >= -60) & (zr <= 40)], 0).round(2).tolist() for k, v in r['crlb_nm_xyz'].items()}
    R['runs'][lab] = r
    print(lab, 'N/loc %.0f' % (K * n), {k: v for k, v in r['mean_0_100nm'].items()}, flush=True)
# photons at which the astig z CRLB (mean over z_e 0..100 nm) equals the paper's 11.7 nm (Fig. 8e)
from scipy.optimize import brentq
def zc(n, key):
    r = C.run(ZERN, ZD, np.arange(-60, 41, 20), n, BG, pattern, 'x'); return np.mean(np.array(r['crlb_nm_xyz'][key])[:, 2])
Np = brentq(lambda n: zc(n, 'astig') - 11.7, 200, 20000, xtol=20)
R['N_for_paper_astig_11.7'] = Np; R['zimflux_z_at_that_N'] = zc(Np, 'zimflux'); R['fact_z_at_that_N'] = zc(Np, 'fact')
print('N for astig 11.7 nm: %.0f, zimflux %.2f, fact %.2f' % (Np, R['zimflux_z_at_that_N'], R['fact_z_at_that_N']), flush=True)
json.dump(R, open(C.OUT + 'crlb_jan13.json', 'w'), indent=1, default=float)
