"""S1 step 5: drift (DME v2 on the astigmatic localizations, paper 3.6.3), filtering (eps_mod < 0.3, as the paper),
nanoruler cluster analysis exactly as zimflux cluster_picassopicksv2 (all_picks_v2.yaml, 94 picks, diameter 4 px; >50 locs;
K-means 2 on the astigmatic locs, ZIMFLUX clusters = same labels; both clusters > 30 locs, ZIMFLUX centre distance < 150 nm,
cluster x > 5000 nm; precision = sigma of a Gaussian fit to the 8 nm z histogram per cluster), lengths and CRLB.
The pick centres are in the paper's drift-corrected frame: a global (dx, dy) offset is found by maximising the number of
localizations inside the picks.
usage: python s1_post.py [eps=0.3] -> out/zimflux/s1/clusters.json, tables in simcode_results/classical/"""
import sys, json, os, numpy as np, yaml, scipy.stats
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
from s1_common import WORK, RES, PX
sys.path.insert(0, '/home/jc/simcode2')
from simcode.util.drift import dme_v2
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
A = dict(np.load(WORK + 'astig.npz')); Zf = dict(np.load(WORK + 'zf.npz')); ncyc = int(np.load(WORK + 'rois.npz')['ncycles'])
gi = Zf['idx']
def table(th, cr, idx):
    c = A['corner'][idx]
    return np.c_[(c[:, 1] + 7.5 + th[:, 0]) * PX, (c[:, 0] + 7.5 + th[:, 1]) * PX, th[:, 2] * 1000], \
        np.c_[cr[:, 0] * PX, cr[:, 1] * PX, cr[:, 2] * 1000]
pa, ca = table(A['th'][gi], A['crlb'][gi], gi); pz, cz = table(Zf['th'], Zf['crlb'], gi)
fr = A['cycle'][gi]
if os.path.exists(WORK + 'drift.npy'):
    drift = np.load(WORK + 'drift.npy')
else:
    drift = dme_v2(pa, fr, ca, numframes=ncyc); np.save(WORK + 'drift.npy', drift)
print('drift range nm', (drift.max(0) - drift.min(0)).round(1), flush=True)
pa_d = pa - drift[fr]; pz_d = pz - drift[fr]
keep = Zf['eps'] < EPS
print('localizations', len(gi), 'eps <', EPS, int(keep.sum()), flush=True)

picks = yaml.safe_load(open('/home/jc/simcode2/out/zimflux/ref/zimflux/ZIMFLUX/all_picks_v2.yaml'))
cen = np.array(picks['Centers']); rad = picks['Diameter'] / 2
P_ = pa_d[keep, :2] / PX
from scipy.spatial import cKDTree
tr = cKDTree(P_)
best = (0, 0, 0)
for swap in (False, True):
    cc = cen[:, ::-1] if swap else cen
    for dx in np.arange(-6, 6.01, 0.1):
        for dy in np.arange(-6, 6.01, 0.1):
            n = sum(len(l) for l in tr.query_ball_point(cc + [dx, dy], rad))
            if n > best[0]: best = (n, dx, dy, swap)
print('pick offset (locs, dx, dy, swap)', best, flush=True)
cc = (cen[:, ::-1] if best[3] else cen) + [best[1], best[2]]

def gauss(x, a, x0, s, c): return a * np.exp(-(x - x0) ** 2 / (2 * s ** 2) + c)
def gsig(v):
    bw = 8; n, b = np.histogram(v, bins=np.arange(v.min(), v.max() + bw, bw)); m0, s0 = scipy.stats.norm.fit(v)
    try:
        p, _ = curve_fit(gauss, (b[1:] + b[:-1]) / 2, n, p0=[n.max(), m0, s0, 0], maxfev=2000,
                         bounds=[[0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]])
        return float(p[2])
    except Exception:
        return float('nan')

PA, PZ, CA, CZ, NA_, NZ_ = pa_d[keep], pz_d[keep], ca[keep], cz[keep], A['th'][gi][keep, 3], Zf['th'][keep, 3]
out = []
for i, c in enumerate(cc):
    s = ((PA[:, 0] / PX - c[0]) ** 2 + (PA[:, 1] / PX - c[1]) ** 2) < rad ** 2
    if s.sum() <= 50: continue
    fa, fz = PA[s], PZ[s]
    km = KMeans(n_clusters=2, random_state=2, algorithm='elkan', n_init=10).fit(fa)
    lab = km.labels_
    if min((lab == 0).sum(), (lab == 1).sum()) <= 30: continue
    ca_ = np.array([fa[lab == k].mean(0) for k in range(2)]); cz_ = np.array([fz[lab == k].mean(0) for k in range(2)])
    if np.linalg.norm(cz_[0] - cz_[1]) >= 150 or km.cluster_centers_[0, 0] <= 5000: continue
    rec = {'pick': i, 'n': int(s.sum()), 'len_astig': float(np.linalg.norm(ca_[0] - ca_[1])),
           'len_zf': float(np.linalg.norm(cz_[0] - cz_[1])),
           'polar_zf_deg': float(np.degrees(np.arctan(np.hypot(*(cz_[0] - cz_[1])[:2]) / abs((cz_[0] - cz_[1])[2]))))}
    for k in range(2):
        m = lab == k
        rec[f'c{k}'] = {'n': int(m.sum()), 'z_astig': float(ca_[k, 2]), 'z_zf': float(cz_[k, 2]),
                        'sig_astig': [gsig(fa[m, j]) for j in range(3)], 'sig_zf': [gsig(fz[m, j]) for j in range(3)],
                        'std_astig': fa[m].std(0).tolist(), 'std_zf': fz[m].std(0).tolist(),
                        'crlb_astig': CA[s][m].mean(0).tolist(), 'crlb_zf': CZ[s][m].mean(0).tolist(),
                        'N': float(NA_[s][m].mean()),
                        # CRLB at the cluster-mean photon count (paper Fig. 8e: CRLB at the cluster-mean parameters), 1/sqrt(N) scaling
                        'crlbN_astig': (CA[s][m] * np.sqrt(NA_[s][m] / NA_[s][m].mean())[:, None]).mean(0).tolist(),
                        'crlbN_zf': (CZ[s][m] * np.sqrt(NZ_[s][m] / NZ_[s][m].mean())[:, None]).mean(0).tolist()}
    out.append(rec)
cl = [r[f'c{k}'] for r in out for k in range(2)]
med = lambda key, j: float(np.nanmedian([c[key][j] for c in cl]))
summ = {'eps': EPS, 'pick_offset': best, 'n_picks_used': len(out), 'n_clusters': len(cl),
        'median_sigma_z_astig_zf': [med('sig_astig', 2), med('sig_zf', 2)],
        'median_sigma_x_astig_zf': [med('sig_astig', 0), med('sig_zf', 0)],
        'median_sigma_y_astig_zf': [med('sig_astig', 1), med('sig_zf', 1)],
        'median_std_z_astig_zf': [med('std_astig', 2), med('std_zf', 2)],
        'median_crlb_z_astig_zf': [med('crlb_astig', 2), med('crlb_zf', 2)],
        'median_crlb_at_meanN_z_astig_zf': [med('crlbN_astig', 2), med('crlbN_zf', 2)],
        'median_crlb_x_astig_zf': [med('crlb_astig', 0), med('crlb_zf', 0)],
        'length_astig_mean_std': [float(np.mean([r['len_astig'] for r in out])), float(np.std([r['len_astig'] for r in out]))],
        'length_zf_mean_std': [float(np.mean([r['len_zf'] for r in out])), float(np.std([r['len_zf'] for r in out]))],
        'length_vs_polar_slope_zf_nm_per_deg': float(np.polyfit([r['polar_zf_deg'] for r in out], [r['len_zf'] for r in out], 1)[0]),
        'paper': {'sigma_z': [18.8, 12.6], 'crlb_z': [11.7, 6.0], 'length': [[83, 7], [83, 8]]}}
print(json.dumps(summ, indent=1))
json.dump({'summary': summ, 'picks': out}, open(WORK + f'clusters_eps{EPS}.json', 'w'), indent=1)
# tables for the results folder (all good localizations; eps column to filter); variant runs (ZF_S1_SUB) do not write there
if os.environ.get('ZF_S1_SUB') and not os.environ.get('ZF_S1_WRITE'): sys.exit()
os.makedirs(RES + 'classical', exist_ok=True)
np.savez_compressed(RES + 'classical/jan13_classical_locs.npz', cycle=fr, frame0=fr * 3,
                    astig_xyz_nm=pa, astig_crlb_nm=ca, zimflux_xyz_nm=pz, zimflux_crlb_nm=cz,
                    astig_xyz_nm_driftcorr=pa_d, zimflux_xyz_nm_driftcorr=pz_d, eps_mod=Zf['eps'],
                    photons_N=Zf['th'][:, 3], bg_per_px_frame=Zf['th'][:, 4], astig_N=A['th'][gi, 3], per_frame_Nk=A['Nk'][gi])
np.save(RES + 'classical/jan13_drift_nm_per_cycle.npy', drift)
