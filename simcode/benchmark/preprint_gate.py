"""SIMCODE (paper model + eq.-2 pattern fit + MVU merge) as a front-end on the gate scenes (simcode/benchmark/gate.py), written in
the gate npz format (front 'sc' keys) so simcode/benchmark/gate_report_detector_matched.py reports it like our sc front-ends.
  network   ~/smlmtorch/scripts/models/sf_conv_g1.3_tio2_L48/checkpoint_3.pt (the paper's model, see
            <out>/simcode_base/run_simcode.py), on the gate frames (photons, clamped >= 0 as MovieProcessor);
            LocalizationDetector prob_threshold 0.7, use_prob_weights False -> x, y, sigma_xy, N0..N5, sigma_N0..5
  pattern   the gate's patt_est (the same pattern estimate our sc front-end uses), converted to the SIMCODE convention
            exc = relint (1 + depth sin(k.r - phase)), phase = phi - pi/2, relint = rel / 6
  NDI       smlmtorch ndi_fit (eq. 2 weighted LSQ on N_t with sigma_N_t, x, y, N_angle0, N_angle1; ndi_with_bg False), with
            ndi_fit_dataset's start / limits; MVU merge (merge_estimates) with the network position
Two outputs per scene / seed:
  tag _scode    the paper's filter chain: net crlb_filter(0.2 px), mod_error < 0.1 (with the pattern used), |ndi - net| < 0.2 px;
                detections failing any filter get score 0 (excluded at the report's thr 0.5); estimate = merged (MVU)
  tag _scodenf  no filters: every p > 0.7 detection, estimate = merged (MVU) -- the 'network + eq. 2 + MVU' path
x_prior = network position, s = merged sigma (SIMCODE's reported precision), I_net = N0..N5.
A global coordinate offset (SIMCODE vs gate frame convention) is measured on the fly (median net - GT on matched iso targets,
printed) -- not removed in the npz (gate_report has no offset fit); fit_offset() below gives the global offset of ANY front
(SIMCODE or ours) for the paired comparison (<out>/simcode_base/gate_paired.py --offset, critic 32 #2).
usage: CUDA_VISIBLE_DEVICES=1 python -m simcode.benchmark.preprint_gate --scen iso,pair2,rs05p,rs25p,rs05pm,rs25pm
"""
import argparse, json, math, os, time
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import numpy as np, torch
import smlmtorch.util.progbar
smlmtorch.util.progbar.USE_AUTO_TQDM = False
from smlmtorch import config_dict
from smlmtorch.nn.sf_model import SFLocalizationModel
from smlmtorch.nn.localization_detector import LocalizationDetector
from smlmtorch.simflux.ndi_fit_torch import ndi_fit
from simcode.simulate import gate_layouts as ST
from simcode.simulate.gate_layouts import CD, dev, S
from simcode.benchmark import gate as GT
from simcode import config as _cfg

# SIMCODE paper networks (smlmtorch repository, read only): directory from env SMLMTORCH_MODELS or config key 'smlmtorch_models'
SMLMTORCH_MODELS = os.environ.get('SMLMTORCH_MODELS') or _cfg._config().get('smlmtorch_models') or os.path.expanduser('~/smlmtorch/scripts/models')
MDIR = os.path.join(SMLMTORCH_MODELS, 'sf_conv_g1.3_tio2_L48')
PITCH_PX = 224.2 / 108.33


def fit_offset(R, ti, oi, xkey='sc_x_sc', ykey='sc_y_sc', lo=400.0, hi=2000.0):
    """global (x, y) offset of a front = median (estimate - GT) over matched target pairs (ti = GT rows, oi = output rows of
    the gate npz dict R) with GT window photons in [lo, hi); does not modify R. -> np.array([dx, dy]) px (zeros if < 20 pairs)"""
    ti, oi = np.asarray(ti, int), np.asarray(oi, int)
    m = (R['gt_phot'][ti] >= lo) & (R['gt_phot'][ti] < hi)
    if m.sum() < 20: return np.zeros(2)
    return np.median(np.c_[R[xkey][oi[m]] - R['gt_x'][ti[m]], R[ykey][oi[m]] - R['gt_y'][ti[m]]], 0)


def load_net(ck=3, mdir=None):
    MDIR_ = mdir or MDIR
    config = config_dict.load(MDIR_ + '/config.yaml')
    config.model.num_intensities = 6
    config.model.input_subpixel_index = False
    m = SFLocalizationModel(**config.model).to(dev).eval()
    st = torch.load(f'{MDIR_}/checkpoint_{ck}.pt', map_location='cpu')
    st = st['model_state_dict'] if 'model_state_dict' in st else st
    st.pop('output_scale', None)
    m.load_state_dict(st)
    return m, LocalizationDetector(prob_threshold=0.7, use_prob_weights=False)


@torch.no_grad()
def run_net(m, det, frames):
    """frames [nb, 6, S, S] photons -> per detection: b, x, y, sx, sy, N [n,6], sN [n,6], bg"""
    outs = []
    for c0 in range(0, frames.shape[0], 8):
        x = frames[c0:c0 + 8].clone().clamp(min=0)
        o, _ = m.forward(x, camera_calib=None, model_state=None)
        si, L = det.detect(o)
        if si is None: continue
        outs.append((si['batch'] + c0, L))
    if not outs: return None
    b = torch.cat([q[0] for q in outs]); L = torch.cat([q[1] for q in outs]).to(dev)
    pn = m.param_names; npar = (L.shape[1] - 1) // 2
    ix, iy, ib, iN = (pn.index(n) + 1 for n in ('x', 'y', 'bg', 'N0'))
    return dict(b=b.to(dev), p=L[:, 0], x=L[:, ix], y=L[:, iy], sx=L[:, ix + npar], sy=L[:, iy + npar], bg=L[:, ib],
                N=L[:, iN:iN + 6], sN=L[:, iN + npar:iN + npar + 6])


def sc_mod(patt, x0, y0):
    """patt [n,6,5] (kx, ky, phi, m, rel; I = rel (1 + m cos(k.r - phi))) -> SIMCODE mod [n,6,6] relative to (x0, y0)"""
    k = patt[..., :2]
    ph = patt[..., 2] - math.pi / 2 - (k[..., 0] * x0[:, None] + k[..., 1] * y0[:, None])
    mod = torch.zeros(*patt.shape[:2], 6, device=patt.device)
    mod[..., 0] = k[..., 0]; mod[..., 1] = k[..., 1]; mod[..., 3] = patt[..., 3]; mod[..., 4] = ph
    mod[..., 5] = patt[..., 4] / 6.0
    return mod


def excitation(mod, dx, dy):
    return mod[..., 5] * (1 + mod[..., 3] * torch.sin(mod[..., 0] * dx[:, None] + mod[..., 1] * dy[:, None] - mod[..., 4]))


@torch.no_grad()
def scode_front(m, det, o):
    r = run_net(m, det, o['frames'])
    if r is None: return None
    n = len(r['b'])
    PF = [[0, 1, 2], [3, 4, 5]]
    mod = sc_mod(o['patt_est'][r['b']].float(), r['x'], r['y'])               # [n,6,6]
    # mod error (pattern_estimator.mod_error): max |I_t / sum I - exc_t|
    exc = excitation(mod, torch.zeros(n, device=dev), torch.zeros(n, device=dev))
    me = (r['N'] / r['N'].sum(1, keepdim=True) - exc).abs().max(1).values
    lim = PITCH_PX
    pr = torch.tensor([[-lim / 2, lim / 2], [-lim / 2, lim / 2], [1, 1e9], [1, 1e9]], device=dev)
    init = torch.zeros(n, 4, device=dev); init[:, 2:] = (r['N'].sum(1) / 2)[:, None]
    params, _, _, crlb = ndi_fit(init, r['N'][:, PF], r['sN'][:, PF].clamp(min=1e-3), mod[:, PF], param_limits=pr, ndims=2)
    xn, yn = r['x'] + params[:, 0], r['y'] + params[:, 1]
    sxn, syn = crlb[:, 0], crlb[:, 1]
    fi_x = sxn ** -2 + r['sx'] ** -2; fi_y = syn ** -2 + r['sy'] ** -2
    xm = (xn * sxn ** -2 + r['x'] * r['sx'] ** -2) / fi_x; ym = (yn * syn ** -2 + r['y'] * r['sy'] ** -2) / fi_y
    dist = torch.hypot(params[:, 0], params[:, 1])
    ok_crlb = (r['sx'] < 0.2) & (r['sy'] < 0.2)
    ok = ok_crlb & (me < 0.1) & (dist < 0.2)
    base = dict(b=r['b'], x_det=r['x'], y_det=r['y'], x_prior=r['x'], y_prior=r['y'], x_sc=xm, y_sc=ym,
                s_sc=torch.stack([fi_x ** -0.5, fi_y ** -0.5], 1), scal_sc=torch.stack([fi_x ** -0.5, fi_y ** -0.5], 1),
                x_ndi=xn, y_ndi=yn, s_ndi=torch.stack([sxn, syn], 1), s_net=torch.stack([r['sx'], r['sy']], 1),
                I_net=r['N'], sI_net=r['sN'], bg_net=r['bg'], phot=r['N'].sum(1), q=me, p_net=r['p'],
                ok_crlb=ok_crlb.float(), ok_me=(me < 0.1).float(), ok_dist=(dist < 0.2).float())
    return dict(base, score=ok.float()), dict(base, score=torch.ones(n, device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='iso,pair2,rs05p,rs25p,rs05pm,rs25pm')
    ap.add_argument('--seeds', default='301,302,303')
    ap.add_argument('--ntarget', type=int, default=3500)
    ap.add_argument('--ckpt', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(_cfg.root('out'), 'gate'))
    # 2026-11-09 (simcode_base2): the paper's networks per dataset (<out>/simcode_base2/simcode_config.txt):
    # --model ni6_3 (0.5 nM) / L64_5 (1809, 2.5 nM movie 2); tags _scode_<model> / _scodenf_<model>. Default (L48) unchanged.
    ap.add_argument('--model', default='L48')
    ap.add_argument('--gt_from', default='_hd_sc5nb15pm')   # reuse gt_* (incl. CRLBs) of this existing tag (scenes are deterministic; verified identical); '' = recompute
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    CD.a.calib = _cfg.model(_cfg.INSITU_05)
    T = 6
    sim, rpsf, lut = CD.models(T)
    MD = {'L48': MDIR, 'ni6_3': os.path.join(SMLMTORCH_MODELS, 'sf_conv_g1.3_tio2_L64_ni6_3'),
          'L64_5': os.path.join(SMLMTORCH_MODELS, 'sf_conv_g1.3_L64_5')}
    m, det = load_net(a.ckpt, MD[a.model])
    tags = ('_scode', '_scodenf') if a.ckpt == 3 else (f'_scode{a.ckpt}', f'_scodenf{a.ckpt}')
    if a.model != 'L48': tags = tuple(t_ + '_' + a.model for t_ in tags)
    for sc in a.scen.split(','):
        for seed in [int(s_) for s_ in a.seeds.split(',')]:
            t0 = time.time(); GTL, OUTS = [], ([], []); need = a.ntarget; it = 0; fov0 = 0; tnet = 0.0
            ref = f'{a.out}/{sc}__s{seed}{a.gt_from}.npz' if a.gt_from else ''
            REF = dict(np.load(ref)) if ref and os.path.exists(ref) else None
            while (fov0 < int(REF['n_fov'])) if REF is not None else (need > 0 and it < 400):
                it += 1
                o = GT.gen(sc, it, seed)
                nb = o['nb']
                G = o['gt']; act = G['onf'].sum(1) > 0
                G = {k_: v[act] for k_, v in G.items()}
                Gx = G['x'] + o['shift_est'][G['fov']][:, 0]; Gy = G['y'] + o['shift_est'][G['fov']][:, 1]
                ph = G['phot'].sum(1); on = G['onf'].sum(1)
                inter = (G['x'] >= 9) & (G['x'] <= S - 10) & (G['y'] >= 9) & (G['y'] <= S - 10)
                vis = (on >= 1) & (ph >= 100)
                tg = torch.nonzero(inter & vis)[:, 0]
                if len(tg) == 0: continue
                if REF is not None:
                    torch.cuda.synchronize(); t1 = time.time()
                    r_ = scode_front(m, det, o)
                    torch.cuda.synchronize(); tnet += time.time() - t1
                    if r_ is not None:
                        for L_, rr in zip(OUTS, r_):
                            rr = dict(rr); rr['b'] = rr['b'] + fov0
                            L_.append({k_: v.detach().float().cpu().numpy() for k_, v in rr.items()})
                    fov0 += nb; need -= len(tg); torch.cuda.empty_cache(); continue
                nn = ST.nn_dist(G, tg)
                fovs = torch.unique(G['fov'][tg]); ai = torch.nonzero(torch.isin(G['fov'], fovs))[:, 0]
                psS = torch.zeros(len(G['x']), CD.psf_channels(rpsf), CD.R, CD.R, device=dev)
                psS[ai] = CD.psf_stack(rpsf, G['x'][ai], G['y'][ai], G['z'][ai], sim.focus)
                Cb = CD.crlb_all(G, o, psS, tg, T, sim.focus); del psS
                n = len(G['x'])
                full = lambda v, w=2: torch.full((n, w), float('nan'), device=dev).index_copy_(0, tg, v.float())
                nnf = torch.full((n,), float('nan'), device=dev).index_copy_(0, tg, nn.float())
                GTL.append({k_: v.detach().float().cpu().numpy() for k_, v in dict(
                    fov=(G['fov'] + fov0).float(), x=Gx, y=Gy, phot=ph, onfr=on, z=G['z'], target=(inter & vis).float(),
                    dontcare=(inter & ~vis).float(), nn=nnf, sf_m=full(Cb['sf_m']), sf_i=full(Cb['sf_i']), sm_m=full(Cb['sm_m']),
                    sm_i=full(Cb['sm_i']), I=G['phot'], fr_m=full(Cb['fr_m'], T)).items()})
                torch.cuda.synchronize(); t1 = time.time()
                r_ = scode_front(m, det, o)
                torch.cuda.synchronize(); tnet += time.time() - t1
                if r_ is not None:
                    for L_, rr in zip(OUTS, r_):
                        rr = dict(rr); rr['b'] = rr['b'] + fov0
                        L_.append({k_: v.detach().float().cpu().numpy() for k_, v in rr.items()})
                fov0 += nb; need -= len(tg)
                torch.cuda.empty_cache()
            cat = lambda L: {k_: np.concatenate([r[k_] for r in L]) for k_ in L[0]} if L else {}
            if REF is not None:
                D = {k_: v for k_, v in REF.items() if k_.startswith('gt_')}; assert int(REF['n_fov']) == fov0
            else:
                D = {'gt_' + k_: v for k_, v in cat(GTL).items()}
            D['n_fov'] = np.array(fov0)
            for tag, L_ in zip(tags, OUTS):
                E = dict(D); E.update({f'sc_{k_}': v for k_, v in cat(L_).items()})
                np.savez_compressed(f'{a.out}/{sc}__s{seed}{tag}.npz', **E)
            # coordinate convention check (net vs GT, matched within 0.5 px, targets)
            from scipy.spatial import cKDTree
            E = cat(OUTS[1]); mt = D['gt_target'] > 0
            dd = []
            for f_ in np.unique(D['gt_fov'][mt])[:200]:
                gi = np.nonzero(mt & (D['gt_fov'] == f_))[0]; oi = np.nonzero(E['b'] == f_)[0]
                if len(oi) == 0: continue
                d_, j_ = cKDTree(np.c_[E['x_det'][oi], E['y_det'][oi]]).query(np.c_[D['gt_x'][gi], D['gt_y'][gi]])
                k_ = d_ < 0.5; dd.append(np.c_[E['x_det'][oi[j_[k_]]] - D['gt_x'][gi[k_]], E['y_det'][oi[j_[k_]]] - D['gt_y'][gi[k_]]])
            dd = np.concatenate(dd) if dd else np.zeros((0, 2))
            print(f'{sc} seed {seed}: {int(D["gt_target"].sum())} targets, {fov0} FOVs, {time.time() - t0:.0f}s (simcode front {tnet:.1f}s); '
                  f'net - GT median offset {np.median(dd, 0) if len(dd) else "-"} px (n {len(dd)})', flush=True)


if __name__ == '__main__':
    main()
