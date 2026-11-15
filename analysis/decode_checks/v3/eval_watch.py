"""Progress evaluation of a DECODE training run (train_v3.py): every snapshot (<model dir>/snapshots/model_e<epoch>.pt) on
  iso  : the fixed isolated-emitter movie out/decode_checks/iso_v2psf (all 6000 windows with start % 6 == 0; output frame =
         window start, dec_common.process_padded): recall and RMS / CRLB per detected-photon level (per-emitter 6-sum SMLM CRLB,
         z free: out/decode_checks/iso_crlb.npz, as iso_eval.py), mean x / y error, GT pixel bias A_bias x / y (pixlock.py logic:
         analysis/pixbias/sim.py gt_rows + bias_curve; A > 0 = pull to the pixel centre)
  dens : one density movie (default D2_v2psf, first 12 000 frames, windows start % 6 == 0): |c1| x / y of all rows (first
         harmonic of the sub-pixel histogram, pblib.ustats), A_bias x / y, RMS of the isolated GT-matched rows, recall of GT
         emitters with >= 100 window photons and the false-positive fraction (rows without a GT emitter within 1 px; one-to-one
         greedy nearest matching per window)
-> <model dir>/eval_log.csv (one row per snapshot) and out/decode_checks/v3/fig_progress_<name>.png (+ the reference lines of
dec_sim1809v2's model_final.pt, evaluated once: out/decode_checks/v3/eval_ref_dec_sim1809v2.json).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/decode_checks/v3/eval_watch.py --names dec_sim1809v3[,other] [--poll 300] [--once]"""
import os, sys, json, time, argparse, glob, csv, re
os.environ.setdefault('TQDM_DISABLE', '1')
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAIN = os.path.expanduser('~/simcode2')
sys.path[:0] = [os.path.join(REPO, 'analysis', 'baselines', 'decode'), os.path.join(REPO, 'analysis', 'pixbias'),
                os.path.join(REPO, 'analysis', 'papersim'), REPO]
import decode_paper as DP
D = DP.D
import pblib as B, sim as PS, metrics as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = os.path.join(MAIN, 'out', 'decode_checks', 'v3')
ISO = os.path.join(MAIN, 'out', 'decode_checks', 'iso_v2psf')
LEVELS = (200, 350, 500, 750, 1200)
MDIR = os.path.join(MAIN, 'out', 'paper2', 'decode', 'lutpsf')


def movie_sums(md, nframes):
    from smlmtorch.util.multipart_tiff import tiff_read_file
    cam = np.load(os.path.join(md, 'camera_sim.npz')); gain = float(cam['gain']); off = cam['offset_adu'].astype(np.float32)
    fr = np.asarray(list(tiff_read_file(os.path.join(md, 'movie.tif'), startframe=0, maxframes=nframes, use_progbar=False)), np.float32)
    n = len(fr) // 6
    return fr[:n * 6].reshape(n, 6, *fr.shape[1:]).sum(1), gain, off * 6


def infer(inst, S):
    sums, gain, off = S
    em, _ = D.process_padded(inst, iter(list(sums)), gain=gain, offset=off, sumframes=1, batch_size=64, infer_batch_size=16)
    if len(em) < 50:      # (an early snapshot that finds nothing: one dummy row far outside, so every metric is defined / nan)
        return dict(f=np.zeros(1, np.int64), x=np.full(1, -100.0), y=np.full(1, -100.0), photons=np.ones(1))
    em = em[em.frame_ix < len(sums)]
    return dict(f=6 * em.frame_ix.numpy().astype(np.int64), x=em.xyz[:, 0].numpy().astype(float), y=em.xyz[:, 1].numpy().astype(float),
                photons=em.phot.numpy().astype(float))


def bias(T, gt, nf):
    tx, ty, _ = PS.gt_rows(gt, nf, T['f'], T['x'], T['y'])
    g = np.isfinite(tx)
    bx = PS.bias_curve(T['x'][g] - tx[g], B.sub(tx[g])); by = PS.bias_curve(T['y'][g] - ty[g], B.sub(ty[g]))
    return bx, by, g, tx, ty


def match(T, gt):
    """per GT event: index of the nearest row in its window within 1 px (-1 if none) (= iso_eval.match)"""
    f0 = gt['ev_t0'].astype(np.int64); gx, gy = gt['ev_x'], gt['ev_y']
    j = np.full(len(gx), -1)
    o = np.argsort(T['f'], kind='stable'); fs = T['f'][o]
    for w in np.unique(f0):
        e = np.nonzero(f0 == w)[0]; a, b = np.searchsorted(fs, [w, w + 1]); r = o[a:b]
        if not len(r): continue
        d = np.hypot(T['x'][r][None] - gx[e, None], T['y'][r][None] - gy[e, None]); k = d.argmin(1); ok = d[np.arange(len(e)), k] < 1.0
        j[e[ok]] = r[k[ok]]
    return j


def eval_iso(T, gt, crlb, nf):
    j = match(T, gt)
    lev = gt['ev_I6']; r = {}
    ok = j >= 0
    ex = np.where(ok, T['x'][np.maximum(j, 0)] - gt['ev_x'], np.nan) * B.PX; ey = np.where(ok, T['y'][np.maximum(j, 0)] - gt['ev_y'], np.nan) * B.PX
    for L in LEVELS:
        m = (lev == L); k = m & ok
        r[f'iso_recall_{L}'] = float(k.sum() / max(m.sum(), 1))
        r[f'iso_rmsc_{L}'] = r[f'iso_rms_{L}'] = np.nan
        if k.sum() > 20:
            e = np.r_[ex[k], ey[k]]; r[f'iso_rms_{L}'] = float(np.sqrt(np.mean(e ** 2)))
            if crlb is not None: c = np.r_[crlb[k, 0], crlb[k, 1]]; r[f'iso_rmsc_{L}'] = r[f'iso_rms_{L}'] / float(np.sqrt(np.mean(c ** 2)))
    r['iso_recall'] = float(ok.mean())
    e = np.r_[ex[ok], ey[ok]]
    if not ok.any(): ok = ok.copy()
    r['iso_rmsc'] = np.nan if not ok.any() else float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(np.r_[crlb[ok, 0], crlb[ok, 1]] ** 2))) if crlb is not None else np.nan
    r['iso_mean_x_nm'] = float(np.mean(ex[ok])) if ok.any() else np.nan; r['iso_mean_y_nm'] = float(np.mean(ey[ok])) if ok.any() else np.nan
    try: bx, by, g, _, _ = bias(T, gt, nf)
    except Exception: bx = by = None
    r['iso_Abias_x_nm'] = bx['A_nm'] if bx else np.nan; r['iso_Abias_y_nm'] = by['A_nm'] if by else np.nan
    r['iso_Abias_x_se'] = bx['A_se_nm'] if bx else np.nan
    return r


def eval_dens(T, gt, nf, W=40):
    m = (T['x'] >= 0.5) & (T['x'] < W - 0.5) & (T['y'] >= 0.5) & (T['y'] < W - 0.5) & (T['f'] + 6 <= nf)
    T = {k: v[m] for k, v in T.items()}
    r = dict(dens_n=int(len(T['x'])))
    ux, uy = B.ustats(B.sub(T['x'])), B.ustats(B.sub(T['y']))
    r['dens_c1_x'] = ux.get('c1_abs', np.nan); r['dens_c1_y'] = uy.get('c1_abs', np.nan); r['dens_c1_floor'] = ux.get('noise', np.nan)
    try: bx, by, g, tx, ty = bias(T, gt, nf)
    except Exception: bx = by = None; g = np.zeros(len(T['x']), bool); tx = ty = np.full(len(T['x']), np.nan)
    r['dens_Abias_x_nm'] = bx['A_nm'] if bx else np.nan; r['dens_Abias_y_nm'] = by['A_nm'] if by else np.nan
    r['dens_Abias_x_se'] = bx['A_se_nm'] if bx else np.nan
    r['dens_rms_iso_nm'] = float(np.sqrt(np.mean(np.r_[T['x'][g] - tx[g], T['y'][g] - ty[g]] ** 2)) * B.PX) if g.sum() else np.nan
    # recall / false positives: one-to-one greedy nearest matching within 1 px per window
    ws = np.arange(0, nf - 5, 6); GW = M.gt_windows(gt, nf, ws)
    o = np.argsort(T['f'], kind='stable'); fs = T['f'][o]
    n_gt = n_tp = 0; matched_row = np.zeros(len(T['x']), bool)
    for w in ws:
        gx, gy, gph = GW[w]; s = gph >= 100; gx, gy = gx[s], gy[s]
        a, b = np.searchsorted(fs, [w, w + 1]); rr = o[a:b]
        n_gt += len(gx)
        if not len(gx) or not len(rr): continue
        d = np.hypot(T['x'][rr][:, None] - gx[None], T['y'][rr][:, None] - gy[None])
        ii, jj = np.nonzero(d < 1.0); od = np.argsort(d[ii, jj])
        ur = set(); ug = set()
        for q in od:
            if ii[q] in ur or jj[q] in ug: continue
            ur.add(ii[q]); ug.add(jj[q])
        n_tp += len(ug); matched_row[rr[list(ur)]] = True
    # rows near any GT emitter (also < 100 photons) are not false positives
    fp = 0
    for w in ws:
        a, b = np.searchsorted(fs, [w, w + 1]); rr = o[a:b]
        if not len(rr): continue
        gx, gy, _ = GW[w]
        um = rr[~matched_row[rr]]
        if not len(um): continue
        if not len(gx): fp += len(um); continue
        d = np.hypot(T['x'][um][:, None] - gx[None], T['y'][um][:, None] - gy[None]).min(1)
        fp += int((d >= 1.0).sum())
    r['dens_recall'] = n_tp / max(n_gt, 1); r['dens_fp_frac'] = fp / max(len(T['x']), 1)
    return r


def eval_real(T, crop, tag):
    """|c1| x / y of the sub-pixel histogram (all rows inside the SIMCODE crop, 0.5 px margin) and of the dimmest photon tercile"""
    x0, y0, sz = [int(v) for v in crop]
    m = (T['x'] >= x0 + 0.5) & (T['x'] < x0 + sz - 0.5) & (T['y'] >= y0 + 0.5) & (T['y'] < y0 + sz - 0.5)
    x, y, ph = T['x'][m], T['y'][m], T['photons'][m]
    r = {f'{tag}_n': int(m.sum())}
    r[f'{tag}_c1_x'] = B.ustats(B.sub(x)).get('c1_abs', np.nan); r[f'{tag}_c1_y'] = B.ustats(B.sub(y)).get('c1_abs', np.nan)
    if len(x) < 30: return {**r, f'{tag}_c1_x_low': np.nan, f'{tag}_c1_y_low': np.nan, f'{tag}_c1_floor': np.nan}
    lo = ph < np.quantile(ph, 1 / 3)
    r[f'{tag}_c1_x_low'] = B.ustats(B.sub(x[lo])).get('c1_abs', np.nan); r[f'{tag}_c1_y_low'] = B.ustats(B.sub(y[lo])).get('c1_abs', np.nan)
    r[f'{tag}_c1_floor'] = 1 / np.sqrt(max(m.sum(), 1))
    return r


def real_ref(ds, Z):
    """the same |c1| numbers from the existing DECODE table of the data set (out/paper2/<ds>/decode, old model), same frames"""
    from simcode.dataset import decode_table
    z = np.load(decode_table(os.path.join(MAIN, 'out', 'paper2', ds, 'decode', 'locs.npz')))
    f0 = int(Z['frame0']); n = len(Z['sums'])
    m = (z['frame'] >= f0) & (z['frame'] < f0 + 6 * n) & ((z['frame'] - f0) % 6 == 0)
    return eval_real(dict(x=z['x'][m].astype(float), y=z['y'][m].astype(float), photons=z['phot'][m].astype(float)), Z['crop'], 'real_' + ds)


class Evaluator:
    """one inference instance per model name (each with its own param: input / output scaling), the cached test inputs"""
    def __init__(self, dens, dens_frames, sims=True, real=(), iso=ISO):
        D.SPD = os.path.join(MAIN, 'out', 'paper2', 'decode'); D.LUTPSF = f'{D.SPD}/lutpsf.txt'
        self.insts = {}; self.sims = sims; self.real = {}
        if sims:
            self.iso_gt = dict(np.load(os.path.join(iso, 'gt.npz'))); self.iso_nf = int(self.iso_gt['nframes'])
            # per-emitter CRLB of the default iso movie only (iso_eval.py); other iso movies: RMS in nm only
            self.crlb = np.load(os.path.join(MAIN, 'out', 'decode_checks', 'iso_crlb.npz'))['smlm'] if os.path.abspath(iso) == ISO else None
            self.S_iso = movie_sums(iso, self.iso_nf)
            self.dens = dens
            if dens:
                dm = os.path.join(MAIN, 'out', 'papersim', dens)
                self.dens_gt = dict(np.load(os.path.join(dm, 'gt.npz'))); self.dens_nf = dens_frames
                self.S_dens = movie_sums(dm, dens_frames)
        for ds in real:
            Z = dict(np.load(os.path.join(OUT, 'cache', f'real_{ds}.npz')))
            self.real[ds] = Z

    def inst(self, name):
        if name not in self.insts:
            pj = json.load(open(f'{MDIR}/{name}/psf.json'))
            DP.setup_psf(pj['psf'], pj['halo'], pj.get('zmax_um', DP.ZMAX))
            # weights are replaced per snapshot; any state dict of the same architecture serves for the set-up
            self.insts[name] = D.load_trained(name, pj.get('variant_name', 'sum6p'), model_file=f'{MDIR}/dec_sim1809v2/model_final.pt')
        return self.insts[name]

    def __call__(self, name, pt):
        inst = self.inst(name)
        sd = torch.load(pt, map_location=inst.param.Hardware.device, weights_only=True)
        inst.model.load_state_dict(sd); inst.model.eval()
        t0 = time.time(); r = {}
        if self.sims:
            r.update(eval_iso(infer(inst, self.S_iso), self.iso_gt, self.crlb, self.iso_nf))
            if self.dens: r.update(eval_dens(infer(inst, self.S_dens), self.dens_gt, self.dens_nf))
        for ds, Z in self.real.items():
            r.update(eval_real(infer(inst, (Z['sums'], float(Z['gain']), Z['offset6'])), Z['crop'], 'real_' + ds))
        r['eval_s'] = round(time.time() - t0, 1)
        return r


def read_csv(fn):
    if not os.path.exists(fn): return []
    with open(fn) as f: return list(csv.DictReader(f))


def plot(names, ref, fn):
    fig, axs = plt.subplots(2, 5, figsize=(21, 7.5))
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ni, nm in enumerate(names):
        E = read_csv(f'{MDIR}/{nm}/eval_log.csv'); L = read_csv(f'{MDIR}/{nm}/train_log.csv')
        c = cols[ni % len(cols)]
        if L:
            ep = np.array([float(r['epoch']) for r in L])
            axs[0, 0].plot(ep, [float(r['val_loss']) for r in L], color=c, lw=1, label=f'{nm} val')
            axs[0, 0].plot(ep, [float(r['train_loss']) for r in L], color=c, lw=0.5, alpha=0.4)
            ax2 = axs[1, 0]; ax2.semilogy(ep, [float(r['lr']) for r in L], color=c, label=nm)
        if not E: continue
        e = np.array([float(r['epoch']) for r in E]); g = lambda k: np.array([float(r.get(k) or np.nan) for r in E])
        for k in [k for k in E[-1] if k.startswith('real_') and k.endswith('_c1_x')]:
            ds = k[5:-5]
            axs[0, 4].plot(e, g(k), 'o-', ms=3, color=c, label=f'{nm} {ds} x'); axs[0, 4].plot(e, g(f'real_{ds}_c1_y'), 's--', ms=3, color=c, label='y')
            axs[1, 4].plot(e, g(f'real_{ds}_c1_x_low'), 'o-', ms=3, color=c, label=f'{nm} {ds} x'); axs[1, 4].plot(e, g(f'real_{ds}_c1_y_low'), 's--', ms=3, color=c, label='y')
        if 'iso_Abias_x_nm' not in E[-1]: continue
        axs[0, 1].plot(e, g('iso_Abias_x_nm'), 'o-', ms=3, color=c, label=f'{nm} x'); axs[0, 1].plot(e, g('iso_Abias_y_nm'), 's--', ms=3, color=c, label='y')
        axs[0, 2].plot(e, g('dens_Abias_x_nm'), 'o-', ms=3, color=c, label=f'{nm} x'); axs[0, 2].plot(e, g('dens_Abias_y_nm'), 's--', ms=3, color=c, label='y')
        axs[0, 3].plot(e, g('dens_c1_x'), 'o-', ms=3, color=c, label=f'{nm} x'); axs[0, 3].plot(e, g('dens_c1_y'), 's--', ms=3, color=c, label='y')
        rk = 'iso_rmsc_' if np.isfinite(g('iso_rmsc_500')).any() else 'iso_rms_'      # RMS / CRLB, or RMS in nm (no CRLB file)
        for li, Lv in enumerate(LEVELS):
            axs[1, 1].plot(e, g(f'{rk}{Lv}'), '-', color=c, alpha=0.35 + 0.13 * li, label=f'{nm} {Lv} ph')
        axs[1, 2].plot(e, g('iso_recall'), 'o-', ms=3, color=c, label=f'{nm} iso'); axs[1, 2].plot(e, g('dens_recall'), 's--', ms=3, color=c, label='D2 dens')
        axs[1, 2].plot(e, g('dens_fp_frac'), ':', color=c, label='D2 false-pos. frac')
        axs[1, 3].plot(e, g('iso_mean_x_nm'), 'o-', ms=3, color=c, label=f'{nm} x'); axs[1, 3].plot(e, g('iso_mean_y_nm'), 's--', ms=3, color=c, label='y')
    for k, ax, lab in (('iso_Abias_x_nm', axs[0, 1], 'dec_sim1809v2 x'), ('dens_Abias_x_nm', axs[0, 2], 'dec_sim1809v2 x'), ('dens_c1_x', axs[0, 3], 'dec_sim1809v2 x'),
                       ('iso_mean_x_nm', axs[1, 3], 'dec_sim1809v2 x'), ('iso_recall', axs[1, 2], 'dec_sim1809v2 iso'), ('iso_rmsc_500', axs[1, 1], 'dec_sim1809v2 500 ph')):
        if k in ref: ax.axhline(ref[k], color='k', lw=0.8, ls='-.', label=lab)
    for k in [k for k in ref if k.startswith('real_') and k.endswith('_c1_x')]:
        axs[0, 4].axhline(ref[k], color='k', lw=0.8, ls='-.', label=f'old {k[5:-5]} model x')
        axs[1, 4].axhline(ref[k + '_low'], color='k', lw=0.8, ls='-.', label=f'old {k[5:-5]} model x')
    for ax in (axs[0, 1], axs[0, 2], axs[1, 3]): ax.axhline(0, color='0.5', lw=0.5)
    axs[1, 1].axhline(1, color='0.5', lw=0.5)
    T = ['loss (val; thin = train)', 'iso: A_bias (nm)', 'D2 (12k frames): A_bias (nm)', 'D2: |c1| sub-pixel histogram',
         'real data: |c1| (all rows)', 'learning rate', 'iso: RMS / CRLB per photon level', 'recall / false-pos. frac', 'iso: mean error (nm)',
         'real data: |c1| (dimmest photon tercile)']
    for ax, t in zip(axs.flat, T):
        ax.set_title(t, fontsize=9); ax.set_xlabel('epoch', fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6)
    axs[0, 0].set_yscale('log')
    fig.tight_layout(); fig.savefig(fn, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', required=True); ap.add_argument('--dens', default='D2_v2psf'); ap.add_argument('--dens_frames', type=int, default=12000)
    ap.add_argument('--sims', type=int, default=1, help='0: no iso / density sims')
    ap.add_argument('--iso', default=ISO, help='isolated-emitter movie dir (model 2: out/decode_checks/v3/iso_m25 = its PSF)')
    ap.add_argument('--real', default='', help='comma list of cached real chunks (cache_real.py), e.g. 1809_2 or 2.5nM_2,20nM_2')
    ap.add_argument('--tag', default=None, help='figure name suffix (default: the names)')
    ap.add_argument('--poll', type=int, default=300); ap.add_argument('--once', action='store_true')
    ap.add_argument('--every', type=int, default=1, help='evaluate every k-th snapshot only')
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('decode-v3-eval')
    names = a.names.split(','); os.makedirs(OUT, exist_ok=True)
    real = [d for d in a.real.split(',') if d]
    ev = Evaluator(a.dens, a.dens_frames, sims=bool(a.sims), real=real, iso=os.path.abspath(a.iso))
    ref = {}
    if a.sims and os.path.abspath(a.iso) == ISO:
        rf = os.path.join(OUT, 'eval_ref_dec_sim1809v2.json')
        if not os.path.exists(rf):
            json.dump(ev('dec_sim1809v2', f'{MDIR}/dec_sim1809v2/model_final.pt'), open(rf, 'w'), indent=1)
        ref.update({k: v for k, v in json.load(open(rf)).items() if not k.startswith('real_')})
    for ds in real:
        ref.update(real_ref(ds, ev.real[ds]))
    json.dump(ref, open(os.path.join(OUT, f'eval_ref_{a.tag or "_".join(names)}.json'), 'w'), indent=1)
    print('ref', ref, flush=True)
    figfn = os.path.join(OUT, 'fig_progress_' + (a.tag or '_'.join(names)) + '.png')
    while True:
        new = 0
        for nm in names:
            fn = f'{MDIR}/{nm}/eval_log.csv'
            done = {int(r['epoch']) for r in read_csv(fn)}
            snaps = sorted(glob.glob(f'{MDIR}/{nm}/snapshots/model_e*.pt'))
            for k, sp in enumerate(snaps):
                e = int(re.findall(r'model_e(\d+)\.pt', sp)[0])
                if e in done or (k % a.every and k != len(snaps) - 1): continue
                try: r = ev(nm, sp)
                except Exception as ex: print('eval failed', sp, repr(ex), flush=True); continue
                row = dict(epoch=e, **r)
                hdr = not os.path.exists(fn)
                with open(fn, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row)); (w.writeheader() if hdr else None); w.writerow(row)
                print(nm, 'epoch', e, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
                new += 1
        if new:
            try: plot(names, ref, figfn)
            except Exception as ex: print('plot failed', repr(ex), flush=True)
        fin = all(os.path.exists(f'{MDIR}/{nm}/model_final.pt') for nm in names)
        if a.once or (fin and not new): break
        time.sleep(a.poll)


if __name__ == '__main__':
    main()
