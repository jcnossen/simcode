"""SI figure 'si_pixel_bias': sub-pixel histograms and the GT pixel bias (pull toward the pixel centre, nm) vs photons and vs
density for SIMCODE, DECODE (our model trained at DECODE's published scale), DECODE's published model in its own conditions,
SMLM and SIMFLUX as references, and the ground-truth floor.

Rows of every method = the paper's common rule (analysis/papersim/common_rule.py; metrics.frc_set): each method's own filters,
disjoint 6-frame windows, inside the field, max(sigma_x, sigma_y) < 0.15 px (16.25 nm) on its own reported uncertainty
(SMLM / SIMFLUX CRLB, DECODE predicted sigma, SIMCODE network sigma (locs_p2_net) with p >= 0.63 = metrics 'v2c'); the sub-pixel statistics
use the rows inside [0.5, W - 0.5) (analysis/decode_checks/pixlock.py logic).
  |c1|    first harmonic of the sub-pixel histogram of f = x - round(x) (pixel centre = 0), analysis/pixbias/pblib.ustats
  A_bias  GT bias: e = est - GT of rows with an isolated GT emitter (>= 100 window photons, no other ON emitter within 2 px) within
          1 px, fitted e = -A sin(2 pi f_GT) + ... (analysis/pixbias/sim.py bias_curve); A > 0 = pull toward the pixel centre
Movies (out/papersim, PSF v2, camera v2): iso = out/decode_checks/iso_v2psf (isolated emitters, 200 / 350 / 500 / 750 / 1200
detected photons per 6-frame window; SIMCODE and SIMFLUX use the TRUE pattern there: their own pattern estimates fail on this
sparse movie), density D0.5 / D1 / D2 / D4 _v2psf (500 detected photons; SIMFLUX = true pattern, SIMCODE = own estimate).
Methods: SMLM base/smlm, SIMFLUX base/sf_gtp, DECODE <run>/decode/<--decode> (default locs_v3.npz = dec_m1809v3), SIMCODE
v2_<--model>/locs_p2 (m1809v2 = production m1809) with the common rule. DECODE (published model): DECODE 0.10.2 with its
published model and its own simulator / conditions (out/decode_checks/published_model/published_model.json; astigmatic PSF,
EMCCD, 7000 +- 3000 photons training, 127 x 117 nm pixels; isolated / 10 / 50 emitters per 40 x 40 px frame).
usage: python analysis/paper/si_pixel_bias.py [--decode locs_v3.npz] [--model m1809v2] [--outdir out/paper_figs_v2] [--redo 0]
outputs: <outdir>/si_pixel_bias.{pdf,svg,png}, _caption.txt, _values.json (cache of the per-movie numbers: <outdir>/si_pixel_bias_cache.json)"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE, os.path.join(REPO, 'analysis', 'papersim'), os.path.join(REPO, 'analysis', 'pixbias')):
    sys.path.insert(0, p)
import figstyle as FS
import common as C, metrics as M
import pblib as B, sim as PS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = os.path.join(REPO, 'out', 'paper_figs_v2')
PSD = os.path.join(REPO, 'out', 'papersim')
ISO = os.path.join(REPO, 'out', 'decode_checks', 'iso_v2psf')
PUB = os.path.join(REPO, 'out', 'decode_checks', 'published_model', 'published_model.json')
DENS = ['D0.5', 'D1', 'D2', 'D4']
LEVELS = (200, 350, 500, 750, 1200)
METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
HIST_MOVIE = 'D1'


def method_tables(run, model, decode_name):
    P = M.method_paths(run, model)
    return dict(smlm=('smlm', P.get('smlm')), sf=('sf_gtp', P.get('sf_gtp')), decode=('decode', os.path.join(run, 'decode', decode_name)),
                v2=('v2c', P.get('v2c')))


def common_rows(kind, path, W, nf, cut=True):
    """cut=False: the same rows without the common precision cut (own filters, SIMCODE p >= 0.63, disjoint windows, inside)"""
    s0 = M.SIGMA_MAX
    if not cut: M.SIGMA_MAX = np.inf
    try:
        T = M.load_method(kind, path)
        _, _, f = M.frc_set(T, W, nf)
    finally:
        M.SIGMA_MAX = s0
    f &= (T['x'] >= 0.5) & (T['x'] < W - 0.5) & (T['y'] >= 0.5) & (T['y'] < W - 0.5)
    return {k: v[f] for k, v in T.items()}


def stats(T, gt, nf, levels=None):
    x, y, f = T['x'], T['y'], T['f']
    ux, uy = B.ustats(B.sub(x)), B.ustats(B.sub(y))
    hx = np.histogram(B.sub(x), 20, (-0.5, 0.5))[0]; hy = np.histogram(B.sub(y), 20, (-0.5, 0.5))[0]
    r = dict(n=int(len(x)), c1_x=ux.get('c1_abs'), c1_y=uy.get('c1_abs'), noise=ux.get('noise'),
             hist_x=(hx / max(hx.mean(), 1e-9)).tolist(), hist_y=(hy / max(hy.mean(), 1e-9)).tolist())
    tx, ty, tph = PS.gt_rows(gt, nf, f, x, y); g = np.isfinite(tx)
    def bias(m):
        bx, by = PS.bias_curve(x[m] - tx[m], B.sub(tx[m])), PS.bias_curve(y[m] - ty[m], B.sub(ty[m]))
        return dict(n=int(m.sum()), A_x=bx['A_nm'] if bx else None, A_x_se=bx['A_se_nm'] if bx else None,
                    A_y=by['A_nm'] if by else None, A_y_se=by['A_se_nm'] if by else None)
    r['bias'] = bias(g)
    if levels:
        L = np.array(levels, float); li = np.abs(tph[:, None] - L[None]).argmin(1)
        r['bias_ph'] = {str(int(L[i])): bias(g & (li == i)) for i in range(len(L))}
    return r


def gt_density(gt, nf, W):
    """mean number of GT emitters ON per disjoint 6-frame window per um^2"""
    ws = np.arange(0, nf - 5, 6)[::10]; GW = M.gt_windows(gt, nf, ws)
    return float(np.mean([len(GW[w][0]) for w in ws]) / (W * C.PX_NM / 1000) ** 2)


def compute(a):
    V = dict(movies={})
    for mv in ['iso'] + DENS:
        md = ISO if mv == 'iso' else os.path.join(PSD, mv + '_v2psf'); run = os.path.join(md, 'full')
        gt = dict(np.load(os.path.join(md, 'gt.npz'))); g = json.load(open(os.path.join(md, 'gen.json')))
        W, nf = int(g['W']), int(g['nframes'])
        R = dict(movie=md, W=W, nframes=nf, density_um2=gt_density(gt, nf, W), methods={}, paths={})
        ge = np.histogram(B.sub(gt['ev_x']), 20, (-0.5, 0.5))[0]
        R['gt'] = dict(c1_x=B.ustats(B.sub(gt['ev_x']))['c1_abs'], c1_y=B.ustats(B.sub(gt['ev_y']))['c1_abs'], hist_x=(ge / ge.mean()).tolist())
        for k, (kind, path) in method_tables(run, a.model, a.decode).items():
            if not path or not os.path.exists(path): print('missing', mv, k, path, flush=True); continue
            T = common_rows(kind, path, W, nf)
            R['methods'][k] = stats(T, gt, nf, LEVELS if mv == 'iso' else None); R['paths'][k] = path
            if mv == 'iso':      # isolated emitters vs photons: also without the precision cut (it removes the dim levels of SMLM / DECODE)
                R.setdefault('methods_nocut', {})[k] = stats(common_rows(kind, path, W, nf, cut=False), gt, nf, LEVELS)
            b = R['methods'][k]['bias']
            print(mv, k, R['methods'][k]['n'], '|c1| %.3f %.3f  A %.2f %.2f' % (R['methods'][k]['c1_x'] or np.nan, R['methods'][k]['c1_y'] or np.nan,
                  b['A_x'] or np.nan, b['A_y'] or np.nan), flush=True)
        V['movies'][mv] = R
    P = json.load(open(PUB))
    pub = dict(px_nm=P['info']['px_size_nm'], training=P['info']['training'], sets={})
    area = 40 * P['info']['px_size_nm'][0] / 1000 * 40 * P['info']['px_size_nm'][1] / 1000
    for s, per_frame in (('iso', 4), ('mod', 10), ('train', 50)):
        Q = P[s]
        pub['sets'][s] = dict(emitters_per_frame=per_frame, density_um2=per_frame / area, n=Q['n'], c1_x=Q['x']['c1_abs'], c1_y=Q['y']['c1_abs'],
                              hist_x=Q['x']['hist'], hist_y=Q['y']['hist'],
                              A_x=Q['bias_x']['A_nm'], A_x_se=Q['bias_x']['A_se_nm'], A_y=Q['bias_y']['A_nm'], A_y_se=Q['bias_y']['A_se_nm'])
    pub['iso_ph'] = {ph: dict(A_x=P['iso']['bias_x_gtph'][ph]['A_nm'], A_x_se=P['iso']['bias_x_gtph'][ph]['A_se_nm'],
                              A_y=P['iso']['bias_y_gtph'][ph]['A_nm'], A_y_se=P['iso']['bias_y_gtph'][ph]['A_se_nm'],
                              crlb_emccd_nm=P['iso']['precision'][ph]['crlb_emccd_nm'], recall=P['iso']['precision'][ph]['recall'])
                     for ph in P['iso']['bias_x_gtph']}
    V['published'] = pub
    return V


def draw(V, a):
    FS.apply()
    fig = plt.figure(figsize=(FS.FIG_W, FS.FIG_W * 0.62))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.15], hspace=0.42)
    top = gs[0].subgridspec(1, 6, wspace=0.12); bot = gs[1].subgridspec(1, 4, wspace=0.5)
    xs = np.linspace(-0.475, 0.475, 20)
    H = V['movies'][HIST_MOVIE]
    cols = [('gt', 'Ground truth', 'k')] + [(k, lab, FS.COL['v2' if k == 'v2' else k]) for k, lab in METH] + [('pub', 'DECODE (published model)', 'tab:orange')]
    ax0 = None
    for i, (k, lab, c) in enumerate(cols):
        ax = fig.add_subplot(top[0, i], sharey=ax0); ax0 = ax0 or ax
        if k == 'gt': hx, hy = H['gt']['hist_x'], None
        elif k == 'pub': S = V['published']['sets']['mod']; hx, hy = S['hist_x'], S['hist_y']
        else:
            if k not in H['methods']: continue
            hx, hy = H['methods'][k]['hist_x'], H['methods'][k]['hist_y']
        ax.step(xs, hx, where='mid', color=c, lw=0.9, label='x')
        if hy is not None: ax.step(xs, hy, where='mid', color=c, lw=0.9, ls=(0, (2, 1)), alpha=0.8, label='y')
        ax.axhline(1, color='0.6', lw=0.4)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([-0.4, 0, 0.4]); ax.set_xticklabels(['-0.4', '0', '0.4'])
        ax.set_title(lab, fontsize=FS.FS_TICK + 0.5, color='k')
        if k == 'pub': ax.text(0.5, 0.03, 'own conditions,\n10 / frame', transform=ax.transAxes, ha='center', va='bottom', fontsize=FS.FS_SMALL)
        else:
            q = H['gt'] if k == 'gt' else H['methods'][k]
            ax.text(0.5, 0.03, '|c$_1$| %.3f' % q['c1_x'] + ('' if k == 'gt' else ' / %.3f' % q['c1_y']), transform=ax.transAxes, ha='center', va='bottom', fontsize=FS.FS_SMALL)
        if i == 0:
            ax.set_ylabel('relative density'); FS.letter(ax, 'a')
        else: plt.setp(ax.get_yticklabels(), visible=False)
        if i == 2: ax.set_xlabel(f'sub-pixel position (px; 0 = pixel centre), {HIST_MOVIE} movie; solid x, dashed y', labelpad=2)
    ax0.set_ylim(0.4, 1.9)
    # b: A_bias vs photons (iso)
    ax = fig.add_subplot(bot[0, 0]); FS.letter(ax, 'b', dx=-0.22)
    I = V['movies']['iso']['methods_nocut']
    for j, (k, lab) in enumerate(METH):
        if k not in I: continue
        c = FS.COL['v2' if k == 'v2' else k]; L = np.array(LEVELS, float) * (1 + 0.025 * (j - 1.5))
        ax_ = [I[k]['bias_ph'][str(l)]['A_x'] for l in LEVELS]; ay_ = [I[k]['bias_ph'][str(l)]['A_y'] for l in LEVELS]
        ex = [I[k]['bias_ph'][str(l)]['A_x_se'] for l in LEVELS]; ey = [I[k]['bias_ph'][str(l)]['A_y_se'] for l in LEVELS]
        ax.errorbar(L, np.array(ax_, float), np.array(ex, float), color=c, marker='o', ms=2.5, lw=0.8, capsize=0, label=lab)
        ax.errorbar(L, np.array(ay_, float), np.array(ey, float), color=c, marker='o', mfc='white', ms=2.5, lw=0.8, ls=(0, (2, 1)), capsize=0)
    ax.axhline(0, color='0.5', lw=0.5); ax.set_xscale('log'); ax.set_xticks(LEVELS); ax.set_xticklabels([str(l) for l in LEVELS])
    ax.minorticks_off(); ax.set_xlabel('detected photons / 6-frame window'); ax.set_ylabel('bias toward pixel centre (nm)')
    ax.set_title('isolated emitters', fontsize=FS.FS_LABEL); ax.legend(loc='lower right', handlelength=1.2)
    # c: published model vs photons (own conditions)
    ax = fig.add_subplot(bot[0, 1], sharey=ax); FS.letter(ax, 'c', dx=-0.22)
    Pp = V['published']['iso_ph']; ph = sorted(Pp, key=float); Lp = np.array([float(p) for p in ph])
    ax.errorbar(Lp, [Pp[p]['A_x'] for p in ph], [Pp[p]['A_x_se'] for p in ph], color='tab:orange', marker='s', ms=2.5, lw=0.8, capsize=0, label='x')
    ax.errorbar(Lp, [Pp[p]['A_y'] for p in ph], [Pp[p]['A_y_se'] for p in ph], color='tab:orange', marker='s', mfc='white', ms=2.5, lw=0.8, ls=(0, (2, 1)), capsize=0, label='y')
    ax.axhline(0, color='0.5', lw=0.5); ax.set_xscale('log'); ax.set_xticks(Lp); ax.set_xticklabels([f'{int(v / 1000)}k' for v in Lp]); ax.minorticks_off()
    ax.set_xlabel('photons / frame (own conditions)'); ax.set_title('DECODE (published model)', fontsize=FS.FS_LABEL); ax.legend(loc='upper right')
    # d / e: vs density
    dens = [V['movies'][m]['density_um2'] for m in DENS]
    axd = fig.add_subplot(bot[0, 2]); FS.letter(axd, 'd', dx=-0.22); axe = fig.add_subplot(bot[0, 3]); FS.letter(axe, 'e', dx=-0.22)
    for j, (k, lab) in enumerate(METH):
        c = FS.COL['v2' if k == 'v2' else k]; d = np.array(dens) * (1 + 0.03 * (j - 1.5))
        R = [V['movies'][m]['methods'].get(k) for m in DENS]
        if any(r is None for r in R): continue
        axd.errorbar(d, [r['bias']['A_x'] for r in R], [r['bias']['A_x_se'] for r in R], color=c, marker='o', ms=2.5, lw=0.8, capsize=0, label=lab)
        axd.errorbar(d, [r['bias']['A_y'] for r in R], [r['bias']['A_y_se'] for r in R], color=c, marker='o', mfc='white', ms=2.5, lw=0.8, ls=(0, (2, 1)), capsize=0)
        axe.plot(d, [r['c1_x'] for r in R], color=c, marker='o', ms=2.5, lw=0.8, label=lab)
        axe.plot(d, [r['c1_y'] for r in R], color=c, marker='o', mfc='white', ms=2.5, lw=0.8, ls=(0, (2, 1)))
    S = V['published']['sets']; ps = ['mod', 'train']
    axd.errorbar([S[s]['density_um2'] for s in ps], [S[s]['A_x'] for s in ps], [S[s]['A_x_se'] for s in ps], color='tab:orange', marker='s', ms=2.5, lw=0, elinewidth=0.8, label='DECODE (published model)')
    axe.plot([S[s]['density_um2'] for s in ps], [S[s]['c1_x'] for s in ps], color='tab:orange', marker='s', ms=2.5, lw=0, label='DECODE (published model)')
    axe.plot(dens, [V['movies'][m]['gt']['c1_x'] for m in DENS], color='k', lw=0.8, marker='.', ms=2, label='ground truth')
    axd.axhline(0, color='0.5', lw=0.5)
    for ax in (axd, axe):
        ax.set_xscale('log'); ax.set_xlabel('emitters ON / µm² / window'); ax.minorticks_off()
        ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
    axd.set_ylabel('bias toward pixel centre (nm)'); axe.set_ylabel('|c$_1$| (sub-pixel histogram)')
    axd.set_title('vs density (500 photons)', fontsize=FS.FS_LABEL); axe.set_title('histogram non-uniformity', fontsize=FS.FS_LABEL)
    axe.set_ylim(bottom=0); axe.legend(loc='upper left', handlelength=1.2, bbox_to_anchor=(1.0, 1.0))
    FS.save(fig, os.path.join(a.outdir, 'si_pixel_bias'), bbox_inches='tight'); plt.close(fig)


def _f(v, fmt='%+.2f'):
    return '-' if v is None or not np.isfinite(v) else fmt % v


def caption(V, a):
    I = V['movies']['iso']['methods']; L = []
    L.append('Pixel bias of the localization methods. (a) Sub-pixel histograms (x solid, y dashed; position within the pixel, 0 = pixel '
             f'centre; 20 bins, normalised to 1 for a uniform histogram) on the {HIST_MOVIE} simulation, with the ground-truth positions and, '
             'for DECODE\'s published model, its own simulated conditions (10 emitters per 40 x 40 px frame). |c1| = amplitude of the first '
             'Fourier harmonic of the histogram (x / y; 0 for a uniform histogram, noise floor 1/sqrt(n)). (b) Bias toward the pixel centre '
             '(first-harmonic amplitude A of the error vs the true sub-pixel position, e = -A sin(2 pi f); filled x, open y; +-1 SE) of '
             'isolated emitters vs detected photons per 6-frame window (each method\'s own filters, without the common precision cut, '
             'which removes most dim SMLM and DECODE localizations). (c) The same for DECODE\'s published model in its own conditions '
             '(astigmatic PSF, EMCCD, 127 x 117 nm pixels, photons per frame). (d, e) Bias and |c1| vs density (emitters ON per um^2 per '
             'window), DECODE\'s published model at 10 and 50 emitters per frame.')
    L.append('Rows: the common rule for every method (own filters, disjoint 6-frame windows, max(sigma_x, sigma_y) < 0.15 px on its own '
             'reported uncertainty; SIMCODE p >= 0.63 with its network sigma). Bias: rows matched within 1 px to a ground-truth emitter with '
             'no other emitter ON within 2 px. SIMFLUX uses the true pattern throughout; on the isolated-emitter movie SIMCODE also uses the '
             'true pattern (its pattern estimate fails on this sparse movie).')
    L.append('DECODE training: DECODE was retrained on our conditions following the training settings published with DECODE '
             '(AdamW, learning rate 6e-4, weight decay 0.1, ReduceLROnPlateau with factor 0.5, patience 20 and cooldown 10, batch 64, '
             '32 768 simulated frames per epoch), from scratch with a 512-step linear learning-rate warm-up (without it the training '
             'diverged in the first epoch). It was trained to convergence (all monitored metrics flat from epoch ~200) and the weights '
             'were frozen at epoch 300 (9.8 M training frames). The training simulations use the in-situ PSF v2 model only (the PSF of '
             'these simulations). With this training DECODE\'s pixel bias is reduced (e.g. D1, all localizations: |c1| 0.30 / 0.25 -> '
             '0.12 / 0.14 and bias 6.7 / 4.3 -> -0.2 / 1.8 nm x / y, vs the earlier, shorter-trained model) but does not go away.')
    L.append('NUMBERS (si_pixel_bias_values.json)')
    I = V['movies']['iso']['methods_nocut']
    for k, lab in METH:
        if k in I: L.append(f"  iso (no precision cut) {lab}: A x / y (nm) " + ', '.join(f"{l}: {_f(I[k]['bias_ph'][str(l)]['A_x'])} / {_f(I[k]['bias_ph'][str(l)]['A_y'])}" for l in LEVELS))
    for m in DENS:
        R = V['movies'][m]
        L.append(f"  {m} ({R['density_um2']:.2f} /um^2): " + '; '.join(f"{lab} |c1| {_f(R['methods'][k]['c1_x'], '%.3f')}/{_f(R['methods'][k]['c1_y'], '%.3f')} A {_f(R['methods'][k]['bias']['A_x'])}/{_f(R['methods'][k]['bias']['A_y'])} nm"
                                                                       for k, lab in METH if k in R['methods']) + f"; GT |c1| {R['gt']['c1_x']:.3f}")
    S = V['published']['sets']
    L.append('  DECODE (published model): ' + '; '.join(f"{s} ({S[s]['emitters_per_frame']}/frame) |c1| {S[s]['c1_x']:.3f}/{S[s]['c1_y']:.3f} A {S[s]['A_x']:+.2f}/{S[s]['A_y']:+.2f} nm" for s in S))
    L.append(f"Tables: DECODE <run>/decode/{a.decode}, SIMCODE v2_{a.model} (common rule 'v2c'), SMLM base/smlm, SIMFLUX base/sf_gtp; published model: {PUB}")
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decode', default='locs_v3.npz'); ap.add_argument('--model', default='m1809v2')
    ap.add_argument('--outdir', default=OUT); ap.add_argument('--redo', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cf = os.path.join(a.outdir, 'si_pixel_bias_cache.json')
    if os.path.exists(cf) and not a.redo and json.load(open(cf)).get('decode') == a.decode:
        V = json.load(open(cf))
    else:
        V = compute(a); V.update(decode=a.decode, model=a.model, made=time.ctime()); json.dump(V, open(cf, 'w'), indent=1, default=float)
    draw(V, a)
    json.dump(V, open(os.path.join(a.outdir, 'si_pixel_bias_values.json'), 'w'), indent=1, default=float)
    open(os.path.join(a.outdir, 'si_pixel_bias_caption.txt'), 'w').write(caption(V, a))
    print(caption(V, a))


if __name__ == '__main__':
    main()
