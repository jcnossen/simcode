"""SI figure 'si_crlb_precision' in the preprint's layout (smlmtorch 5f7fdcb scripts/estimation_vs_crlb_fastpsf.py /
estimation_vs_crlb_varying_bg.py; paper figure figures/si_crlb_precision.pdf), redrawn from s10.py's json:
  A  'Localization precision vs signal photons (X axis)', constant background 1 photon/px/frame: CRLB SMLM / SIMFLUX, SIMCODE bound
     (dashed), SMLM MLE, SIMFLUX MLE, NN Output, SIMCODE (RMSD, 2 % trim)
  B  'Localization precision for fluctuating fluorescence background', bg ~ U(1, 6) per frame: SMLM CRLB, NN Output RMSD + predicted
     precision, SIMCODE bound, SIMCODE RMSD + predicted precision (no SIMFLUX, as the preprint)
  C  'Example 6-frame stack': the 6 frames of one U(1, 6)-background ROI (re-rendered with s10.py's scene / seed of that bin), each
     frame's bg, the true position (red +)
'SIMCODE bound' (legend) = the SIMFLUX CRLB with a background per frame (crlb_sf_pfbg; no separate SIMCODE CRLB is computed;
SIMCODE fits a background per frame). Units: x precision in pixels (108.33 nm), log axes. Open symbols: bins where the network detected < 50 % of
the emitters (NN Output / SIMCODE). NN Output = pass-1 head prior; SIMCODE = final pass-2 output with the true pattern.
usage: python analysis/papersim/si_crlb_figure_pp.py [--tag m1809v2] [--example_bin 5] [--psf v2]
output: out/paper_figs_v2/si_crlb_precision.{pdf,svg,png} (+ _values.json)"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C


def example_stack(tag_json, bi, psf):
    """re-render the U(1, 6) scene of bin bi exactly as s10.py did (same rng / generator seeds) -> ROI [6,16,16] of emitter 0, gt x / y
    in the ROI, bg [6]"""
    import torch, s10
    P = tag_json['protocol']; seed = 10
    coll = P.get('collected_at_z', 1.0); ph = P['photons'][bi]
    nfov = int(np.ceil(P['n_per_bin'] / (s10.NC * s10.NC)))
    if psf == 'v2':
        R = C.RendererV2(s10.FOV, s10.FOV); R.z = P['z_um']
    else:
        R = C.Renderer(s10.FOV, s10.FOV)
    rng = np.random.default_rng([seed, bi, 1])
    gen = torch.Generator(device='cuda'); gen.manual_seed(seed * 1000 + bi * 10 + 1)
    S = s10.scene(nfov, ph / coll, 'unif16', rng)
    fr = s10.render(S, R, gen)
    roi = s10.rois(fr, S)[0]
    return roi, S['x'][0] - S['cx0'][0], S['y'][0] - S['cy0'][0], S['bg'][S['fov'][0]], ph


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='m1809v2'); ap.add_argument('--example_bin', type=int, default=7)
    ap.add_argument('--psf', default='v2'); ap.add_argument('--outdir', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2'))
    ap.add_argument('--name', default='si_crlb_precision')
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.load(open(os.path.join(C.OUT, 's10', f's10_{a.tag}.json')))
    def ser(cond, key, f='rmsd_trim2'):
        q = [R['cond'][cond][b] for b in sorted(R['cond'][cond], key=int)]; ph = np.array([x['photons'] for x in q])
        if key.startswith('crlb'): return ph, np.array([x[key][0] for x in q]), None
        rec = np.array([x[key].get('recall', 1.0) for x in q], float); rec[~np.isfinite(rec)] = 1.0
        return ph, np.array([x[key][f][0] for x in q]), rec
    fig = plt.figure(figsize=(6.6, 10.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.42], hspace=0.38, left=0.13, right=0.97, top=0.965, bottom=0.03)
    A = fig.add_subplot(gs[0]); B = fig.add_subplot(gs[1])
    # A (preprint: CRLB SMLM black dashed, SIMFLUX green dashed, SIMCODE grey dashed; SMLM black o, SIMFLUX green s, NN blue ^, SIMCODE red v)
    hA = []
    for k, c, lab in (('crlb_smlm', 'k', 'SMLM CRLB'), ('crlb_sf', 'green', 'SIMFLUX CRLB'), ('crlb_sf_pfbg', 'grey', 'SIMCODE bound')):
        ph, y, _ = ser('const1', k); hA += A.plot(ph, y, '--', color=c, lw=1.2, label=lab)
    for k, c, mk, lab in (('smlm', 'k', 'o', 'SMLM MLE'), ('sf', 'green', 's', 'SIMFLUX MLE'), ('nn_p1', 'tab:blue', '^', 'NN Output'), ('v2_p2', 'red', 'v', 'SIMCODE')):
        ph, y, rec = ser('const1', k); low = rec < 0.5 if k in ('nn_p1', 'v2_p2') else np.zeros(len(ph), bool)
        hA += A.plot(ph, y, '-', color=c, marker=mk, ms=5, label=lab, markevery=list(np.nonzero(~low)[0]))
        if low.any(): A.plot(ph[low], y[low], mk, color=c, ms=5, mfc='white', ls='none')
    A.set_title('Localization precision vs signal photons (X axis)')
    # B
    hB = []
    ph, y, _ = ser('unif16', 'crlb_smlm'); hB += B.plot(ph, y, '--', color='tab:blue', lw=1, alpha=0.7, label='SMLM CRLB')
    ph, y, rec = ser('unif16', 'nn_p1'); low = rec < 0.5
    hB += B.plot(ph, y, '-', color='tab:blue', marker='o', ms=5, label='NN Output RMSD', markevery=list(np.nonzero(~low)[0]))
    if low.any(): B.plot(ph[low], y[low], 'o', color='tab:blue', ms=5, mfc='white', ls='none')
    ph, y, _ = ser('unif16', 'nn_p1', 'pred_mean'); hB += B.plot(ph, y, ':', color='tab:blue', marker='x', ms=5, lw=1.5, label='NN Output predicted precision')
    ph, y, _ = ser('unif16', 'crlb_sf_pfbg'); hB += B.plot(ph, y, '--', color='red', lw=1, alpha=0.6, label='SIMCODE bound')
    ph, y, rec = ser('unif16', 'v2_p2'); low = rec < 0.5
    hB += B.plot(ph, y, '-', color='red', marker='v', ms=5, label='SIMCODE RMSD', markevery=list(np.nonzero(~low)[0]))
    if low.any(): B.plot(ph[low], y[low], 'v', color='red', ms=5, mfc='white', ls='none')
    ph, y, _ = ser('unif16', 'v2_p2', 'pred_mean'); hB += B.plot(ph, y, ':', color='red', marker='x', ms=5, lw=1.5, label='SIMCODE predicted precision')
    B.set_title('Localization precision for fluctuating fluorescence background')
    for X, h, L in ((A, hA, 'A'), (B, hB, 'B')):
        X.set_xscale('log'); X.set_yscale('log'); X.grid(True, which='both', alpha=0.3)
        X.set_ylim(1e-2, 1.6); X.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        X.set_xlabel('Total signal [photons]'); X.set_ylabel('x precision [pixels]'); X.legend(handles=h, fontsize=9, loc='upper right')
        X.text(-0.1, 1.02, L, transform=X.transAxes, fontsize=14, fontweight='bold', ha='right', va='bottom')
    # C example stack
    roi, ex, ey, bg, phx = example_stack(R, a.example_bin, a.psf)
    gc = gs[2].subgridspec(1, 6, wspace=0.08)
    import matplotlib.ticker
    vmax = roi.max()
    for t in range(6):
        X = fig.add_subplot(gc[t]); X.imshow(roi[t], cmap='gray', vmin=0, vmax=vmax); X.plot(ex, ey, '+', color='red', ms=8, mew=1.2)
        X.set_xticks([]); X.set_yticks([]); X.set_title(f'bg={bg[t]:.1f}', fontsize=9)
        if t == 0:
            X.set_ylabel('bg~U(1,6)', fontsize=8)
            X.text(-0.35, 1.35, 'C', transform=X.transAxes, fontsize=14, fontweight='bold', ha='right', va='bottom')
    fig.text(0.55, gc[0].get_position(fig).y1 + 0.022, f'Example 6-frame stack ({phx:.0f} photons)', ha='center', fontsize=10)
    os.makedirs(a.outdir, exist_ok=True); out = os.path.join(a.outdir, a.name)
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        fig.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    json.dump(dict(source=f's10_{a.tag}.json', example=dict(bin=a.example_bin, photons=phx, bg=list(map(float, bg)), x_roi=float(ex), y_roi=float(ey)),
                   simcode_crlb='crlb_sf_pfbg (SIMFLUX model with one background per frame)'), open(out + '_values.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
