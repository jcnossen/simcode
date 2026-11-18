"""Stage g (report): <out>/report/report.html (self-contained: tables + inline PNG figures) and report.json: settings, every
stage summary with its checks, camera maps, PSF images per pattern angle and the held-out table, the in-situ pattern vs the
pipeline's estimate, counts of the localization tables, the FRC of the paper's selection (simcode.evaluation.frc, 300-frame block halves,
drift-corrected with DME v2 when available), the gate table of stage e."""
import base64, html, io, json, os
import numpy as np


def _png(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=80, bbox_inches='tight')
    import matplotlib.pyplot as plt; plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(b.getvalue()).decode() + '">'


def _tab(rows, head=None):
    h = '<table>' + ('<tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in head) + '</tr>' if head else '')
    for r in rows:
        h += '<tr>' + ''.join(f'<td>{html.escape(_fmt(c))}</td>' for c in r) + '</tr>'
    return h + '</table>'


def _fmt(v):
    if isinstance(v, float): return f'{v:.4g}'
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, float) for x in v): return '[' + ', '.join(f'{x:.4g}' for x in v) + ']'
    s = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) < 300 else s[:300] + ' ...'


def frc_of(L, sel=None):
    """FRC (300-frame block halves) of the rows sel of a simcode.dataset.Dataset table, after its DME v2 drift (estimated
    on the good, disjoint windows of the whole table) when that is available -> (frc nm, True or the reason for no drift
    correction, drift or None)"""
    try:
        from simcode.util.drift import dme_package
        dme_package()                                    # puts the configured DME checkout on the path
        rows = (L.good & L.disjoint) if ('good' in L and 'disjoint' in L) else None
        d = L.estimate_drift(rows=rows).numpy(); dr = True
    except Exception as e:
        d, dr = None, f'no drift correction ({type(e).__name__}: {e})'
    S = L if sel is None else L[sel]
    if d is not None: S.apply_drift()
    return float(S.frc()), dr, d


def run(ds):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from simcode.pipeline.localization_pipeline import STAGES
    c = ds.c; d = ds.dir('report'); R = dict(name=c['name'], config=c, stages={})
    parts = [f'<h1>SIMCODE v2 automatic pipeline: {html.escape(c["name"])}</h1><p>movie {html.escape(c["movie"])}<br>out {html.escape(ds.out)}</p>']
    rows = []
    for s in STAGES[:-1]:
        S = ds.summary(s); R['stages'][s] = S
        if S is None: rows.append([s, 'not run', '', '']); continue
        ck = S.get('checks', [])
        rows.append([s, f'{S.get("time_s", 0):.0f} s', f'{sum(k["ok"] for k in ck)} / {len(ck)} ok',
                     '; '.join(f'{k["name"]}: {_fmt(k.get("value"))}' for k in ck if not k['ok'])])
    parts.append('<h2>Stages</h2>' + _tab(rows, ['stage', 'time', 'checks', 'failed / warnings']))
    cam = R['stages'].get('camera')
    if cam:
        parts.append('<h2>Camera</h2>' + _tab([[k, cam.get(k)] for k in ('gain', 'gain_source', 'gain_measured', 'r2', 'level_span_adu', 'offset',
                                                                           'offset_source', 'readnoise_e', 'readnoise_source', 'bg_e_per_px')]))
    mp = os.path.join(ds.out, 'cammaps', 'camera_maps.npz')
    if os.path.exists(mp):
        M = np.load(mp); fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for a, k in zip(ax, ('offset_adu', 'readnoise_e')):
            v = M[k]; im = a.imshow(v, vmin=np.percentile(v, 1), vmax=np.percentile(v, 99)); a.set_title(k); plt.colorbar(im, ax=a)
        parts.append(_png(fig) + _tab([[k, v] for k, v in R['stages']['cammaps']['stats'].items()]))
    sp = R['stages'].get('spots'); tr = R['stages'].get('tracks')
    if sp: parts.append('<h2>Spots and tracks</h2>' + _tab([[k, v] for k, v in sp['stats'].items()] + ([[k, v] for k, v in tr['stats'].items()] if tr else [])))
    ps = R['stages'].get('psf')
    if ps:
        parts.append('<h2>In-situ PSF</h2><p>init: ' + html.escape(str(ps['init'])) + '</p>')
        fr = [[n, f['nll_per_px'], f['resid_var'], f['focus'], float(np.sqrt(np.sum(np.square(f['zernike'])))), f.get('blur_px2'), f.get('gamma'), f.get('zeta'), f['z_median'], f['fit_s']]
              for n, f in ps['fits'].items()]
        parts.append(_tab(fr, ['fit', 'nll/px', 'resid var', 'focus um', '|Z| waves', 'blur px2', 'gamma', 'zeta', 'z median', 'fit s']))
        if ps.get('heldout'):
            parts.append('<h3>Held-out (fit cb0, test cb1)</h3>' + _tab([[k, v] for k, v in ps['heldout'].items()]))
        sp_ = ps.get('spread')
        if sp_:
            fp, sc = sp_['fpc'], sp_.get('scale', 1.0)
            if fp:
                txt = (f'sd = the sd over the draws x {fp:.2f} (1 / sqrt(1 - n / N)): the sd of a fit on an independent set of n samples'
                       + (f', x {sc:.2f} (sqrt(n / psf.nsamples)) for the {sp_["nsamples"]} samples of the PSF fit' if sc != 1 else '')
                       + f'. One draw (fits + held-out test): {sp_.get("wall_s_per_draw", float("nan")):.0f} s.')
            else:
                txt = f'every draw uses all {sp_["n_candidates"]} cb0 samples: no spread estimate (psf.nsamples or psf.spread_nsamples < the cb0 samples).'
            parts.append(f'<h3>Sampling spread ({sp_["k"]} draws of {sp_["n"]} of the {sp_["n_candidates"]} cb0 samples)</h3><p>{html.escape(txt)}</p>')
            parts.append(_tab([[q, sp_['mean'].get(q), sp_['sd_draws'].get(q), sp_['sd'].get(q)] for q in sp_['sd_draws']],
                              ['quantity', 'mean', 'sd over draws', f'sd ({sp_.get("nsamples", sp_["n"])} samples)']))
        im = os.path.join(ds.out, 'psf', 'psf_images.npy')
        if os.path.exists(im):
            P = np.load(im); nz = P.shape[0]; fig, ax = plt.subplots(3, nz, figsize=(2 * nz, 6))
            for i in range(nz):
                ax[0, i].imshow(P[i, 0]); ax[1, i].imshow(P[i, 1]); ax[2, i].imshow(P[i, 1] - P[i, 0], cmap='RdBu')
                for a in ax[:, i]: a.set_xticks([]); a.set_yticks([])
            ax[0, 0].set_ylabel('angle 0'); ax[1, 0].set_ylabel('angle 1'); ax[2, 0].set_ylabel('1 - 0')
            parts.append(_png(fig))
        pt = ps['pattern']
        parts.append('<h3>Pattern (track intensities)</h3>' + _tab([[k, pt[k]] for k in ('k', 'pitch_px', 'phase', 'mod', 'relint', 'n', 'sectors')]))
    tn = R['stages'].get('train')
    if tn and tn.get('accept'):
        parts.append('<h2>Model</h2>' + _tab([['model', tn.get('model')], ['quick', tn['quick']], ['gates', tn['gates']]] + [[k, v] for k, v in tn['accept'].items()]))
    pr = R['stages'].get('process')
    if pr:
        S = pr['summary']; px = c['optics']['pixel_um'] * 1e3
        rows = [['model', pr.get('model')], ['estimated pitch px', pr.get('pattern_pitch_px')], ['timing', S.get('timing')]]
        from simcode import table as SP
        for n in SP.TABLES:
            if n in S: rows.append([n, {k: S[n].get(k) for k in ('n_rows', 'n_main', 'n_sel', 'median_photons', 'median_sigma_nm')}])
        parts.append('<h2>Processing</h2>' + _tab(rows))
        last = SP.DEFAULT_TABLE
        f = os.path.join(pr['run'], last + '.hdf5')
        if os.path.exists(f):
            from simcode.dataset import Dataset
            try:
                L = Dataset.load_table(f); sel = L.mask().numpy()
                v, dr, D = frc_of(L, sel)
                R['frc'] = dict(table=last, selection='max(sx, sy) <= 12 nm, p >= 0.63, inside the crop', n=int(sel.sum()), frc_nm=v, drift=dr)
                parts.append(_tab([['FRC (selection <= 12 nm, 300-frame block halves)', f'{v:.2f} nm, {int(sel.sum())} rows, drift {dr}']]))
                if D is not None:
                    fig, a = plt.subplots(figsize=(7, 2.5)); a.plot(D[:, 0] * px, label='x'); a.plot(D[:, 1] * px, label='y')
                    a.set_xlabel('frame'); a.set_ylabel('drift nm'); a.legend(); parts.append(_png(fig))
            except Exception as e:
                parts.append(f'<p>FRC not computed: {html.escape(repr(e))}</p>')
    css = ('<style>body{font-family:sans-serif;max-width:1100px;margin:16px auto;padding:0 16px;background:#fff;color:#111}'
           'table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;font-size:13px;vertical-align:top}'
           'img{max-width:100%}</style>')
    open(os.path.join(d, 'report.html'), 'w').write('<!doctype html><html><head><meta charset="utf-8"><title>SIMCODE auto report</title>'
                                                    + css + '</head><body>' + '\n'.join(parts) + '</body></html>')
    json.dump(R, open(os.path.join(d, 'report.json'), 'w'), indent=1, default=str)
    return dict(html=os.path.join(d, 'report.html'), frc=R.get('frc'), checks=[])
