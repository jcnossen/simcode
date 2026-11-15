"""Old vs new SMLM / SIMFLUX baselines of an experimental figure (fig_exp_new.py values json): the figure made with the earlier
baseline tables (psf3d_avg PSF at 0.055 um, camera gain 0.56 + old offset map) against the one with the PSF-v2 + camera-v2
tables (paper_pipeline.py --camera calib --psf_z_run; out/paper2/<ds>/baselines_psfv2).

usage: python analysis/paper/baseline_compare.py <ds> <old values.json> <new values.json> <old baseline dir> <new baseline dir> <out prefix>
  baseline dir = the paper_pipeline.py --out folder (run.json, smlm/info.json, sf/info.json)
outputs: <out prefix>.md / .json: per method fair FRC [nm], n FRC set, RSP, all rows in the frame range, and the gain of SIMCODE v2
over SIMFLUX = 100 (FRC_SIMFLUX - FRC_SIMCODE v2) / FRC_SIMFLUX; plus the baseline runs' PSF / fit plane / camera / ROI counts."""
import sys, json, os


def load(p):
    return json.load(open(p))


def base_info(d):
    r = load(f'{d}/run.json'); s = load(f'{d}/smlm/info.json'); f = load(f'{d}/sf/info.json')
    return dict(psf=r['args']['psf'], psf_z_fit_um=s.get('psf_z_fit_um'), gain=r['gain'], offset=r['offset'], camera=r['args']['camera'],
                numrois=s['numrois'], n_after_chisq=s['n_after_chisq'], chisq_per_px_median=s.get('chisq_per_px_median'),
                smlm_median_photons=s.get('median_photons'), smlm_median_bg=s.get('median_bg'), smlm_median_crlb_nm=s.get('median_crlb_nm'),
                sf_n_modsel=f['n_modsel'], sf_n_out=f['n_out'], sf_median_crlb_nm=f.get('median_crlb_nm'),
                sf_median_photons=f.get('median_photons'), pattern=f.get('pattern'), timing_s=r.get('timing', {}).get('total_s'))


def main():
    ds, vo, vn, bo, bn, out = sys.argv[1:7]
    O, N = load(vo), load(vn)
    assert O['labels'] == N['labels'], (O['labels'], N['labels'])
    L = O['labels']; iv = L.index('SIMCODE v2') if 'SIMCODE v2' in L else L.index('SIMCODE'); isf = L.index('SIMFLUX')
    rows = []
    for i, lab in enumerate(L):
        rows.append(dict(method=lab, frc_old=O['frc_nm'][i], frc_new=N['frc_nm'][i], n_frc_old=O['n_frc'][i], n_frc_new=N['n_frc'][i],
                         rsp_old=O['pearson'][i], rsp_new=N['pearson'][i], n_all_old=O['n_all'][i], n_all_new=N['n_all'][i]))
    gain = {k: 100 * (V['frc_nm'][isf] - V['frc_nm'][iv]) / V['frc_nm'][isf] for k, V in (('old', O), ('new', N))}
    res = dict(dataset=ds, values_old=os.path.abspath(vo), values_new=os.path.abspath(vn), rows=rows,
               gain_v2_vs_simflux_pct=gain, baseline_old=base_info(bo), baseline_new=base_info(bn), aois_old=O['aois'], aois_new=N['aois'])
    json.dump(res, open(out + '.json', 'w'), indent=1)
    f = lambda x: f'{x:,}'.replace(',', ' ')
    md = [f'## {ds}: SMLM / SIMFLUX baselines old (psf3d_avg, gain 0.56) vs new (PSF v2 + camera v2)', '',
          f'old figure: `{vo}`', f'new figure: `{vn}`', '',
          '| method | FRC old [nm] | FRC new [nm] | n FRC old | n FRC new | RSP old | RSP new | rows old | rows new |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['method']} | {r['frc_old']:.2f} | {r['frc_new']:.2f} | {f(r['n_frc_old'])} | {f(r['n_frc_new'])} | "
                  f"{r['rsp_old']:.3f} | {r['rsp_new']:.3f} | {f(r['n_all_old'])} | {f(r['n_all_new'])} |")
    md += ['', f"gain of {L[iv]} over SIMFLUX (100 (FRC_SF - FRC_v2) / FRC_SF): old {gain['old']:.1f} %, new {gain['new']:.1f} %", '',
           '| baseline run | old | new |', '|---|---|---|']
    bi_o, bi_n = res['baseline_old'], res['baseline_new']
    for k in ('psf', 'psf_z_fit_um', 'camera', 'gain', 'offset', 'numrois', 'n_after_chisq', 'chisq_per_px_median', 'smlm_median_photons',
              'smlm_median_bg', 'smlm_median_crlb_nm', 'sf_n_modsel', 'sf_n_out', 'sf_median_crlb_nm', 'sf_median_photons', 'timing_s'):
        g = lambda v: (f(v) if isinstance(v, int) else (f'{v:.4g}' if isinstance(v, float) else str(v)))
        md.append(f'| {k} | {g(bi_o[k])} | {g(bi_n[k])} |')
    md += ['', 'rows = all rows of the table in the frame range (before the fair-FRC filter); n FRC = the fair-FRC set (disjoint windows,',
           'max sigma < 0.15 px, crop 5-115 px, own DME v2 drift); RSP = Pearson correlation with the widefield mean image.']
    if bi_o['psf'] != bi_n['psf']:
        md.append('Photons are in each run\'s own units (gain 0.56 vs 0.4256 e/ADU; psf3d_avg vs PSF v2 intensity parameter).')
    open(out + '.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
