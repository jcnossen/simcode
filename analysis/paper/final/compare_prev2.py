"""Old (*_prev2) vs new numbers of the regenerated paper outputs (2026-11-17, final method: SIMCODE = the pass-2 network table,
DECODE = v3): per figure, SIMCODE and DECODE count, FRC / GT-FRC, RSP, median sigma where the values file has them.
usage: python compare_prev2.py <out md>"""
import json, os, sys

PF = os.path.expanduser('~/simcode2/out/paper_figs_v2'); PS = os.path.expanduser('~/simcode2/out/papersim/figures')
f1 = lambda v, d=1: '-' if v is None or v != v else f'{v:.{d}f}'


def real(name):
    o, n = (json.load(open(f'{PF}/{name}{s}_values.json')) for s in ('_prev2', ''))
    L = []
    for i, lab in enumerate(o['labels']):
        if lab not in ('SIMCODE', 'DECODE'): continue
        L.append(f"| {name} | {lab} | {o['n_frc'][i]} -> {n['n_frc'][i]} | FRC {f1(o['frc_nm'][i])} -> {f1(n['frc_nm'][i])} nm | "
                 f"RSP {f1(o['pearson'][i], 3)} -> {f1(n['pearson'][i], 3)} | {f1(o.get('median_sigma_nm', [None] * 9)[i])} -> "
                 f"{f1(n.get('median_sigma_nm', [None] * 9)[i])} |")
    return L


def fig2(name):
    o, n = (json.load(open(f'{PF}/{name}{s}_values.json')) for s in ('_prev2', ''))
    L = []
    for k in ('decode', 'v2'):
        a, b = o['methods'][k], n['methods'][k]
        L.append(f"| {name} | {a['label']} | {a['n']} -> {b['n']} | GT-FRC {f1(a['gt_frc_nm'])} -> {f1(b['gt_frc_nm'])} nm; frames to 30 nm "
                 f"{f1(a['ttf']['frames_to'].get('30'), 0)} -> {f1(b['ttf']['frames_to'].get('30'), 0)}; false pos. {f1(a['fp_3sigma'], 3)} -> "
                 f"{f1(b['fp_3sigma'], 3)}; rms TP {f1(a['rms_tp_nm'])} -> {f1(b['rms_tp_nm'])} nm | - | {f1(a.get('med_sig_nm'))} -> {f1(b.get('med_sig_nm'))} |")
    return L


def table_md(path_old, path_new, keys=('DECODE', 'SIMCODE')):
    """rows of a markdown table whose method column contains one of keys: old line / new line"""
    rd = lambda p: [l for l in open(p).read().splitlines() if l.startswith('|') and any(k in l for k in keys)]
    return [f'old: {a}\n\nnew: {b}' for a, b in zip(rd(path_old), rd(path_new))]


def main():
    L = ['# Old (*_prev2) vs new paper numbers (final method, 2026-11-17)', '',
         '| figure | method | n | FRC / GT-FRC | RSP | median sigma (nm) |', '|---|---|---|---|---|---|']
    for nm in ('1809_2_fig3_psfv2', '2.5nM_2_fig4_psfv2'):
        if os.path.exists(f'{PF}/{nm}_values.json'): L += real(nm)
    if os.path.exists(f'{PF}/fig_sim_Dexp_v2psf_wsm3d_values.json'): L += fig2('fig_sim_Dexp_v2psf_wsm3d')
    for nm in ('fig_sim_D_v2psf_wsm3d_common', 'fig_sim_Dexp_v2psf_wsm3d_common'):
        a, b = f'{PS}/{nm}_prev2_table.md', f'{PS}/{nm}_table.md'
        if os.path.exists(a) and os.path.exists(b): L += ['', f'## {nm}', ''] + table_md(a, b)
    for nm, keys in (('si_simulated_live_smlm', ('DECODE', 'SIMCODE')), ('si_crlb_precision', ('SIMCODE', 'NN'))):
        a, b = f'{PF}/{nm}_prev2_table.md', f'{PF}/{nm}_table.md'
        if os.path.exists(a) and os.path.exists(b): L += ['', f'## {nm}', ''] + table_md(a, b, keys)
    open(sys.argv[1], 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
