"""Collect the referee-check outputs (out/refcheck/*.json) into out/paper_figs_v2/REFEREE_CHECKS.json (one file, keyed by the
check letters of REFEREE_CHECKS.md; curves and per-tile lists kept, docstrings dropped)."""
import os, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, 'out', 'refcheck')
J = lambda f: json.load(open(os.path.join(R, f)))


def strip(d):
    if isinstance(d, dict): return {k: strip(v) for k, v in d.items() if k != 'protocol'}
    if isinstance(d, list): return [strip(v) for v in d]
    return d


out = dict(
    note='Referee checks of Figs 2-4 (report only; no figure changed). Text: REFEREE_CHECKS.md. Code: analysis/refcheck/ (branch refcheck).',
    sources={k: os.path.join(R, v) for k, v in dict(sim='sim_checks.json', sim_jack='sim_jack.json', d='d_simreal.json',
             real_1809_2='real_1809_2.json', real_25nM='real_2.5nM_2.json', real_25nM_extra='real_2.5nM_2_extra.json',
             i='i_smlmzoom.json', g_counts='sf_loose/2.5nM_2/counts.json').items()},
    a_b_sim=strip(J('sim_checks.json')['sets']),
    c_sim_jackknife=strip(J('sim_jack.json')['sets']['Dexp1']['jackknife']),
    d_sim_vs_real=strip(J('d_simreal.json')),
    e_f_h_real={'1809_2': strip(J('real_1809_2.json')), '2.5nM_2': strip(J('real_2.5nM_2.json'))},
    g_sf_loose=dict(counts=J('sf_loose/2.5nM_2/counts.json'), eval=strip(J('real_2.5nM_2_extra.json'))),
    i_smlm_zoom=strip(J('i_smlmzoom.json')))
fn = os.path.join(REPO, 'out', 'paper_figs_v2', 'REFEREE_CHECKS.json')
json.dump(out, open(fn, 'w'), indent=1, default=float)
print(fn, os.path.getsize(fn))
