"""Print the metrics table of a papersim run: python analysis/papersim/table.py <run dir> [model]"""
import json, sys, os
run = sys.argv[1]; ns = sys.argv[2] if len(sys.argv) > 2 else 'm1809'
R = json.load(open(os.path.join(run, f'metrics_{ns}.json')))
print(f"{R['run']}  ({R['nframes']} frames)")
print(f"{'method':8s} {'n_all':>7s} {'n_disj':>7s} {'n_frc':>6s} {'FRC nm':>7s} {'J_end':>6s} {'J_frcset':>8s} {'recall':>6s} {'rms x,y nm':>14s} {'med sig nm':>14s} {'med|e|/s':>12s} {'offset nm':>13s}")
for k, r in R['methods'].items():
    g = r['gt_match']
    f = lambda v: ','.join(f'{x:.1f}' for x in v)
    print(f"{k:8s} {r['n_all']:7d} {r['n_disjoint']:7d} {r['n_frc']:6d} {r.get('frc_nm', float('nan')):7.1f} {r['jaccard_end']:6.3f} {r['jaccard_frcset_end']:8.3f} "
          f"{g.get('recall_gt_ph150', float('nan')):6.3f} {f(g.get('rms_nm', [])):>14s} {f(g.get('median_sigma_nm', [])):>14s} "
          f"{','.join(f'{x:.2f}' for x in g.get('median_abs_err_over_sigma', [])):>12s} {f(g.get('mean_offset_nm', [])):>13s}")
