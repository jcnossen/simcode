"""print compact tables of out/pixbias/sim_*.json"""
import json, sys, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out', 'pixbias')
for nm in sys.argv[1:]:
    R = json.load(open(os.path.join(OUT, f'sim_{nm}.json')))
    for ps in R:
        for sn in R[ps]:
            r = R[ps][sn]
            for en in r:
                if en == 'leak': continue
                for ax in ('x', 'y'):
                    q = r[en][ax]
                    if q['all'] is None: continue
                    ph = ' '.join(f'{l}:{v["A_nm"]:+.2f}±{v["A_se_nm"]:.2f}' for l, v in q['ph'].items() if v)
                    print(f'{nm:10s} {ps:7s} {sn:5s} {en:6s} {ax} n={q["all"]["n"]:7d} A={q["all"]["A_nm"]:+.2f}±{q["all"]["A_se_nm"]:.2f} '
                          f'rms={q["all"]["rms_nm"]:.1f} | hist c1 est {q["hist_est"]["c1_re"]:+.4f} gt {q["hist_gt"]["c1_re"]:+.4f} A_hist={q["hist_est"]["A_nm"]:+.2f} '
                          f'| eq: A={q["eq"]["all"]["A_nm"]:+.2f} rms={q["eq"]["all"]["rms_nm"]:.1f} | ph {ph}')
            if 'leak' in r:
                print('   leak', {ax: {l: round(v['slope'], 2) for l, v in d.items()} for ax, d in r['leak'].items()})
