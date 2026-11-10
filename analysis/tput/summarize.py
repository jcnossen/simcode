"""markdown tables from acc_gate.json, movie_*.json, bench_*.json, movies/*/timing.json; the directory is an argument (default
<simcode2 out>/tput); acc_gate.json / movie_*.json (accuracy acceptance) are optional.
Keeps per section the first measurement whose time window had no foreign GPU process (gpu_samples.log of gpu_sampler.sh)
-> clean_<data>.json.
usage: python analysis/tput/summarize.py [tput dir]"""
import json, os, glob, sys
if len(sys.argv) > 1: T = os.path.join(sys.argv[1], '')
else:
    from simcode2 import config
    T = os.path.join(config.out_dir('tput'), '')
L = []
G = json.load(open(T + 'acc_gate.json')) if os.path.exists(T + 'acc_gate.json') else {}
vs = [v for v in next(iter(G.values())) if v != 'orig'] if G else []
L.append('Gate (pattern solve variants vs the original solve, same head outputs; d RMS/CRLB [95 % CI], 200-700 / 700-2000 / >2000 GT ph; recall, fringe-error %, zcal; speed-up on a busy GPU)')
L.append('| front, scene | variant | x speed | d 200-700 | d 700-2000 | d >2000 | recall (orig) | fringe % (orig) | zcal (orig) | rows moved > 0.1 px |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for key, R in G.items():
    o = R['orig']
    for v in vs:
        r = R[v]; p = r['paired']
        f = lambda b: f"{p[b]['d']:+.4f} [{p[b]['ci'][0]:+.4f},{p[b]['ci'][1]:+.4f}]" if b in p else '-'
        L.append(f"| {key} | {v} | {o['time'] / r['time']:.1f} | {f('200-700')} | {f('700-2000')} | {f('2000-1e+09')} | {r['recall']:.4f} ({o['recall']:.4f}) | "
                 f"{r['fringe_err_pct']:.2f} ({o['fringe_err_pct']:.2f}) | {r['zcal']:.3f} ({o['zcal']:.3f}) | {100 * r['dpos']['frac_gt_0p1']:.3f} % |")
for f in sorted(glob.glob(T + 'movie_*.json')):
    M = json.load(open(f)); nm = os.path.basename(f)[6:-5]
    L.append(f'\nMovie {nm}: paired d RMS/CRLB [event-block 95 % CI] vs the original; recall')
    L.append('| pair | 200-400 | 400-700 | 700-1200 | 1200-2000 | 2000-4000 | 200-2000 | recall B (A) |')
    L.append('|---|---|---|---|---|---|---|---|')
    pairs = sorted({k.split('|')[0] for k in M['paired']})
    for pr in pairs:
        b, a = pr.split('-')
        if a not in ('orig', 'p2orig'): continue
        c = lambda bn: (lambda q: f"{q['d']:+.3f} [{q['ci'][0]:+.3f},{q['ci'][1]:+.3f}]")(M['paired'][f'{pr}|{bn}']) if f'{pr}|{bn}' in M['paired'] else '-'
        L.append(f"| {pr} | {c('200-400')} | {c('400-700')} | {c('700-1200')} | {c('1200-2000')} | {c('2000-4000')} | {c('200-2000')} | "
                 f"{M['det'][b]['recall']:.3f} ({M['det'][a]['recall']:.3f}) |")
SAMP = []
if os.path.exists(T + 'gpu_samples.log'):
    for ln in open(T + 'gpu_samples.log'):
        p = ln.split()
        if len(p) >= 2: SAMP.append((float(p[0]), p[1], set(p[2:])))


def valid(B, sec):
    """True if every sampler reading on the bench GPU within the section's epoch window shows no pid but the bench's"""
    if 'epoch' not in sec or 'pid' not in B: return None
    t0, t1 = sec['epoch']; g = str(B.get('cuda_visible'))
    rd = [pids for t, gg, pids in SAMP if gg == g and t0 <= t <= t1 + 2]
    if not rd: return None
    return all(pids <= {str(B['pid'])} for pids in rd)


DSETS = sorted({os.path.basename(f)[6:-5].split('_contaminated')[0] for f in glob.glob(T + 'bench_*.json') if 'smoke' not in f})
for nm in DSETS:
    # pool all tries of this dataset; per section use the first measurement whose own time window was clean (sampler)
    tries = [json.load(open(f)) for f in sorted(glob.glob(T + f'bench_{nm}*.json')) if 'smoke' not in f]
    B = {}; NF = tries[0]['frames']; gpu = tries[0]['gpu']; bad = []
    keys = sorted({k for t in tries for k, v in t.items() if isinstance(v, dict)})
    for k in keys:
        for t in tries:
            if k in t and valid(t, t[k]) is True:
                B[k] = t[k]; break
        else:
            bad.append(k)
    L.append(f'\nThroughput {nm} ({NF} frames, {gpu}; sections without a clean measurement: {bad or "none"}): seconds per chunk')
    L.append('| path | read | detector | head | pattern solve (p1 rows) | p2 detector | p2 head | p2 solve | p1 wall | p2 wall | peak GB |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for v in ('orig', 'fast', 'fastc'):
        p1, so, p2 = B.get(f'{v}_pass1', {}), B.get(f'{v}_solve', {}), B.get(f'{v}_pass2', {})
        if not (p1 or so or p2): continue
        g = lambda d, k: f"{d[k]:.1f}" if k in d else '-'
        L.append(f"| {v} | {g(p1, 'read')} | {g(p1, 'detector')} | {g(p1, 'head')} | {g(so, 'solve')} | {g(p2, 'detector')} | {g(p2, 'head')} | {g(p2, 'solve')} | "
                 f"{g(p1, 'wall')} | {g(p2, 'wall')} | {max(p1.get('peak_gb', 0), so.get('peak_gb', 0), p2.get('peak_gb', 0)):.1f} |")
    for k in [k for k in B if k.startswith('head_')]:
        h = B[k]; d = h['vs_orig']
        L.append(f"  {k}: head {h['head']:.2f}s wall {h['wall']:.2f}s ({h['frames']} frames) | same list {d.get('same_list')} dx max {d.get('dx_max', float('nan')):.1e} dI max {d.get('dI_max', float('nan')):.2f} med {d.get('dI_med', float('nan')):.3f}")
    json.dump(B, open(T + f'clean_{nm}.json', 'w'), indent=1)
print('\n'.join(L))
