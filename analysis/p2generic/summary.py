"""Generic pass 2 (2026-11-13): one-table summary of a variant's evaluation (eval_gen.sh outputs): relative RMS/CRLB difference at
200-2000 ph with 95 % CI, (a) standard and (b) real-1809_2-pattern gates vs m1809 pass 2, (d) OOD vs pass 1, (c) m05r3 sim movie.
usage: python summary.py <variant: g1809 | g2c | ...>"""
import json, os, sys, re
v = sys.argv[1]; P = '/home/jc/simcode2/out/p2generic'; GN = 'g2' if v == 'g1809' else v; SX = '' if v == 'g1809' else '_' + v
def row(f, ref):
    if not os.path.exists(f): return '(missing)'
    J = json.load(open(f)); out = []
    for k, r in J.items():
        sc, pr = k.split('|')
        if pr != f'{GN}-{ref}': continue
        x = r['paired'].get('200-2000')
        if x is None: continue
        B = x['B']['rms']; out.append(f'{sc} {100 * x["d"] / B:+.1f} % [{100 * x["ci"][0] / B:+.1f}, {100 * x["ci"][1] / B:+.1f}]')
    return '; '.join(out)
print(f'## {v}: RMS/CRLB 200-2000 ph, relative difference')
print('(a) vs m1809 pass 2:', row(f'{P}/eval_a{SX}.json', 's2'))
print('(a) vs pass 1:      ', row(f'{P}/eval_a{SX}.json', 's1'))
print('(b) vs m1809 pass 2:', row(f'{P}/eval_realpat_1809_2{SX}.json', 's2'))
for o in ('ood_nr', 'ood_p3s4', 'ood_rph'):
    print(f'(d) {o} vs pass 1:', row(f'{P}/eval_{o}{SX}.json', 's1'))
f = f'{P}/simcmp_m05r3_p2_{v}_vs_m1809.txt'
if os.path.exists(f):
    t = open(f).read()
    for key in ('recall A - B', 'GT ph 200-2000 ', 'fully on, 200-2000', 'switching, 200-2000'):
        m = [l.strip() for l in t.splitlines() if key in l]
        if m: print('(c)', re.sub(r'\s+', ' ', m[0]))
f = f'{P}/p12_m05r3_{v}.md'
if os.path.exists(f):
    m = [l for l in open(f) if l.startswith('| good |') or l.startswith('| good, 200')]
    print('(c) p2 vs p1:', ' / '.join(re.sub(r'\s+', ' ', x.strip()) for x in m))
