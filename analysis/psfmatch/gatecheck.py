"""PSF-matched retrain (2026-11-11, user decision 20:50): correctness gates before production of a matched set S.
  (1) sigma: every held-out line of out/psfmatch/sigu_test_p{1,2}_<S>.txt PASS (strict robust zcal after the rescale in [0.9, 1.15])
  (2) GT sanity on the set's sim movies vs the earlier run with the current nets (out/psfmatch/simcmp_<m>_p<p>_<S>_vs_cur.txt), pass 2:
      GT RMS ratio (200-2000 ph, identical GT windows) <= 1.05, gross > 50 nm (200-2000) <= max(1.5 x current, current + 1 pt),
      recall of GT target windows >= current - 0.02
  (3) reported only: pass-2 close-pair numbers on the own gates (accept_<tag>.txt, s2 vs c2)
usage: python gatecheck.py <S>   -> prints the checks, exit 0 iff (1) and (2) pass"""
import re, sys, glob
S = sys.argv[1]; O = 'out/psfmatch'; ok = True
for p in (1, 2):
    for L in open(f'{O}/sigu_test_p{p}_{S}.txt'):
        if L.startswith('#') or L.startswith('set'): continue
        f = L.split(); r = f[-1] == 'True'; ok &= r
        print(f'(1) p{p} {f[0]:26s} zcal strict {f[4]} -> after rescale {f[5]}  {"PASS" if r else "FAIL"}')
for fn in sorted(glob.glob(f'{O}/simcmp_*_p2_{S}_vs_cur.txt')):
    t = open(fn).read()
    rec = [float(x) for x in re.findall(r'recall\(target windows\) ([0-9.]+)', t)]
    m = re.search(r'GT ph +200-2000 .*?RMS nm A +([0-9.]+) B +([0-9.]+) ratio ([0-9.]+) +gross>50nm % A +([0-9.]+) B +([0-9.]+)', t)
    ra, rb, rat, ga, gb = map(float, m.groups())
    r = rat <= 1.05 and ga <= max(1.5 * gb, gb + 1.0) and rec[0] >= rec[1] - 0.02; ok &= r
    print(f'(2) {fn.split("/")[-1]}: RMS {ra:.2f} vs {rb:.2f} nm (ratio {rat:.3f}), gross {ga:.2f} vs {gb:.2f} %, recall {rec[0]:.4f} vs {rec[1]:.4f}  {"PASS" if r else "FAIL"}')
tag = {'m1809': 'm1809a', 'm25': 'm25a'}.get(S, 'm25a'); cur = None
for L in open(f'{O}/accept_{tag}.txt'):
    if L.startswith('=='): cur = L.strip()
    if cur and ('pair' in cur) and ('s2 (A) vs c2' in cur or 's1 (A) vs c1' in cur) and ('200-2000' in L or 'detect' in L):
        print(f'(3) {cur} | {" ".join(L.split())[:170]}')
print('GATES', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
