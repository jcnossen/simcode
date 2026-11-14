"""nr80 evaluation: correctness gates as analysis/psfmatch/gatecheck.py defines them, on the nr80 evaluation outputs, and the
network-set entry. Reads out/nanoruler/eval/:
  (1) sigma: every held-out line of sigu_test_p{1,2}_nr80.txt PASS (robust zcal of the m12 selection after the rescale in [0.9, 1.15]; ev_sigcal.py)
  (2) GT sanity, pass 2, simcmp_nr_a_p2_nr80_vs_cur.txt (A = nr80, B = the current production nets on the same sim movie):
      GT RMS ratio (200-2000 ph, identical GT windows) <= 1.05, gross > 50 nm <= max(1.5 x current, current + 1 pt),
      recall of the GT target windows >= current - 0.02
  (3) reported only: pair1 / pair2 numbers of accept_nr80.txt (s2 vs c2, s1 vs c1; bands 200-2000 and 800-2500, detection)
Writes out/nanoruler/models/netset_nr80.json (one model in the format of simcode/models.json, absolute paths: usable as simcode.run --model <this file>; with the gate result
in "_gates"); every file it names must exist. usage: python ev_gates.py -> prints the checks; exit 0 iff (1) and (2) pass"""
import json, os, re, sys
O = '/home/jc/simcode2/out/nanoruler/eval'; M = '/home/jc/simcode2/out/nanoruler/models'; S = 'nr80'; ok = True
for p in (1, 2):
    n = 0
    for L in open(f'{O}/sigu_test_p{p}_{S}.txt'):
        if L.startswith('#') or L.startswith('set') or not L.strip(): continue
        f = L.split(); r = f[-1] == 'True'; ok &= r; n += 1
        print(f'(1) p{p} {f[0]:26s} n m12 {f[3]}  zcal m12 {f[4]} -> after rescale {f[5]}  {"PASS" if r else "FAIL"}')
    if n == 0: ok = False; print(f'(1) p{p}: no test line  FAIL')
fn = f'{O}/simcmp_nr_a_p2_{S}_vs_cur.txt'; t = open(fn).read()
rec = [float(x) for x in re.findall(r'recall\(target windows\) ([0-9.]+)', t)]
m = re.search(r'GT ph +200-2000 .*?RMS nm A +([0-9.]+) B +([0-9.]+) ratio ([0-9.]+) +gross>50nm % A +([0-9.]+) B +([0-9.]+)', t)
if m is None or len(rec) != 2:
    ok = False; print(f'(2) {os.path.basename(fn)}: 200-2000 line or recall not found (too few identical GT windows?)  FAIL')
else:
    ra, rb, rat, ga, gb = map(float, m.groups())
    r = rat <= 1.05 and ga <= max(1.5 * gb, gb + 1.0) and rec[0] >= rec[1] - 0.02; ok &= r
    print(f'(2) {os.path.basename(fn)}: RMS {ra:.2f} vs {rb:.2f} nm (ratio {rat:.3f}), gross {ga:.2f} vs {gb:.2f} %, recall {rec[0]:.4f} vs {rec[1]:.4f}  {"PASS" if r else "FAIL"}')
for L in t.splitlines():
    if '800-2500' in L: print('(2, reported) ' + ' '.join(L.split()))
cur = None
for L in open(f'{O}/accept_{S}.txt'):
    if L.startswith('=='): cur = L.strip()
    if cur and ('pair' in cur) and ('s2 (A) vs c2' in cur or 's1 (A) vs c1' in cur) and ('200-2000' in L or '800-2500' in L or 'detect' in L):
        print(f'(3) {cur} | {" ".join(L.split())[:170]}')
E_ = dict(_doc='PSF-matched set nr80 for the nanoruler movie of the original SIMFLUX setup (65 nm px, pitch 3.38 px; calib out/nanoruler/calib_nr80.json; '
               'm1809 recipe + --dens 0.0005,0.06, --tau 3,80, trainers through analysis/nanoruler/nrwrap.py). Absolute paths. cal_real / p2_cal_real: '
               'the real-data sigma multiplier is COPIED from the other setup (not re-measured). sig_u fitted on cycle-aligned windows of the '
               'simulated nanoruler movie nr_a frames [0, 6000) (analysis/nanoruler/ev_sigcal.py). Runs of this setup must go through '
               'analysis/nanoruler/ev_run.py (pitch search range, crop).',
          _gates='PASS' if ok else 'FAIL',
          det=f'{M}/det_{S}.pt', head=f'{M}/hd_{S}.pt', cal_sim=f'{M}/calib/calib_sc_hd_{S}.json', cal_real=f'{M}/calib/calib_sc_hd_{S}_real.json',
          p2_det=f'{M}/p2det_{S}.pt', p2_head=f'{M}/p2headL_{S}.pt', p2_cal_sim=f'{M}/calib/calib_p2L_{S}.json',
          p2_cal_real=f'{M}/calib/calib_p2L_{S}_real.json', sig_u=f'{M}/calib/sigu_p1_{S}.json,{M}/calib/sigu_p2_{S}.json')
miss = [p for k, v in E_.items() if not k.startswith('_') for p in v.split(',') if not os.path.exists(p)]
if miss: print('model: MISSING files', miss); ok = False
else:
    json.dump({S: E_}, open(f'{M}/netset_{S}.json', 'w'), indent=1); print(f'wrote {M}/netset_{S}.json (_gates {E_["_gates"]})')
print('GATES', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
