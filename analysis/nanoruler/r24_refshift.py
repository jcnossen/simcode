"""R24: one-off repair of tables converted by r24_tab.py BEFORE the common-frame rule existed: adds the constant (mean of the published total
SIMFLUX drift - mean of the own drift) to <name>_drift.npy and subtracts it from the table positions. Marks the json ('const_added_px').
usage: python r24_refshift.py <mv> <table.npz> [...]"""
import json, sys, os
import numpy as np
mv = sys.argv[1]; rd = np.load(f'/home/jc/simcode2/out/nanoruler/{mv}/ref/drift_pub_total_simflux.npy')
for fn in sys.argv[2:]:
    b = os.path.splitext(fn)[0]; J = json.load(open(b + '_drift.json'))
    if 'const_added_px' in J or J.get('drift') != 'own': print(fn, 'skipped'); continue
    d = np.load(b + '_drift.npy'); c = rd[:len(d)].mean(0) - d.mean(0); T = dict(np.load(fn)); T['x_nm'] = T['x_nm'] - c[0] * 65; T['y_nm'] = T['y_nm'] - c[1] * 65
    np.savez_compressed(fn, **T); np.save(b + '_drift.npy', d + c); J['const_added_px'] = c.tolist(); J['ref_drift'] = 'drift_pub_total_simflux.npy (r24_refshift.py)'; json.dump(J, open(b + '_drift.json', 'w'), indent=1)
    print(fn, 'shifted by %.2f / %.2f nm' % (-c[0] * 65, -c[1] * 65))
