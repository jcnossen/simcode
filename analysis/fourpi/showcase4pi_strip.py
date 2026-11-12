"""Remove the candidate patch arrays (img) from showcase front files once the network pass is done (disk budget).
The rest of the front file (candidates, prior, roi_sum) stays for the evaluation; simfront4pi.py --save_img 1 rebuilds it.
Usage: python analysis/fourpi/showcase4pi_strip.py sc_d01 sc_d03 ..."""
import os, sys
import numpy as np
for t in sys.argv[1:]:
    f = f'/home/jc/simcode2/out/4pi/simmovie/{t}/front_SIM_000_v3L_det.npz'
    F = np.load(f)
    if 'img' not in F.files: print(t, 'no img'); continue
    if not all(os.path.exists(f'/home/jc/simcode2/out/4pi/simmovie/{t}/locs_SIM_000_v3L_det_{p}.npz') for p in ('iterv10', 'oraclev10_phiO')):
        print(t, 'network pass not done, kept'); continue
    D = {k: F[k] for k in F.files if k != 'img'}
    np.savez(f[:-4] + '_tmp.npz', **D); os.replace(f[:-4] + '_tmp.npz', f)
    print(t, 'img removed', os.path.getsize(f) // 2**20, 'MB')
