"""Reverse the phase-step order of a cached (reordered) nanoruler movie without touching the pixels.

The cached movies store every 6-frame acquisition cycle as [0, 2, 4, 1, 3, 5] (stored step s = frame % 6, orientation s // 3).
On this setup the stored pattern then steps BACKWARD in phase (-2.2, -2.1 | -1.9, -1.9 rad), the nr80 pass-2 networks were
trained on forward steps. This tool writes the movie with the three steps of each orientation in reversed order:
    new stored frame 6 c + j  =  old stored frame 6 c + FLIP[j],  FLIP = [2, 1, 0, 5, 4, 3]
    (= acquisition order [4, 2, 0, 5, 3, 1] within the cycle)
so the stored pattern steps forward. A cycle-aligned window (start % 6 == 0) holds exactly the same six acquisition frames as
before; windows starting at 6 c + 3 hold acquisition frames {5, 3, 1} of cycle c and {4, 2, 0} of cycle c + 1.
Streaming (one cycle in memory), BigTIFF when the input is BigTIFF or larger than 3.9 GB, written to <out>.tmp and renamed after
the verification (frame count; --verify random cycles compared page by page with the input). Idle I/O priority (ionice -c 3)
and a write-rate limit (--rate MB/s, default 60). A sidecar <out>.json records the permutation.
  python ev_flipsteps.py movie <in.tif> <out.tif> [--rate 60] [--verify 40]
  python ev_flipsteps.py gt <sim_gt.npz> <out sim_gt.npz>
      ground truth of a simulated movie (sim_nr.py) for the flipped movie: the per-STORED-frame arrays patt, drift, bg_frame
      and the stored frame index fr_frame are permuted the same way; everything in acquisition time (t0, t1, drift_acq,
      dphi_acq, fr_acq) and per cycle (ph6) is unchanged; key stored_perm = [4, 2, 0, 5, 3, 1] (stored position -> acquisition
      index; ev_sigcal.gt_windows sets sim_nr_lib.PERM from it, so on-fractions per stored frame are exact)."""
import argparse, json, os, subprocess, sys, time
import numpy as np
FLIP = np.array([2, 1, 0, 5, 4, 3])
PERM0 = np.array([0, 2, 4, 1, 3, 5])


def movie(a):
    import tifffile
    sys.path.insert(0, '/home/jc/simcode2')
    try:
        from simcode.util.proctitle import set_title; set_title('nr-flip')
    except Exception: pass
    subprocess.call(['ionice', '-c', '3', '-p', str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert os.path.abspath(a.inp) != os.path.abspath(a.out) and not os.path.exists(a.out), 'output exists (or equals the input)'
    ti = tifffile.TiffFile(a.inp); N = len(ti.pages); assert N % 6 == 0, f'{N} frames: not a multiple of 6'
    p0 = ti.pages[0]; big = bool(ti.is_bigtiff) or os.path.getsize(a.inp) > 3.9e9
    print(f'{a.inp}: {N} frames {p0.shape} {p0.dtype}, bigtiff out {big}', flush=True)
    tmp = a.out + '.tmp'; t0 = time.time(); nbytes = 0
    with tifffile.TiffWriter(tmp, bigtiff=big) as tw:
        for c in range(N // 6):
            fr = [ti.pages[6 * c + j].asarray() for j in range(6)]
            for j in range(6):
                tw.write(fr[FLIP[j]], contiguous=True); nbytes += fr[0].nbytes
            if a.rate > 0:
                dt = nbytes / (a.rate * 1e6) - (time.time() - t0)
                if dt > 0: time.sleep(dt)
            if c % 1000 == 0: print(f'cycle {c} / {N // 6}, {nbytes / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s', flush=True)
    to = tifffile.TiffFile(tmp); assert len(to.pages) == N, f'output has {len(to.pages)} frames'
    rng = np.random.default_rng(0); cyc = np.unique(np.r_[0, N // 6 - 1, rng.integers(0, N // 6, a.verify)])
    for c in cyc:
        for j in range(6):
            assert np.array_equal(to.pages[int(6 * c + j)].asarray(), ti.pages[int(6 * c + FLIP[j])].asarray()), f'verification failed at cycle {c} frame {j}'
    to.close(); os.replace(tmp, a.out)
    json.dump(dict(source=os.path.abspath(a.inp), flip=FLIP.tolist(), stored_perm=PERM0[FLIP].tolist(), n_frames=N,
                   note='new stored frame 6c+j = source stored frame 6c+flip[j]; stored position j = acquisition index stored_perm[j] of the cycle '
                        '(analysis/nanoruler/ev_flipsteps.py); pattern phase steps are forward in this file'), open(a.out + '.json', 'w'), indent=1)
    print(f'wrote {a.out}: {N} frames, verified {len(cyc)} cycles, {time.time() - t0:.0f} s', flush=True)


def gt(a):
    G = dict(np.load(a.inp, allow_pickle=True)); NF = int(G['nf']); assert NF % 6 == 0 and 'stored_perm' not in G
    idx = (6 * (np.arange(NF) // 6) + FLIP[np.arange(NF) % 6])           # new stored frame -> old stored frame
    for k in ('patt', 'drift', 'bg_frame'):
        if k in G and np.ndim(G[k]) and len(G[k]) == NF: G[k] = G[k][idx]
    if 'fr_frame' in G: G['fr_frame'] = 6 * (G['fr_frame'] // 6) + FLIP[G['fr_frame'] % 6]      # FLIP is an involution
    G['stored_perm'] = PERM0[FLIP]
    np.savez_compressed(a.out, **G); print('wrote', a.out, 'stored_perm', G['stored_perm'].tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['movie', 'gt']); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--rate', type=float, default=60.0); ap.add_argument('--verify', type=int, default=40)
    a = ap.parse_args(); (movie if a.mode == 'movie' else gt)(a)
