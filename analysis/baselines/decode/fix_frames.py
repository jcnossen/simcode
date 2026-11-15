"""Frame-label fix of DECODE tables written before 2026-11-15 (no new inference).

Those tables label each row one 6-frame window early: the fork's InferenceDataset uses pad=None, so output frame_ix i is the
centre sum i + 1 (dec_common.process_padded). The fixed copy has frame + 6 (one window of 6-frame sums) and is written NEXT to
the original as locs_fixed.npz (dec_common.FRAME_FIX_FILE); the original is not changed. The first window of each start offset
stays without rows (it was never localized), so a rerun with the fixed inference differs from this copy only there and in the
last window. Readers pick the fixed copy through simcode.dataset.decode_table(<dir>/locs.npz).
usage: python analysis/baselines/decode/fix_frames.py <locs.npz> [<locs.npz> ...] [--shift 6] [--force]"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_LABEL = 'window start (pad same, 2026-11-15 fix)'      # = dec_common.FRAME_LABEL (dec_common imports the decode fork)
FIX_FILE = 'locs_fixed.npz'


def fix(path, shift=6, force=False):
    out = os.path.join(os.path.dirname(os.path.abspath(path)), FIX_FILE)
    if os.path.exists(out) and not force:
        print('exists', out, flush=True); return out
    z = np.load(path, allow_pickle=False)
    info = json.loads(str(z['info'])) if 'info' in z.files and z['info'].ndim == 0 else {}
    if info.get('frame_label') == FRAME_LABEL:
        raise ValueError(f'{path} already has the fixed frame labels')
    R = {k: z[k] for k in z.files if k != 'info'}
    R['frame'] = R['frame'].astype(np.int64) + shift
    info.update(frame_label=FRAME_LABEL, frame_fix=dict(shift=shift, source=os.path.abspath(path),
                note='frames of the source + one window (InferenceDataset pad=None: output i = centre sum i + 1); the first '
                     'window of each start offset has no rows'))
    np.savez_compressed(out + '.tmp.npz', **R, info=json.dumps(info))
    os.replace(out + '.tmp.npz', out)
    print(path, '->', out, 'rows', len(R['frame']), 'frames', int(R['frame'].min()), '-', int(R['frame'].max()), flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('paths', nargs='+'); ap.add_argument('--shift', type=int, default=6)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for p in a.paths: fix(p, a.shift, a.force)
