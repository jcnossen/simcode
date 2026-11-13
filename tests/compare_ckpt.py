"""
Bit-identity of two checkpoints (or all <stage>/final.pt of two directories): every tensor of the state dict must be equal.
  python tests/compare_ckpt.py <a.pt | dir> <b.pt | dir>
Used to verify that a change of a trainer does not change training (fixed seed, a few steps, same GPU model).
"""
import glob, os, sys
import torch


def cmp(fa, fb):
    A = torch.load(fa, map_location='cpu', weights_only=False); B = torch.load(fb, map_location='cpu', weights_only=False)
    bad = []
    for key in ('model', 'state'):
        if key not in A: continue
        sa, sb = A[key], B[key]
        if set(sa) != set(sb): bad.append(f'{key}: different parameter names')
        for k in sorted(set(sa) & set(sb)):
            if not torch.equal(sa[k], sb[k]):
                bad.append(f'{key}.{k}: max abs diff {float((sa[k].float() - sb[k].float()).abs().max()):.3g}')
    return bad


def main(a, b):
    pairs = [(a, b)] if os.path.isfile(a) else [(f, os.path.join(b, os.path.relpath(f, a))) for f in sorted(glob.glob(os.path.join(a, '*', 'final.pt')))]
    nbad = 0
    for fa, fb in pairs:
        if not os.path.exists(fb): print('MISSING', fb); nbad += 1; continue
        bad = cmp(fa, fb); nbad += bool(bad)
        print(('DIFF  ' if bad else 'same  ') + fa + ''.join('\n    ' + s for s in bad[:5]) + (f'\n    ... {len(bad)} tensors differ' if len(bad) > 5 else ''))
    print(f'{len(pairs) - nbad} of {len(pairs)} checkpoints identical')
    return 1 if nbad or not pairs else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
