"""
Bit-identity check of two output directories (or two files): every array of every .npz that matches the patterns must be
np.array_equal (NaN == NaN), same keys, same dtypes. Used to verify that a refactoring does not change results.

  python tests/compare_npz.py <dir or file A> <dir or file B> [--pat 'pass1_*.npz,pass2_*.npz'] [--skip pattern,claim]

Keys in --skip are not compared (e.g. 'pattern' = the path of the pattern file, which differs between output directories).
Exit code 0 = identical, 1 = differences (printed per file / key with the max abs difference).
"""
import argparse, glob, os, sys
import numpy as np


def compare_files(fa, fb, skip=()):
    A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    ka, kb = set(A.files) - set(skip), set(B.files) - set(skip)
    bad = []
    if ka != kb: bad.append(f'keys differ: only A {sorted(ka - kb)}, only B {sorted(kb - ka)}')
    for k in sorted(ka & kb):
        a, b = A[k], B[k]
        if a.dtype != b.dtype or a.shape != b.shape:
            bad.append(f'{k}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}'); continue
        if a.dtype.kind in 'fc': same = np.array_equal(a, b, equal_nan=True)
        else: same = np.array_equal(a, b)
        if not same:
            d = np.nanmax(np.abs(a.astype(float) - b.astype(float))) if a.dtype.kind in 'fiub' else float('nan')
            n = int(np.sum(a != b)) if a.dtype.kind != 'f' else int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))
            bad.append(f'{k}: {n} of {a.size} differ, max abs diff {d:.3g}')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--pat', default='*.npz')
    ap.add_argument('--skip', default='pattern')
    a = ap.parse_args(argv)
    skip = [s for s in a.skip.split(',') if s]
    if os.path.isfile(a.a): pairs = [(a.a, a.b)]
    else:
        names = sorted({os.path.basename(f) for p in a.pat.split(',') for f in glob.glob(os.path.join(a.a, p))})
        if not names: print('no files match in', a.a); return 1
        pairs = [(os.path.join(a.a, n), os.path.join(a.b, n)) for n in names]
    nbad = 0
    for fa, fb in pairs:
        if not os.path.exists(fb): print(f'MISSING {fb}'); nbad += 1; continue
        bad = compare_files(fa, fb, skip)
        print(('DIFF  ' if bad else 'same  ') + os.path.basename(fa) + ''.join('\n    ' + s for s in bad))
        nbad += bool(bad)
    print(f'{len(pairs) - nbad} of {len(pairs)} files identical')
    return 1 if nbad else 0


if __name__ == '__main__':
    sys.exit(main())
