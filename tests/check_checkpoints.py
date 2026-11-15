"""
Checkpoint check (CPU, no model is built): for every *.pt under the given directories list the python globals its pickle
refers to (torch zip format: the GLOBAL / STACK_GLOBAL opcodes of data.pkl) and the 'arch' of its args. A checkpoint that
refers to a module of this package by name (simcode*.*) breaks when the package is renamed; state
dicts + plain args dicts refer only to torch / collections / numpy.

  python tests/check_checkpoints.py <dir> [<dir> ...]      exit code 1 if a checkpoint refers to simcode*.*
"""
import glob, os, pickletools, sys, zipfile


def globals_of(path):
    out = set()
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return None
    for n in z.namelist():
        if not n.endswith('.pkl'): continue
        strs = []
        for op, arg, _ in pickletools.genops(z.read(n)):
            if op.name == 'GLOBAL': out.add(arg.replace(' ', '.'))
            elif op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'): strs.append(arg)
            elif op.name == 'STACK_GLOBAL' and len(strs) >= 2: out.add(strs[-2] + '.' + strs[-1])
    return out


def main(dirs):
    bad = 0; n = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '**', '*.pt'), recursive=True)):
            g = globals_of(f); n += 1
            if g is None: print(f'{f}: not a torch zip checkpoint (skipped)'); continue
            mine = lambda x: x.split('.')[0].startswith('simcode') or x.split('.')[0] == '__main__'
            own = sorted(x for x in g if mine(x))
            rest = sorted({x.split('.')[0] for x in g if not mine(x)})
            if own: bad += 1
            print(f'{"REFERS " if own else "ok     "}{os.path.relpath(f, d)}: modules {rest}' + (f' OWN {own}' if own else ''))
    print(f'{n} checkpoints, {bad} refer to package classes')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
