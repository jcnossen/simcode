"""
Static check of the source tree (no GPU, nothing is executed): every tracked .py file must compile, every import of a
package module (simcode.*) must point at an existing module and name, every attribute of an imported package module
(`from simcode.x import y as Y; Y.name`) must be a top-level name of that module, and every import of a sibling analysis
script must resolve to a file. Shell scripts and markdown files are scanned for mentions of .py / .sh files that do not exist.

  python tests/check_imports.py            exit code 0 = clean
Used after removals / renames (run it directly; pytest does not collect this file).
"""
import ast, importlib.util, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = [d for d in ('simcode',) if os.path.isdir(os.path.join(ROOT, d))]


def tracked():
    try:
        out = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, stderr=subprocess.DEVNULL).decode().split('\n')
        fs = [f for f in out if f and os.path.exists(os.path.join(ROOT, f))]
        if fs: return fs
    except Exception:
        pass
    fs = []
    for d, _, names in os.walk(ROOT):
        if any(p in d for p in ('/.git', '/out', '/runs', '/external', '__pycache__')): continue
        fs += [os.path.relpath(os.path.join(d, n), ROOT) for n in names]
    return fs


def names_of(path):
    """top-level names a module defines (defs, classes, assignments, imports)"""
    t = ast.parse(open(path, encoding='utf-8', errors='replace').read())
    out = set()
    def walk(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
            elif isinstance(n, ast.Assign):
                for tg in n.targets:
                    for e in ast.walk(tg):
                        if isinstance(e, ast.Name): out.add(e.id)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name): out.add(n.target.id)
            elif isinstance(n, ast.Import):
                for a in n.names: out.add((a.asname or a.name).split('.')[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names: out.add(a.asname or a.name)
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for fld in ('body', 'orelse', 'finalbody'): walk(getattr(n, fld, []))
                for h in getattr(n, 'handlers', []): walk(h.body)
    walk(t.body)
    return out


def mod_path(dotted):
    p = os.path.join(ROOT, *dotted.split('.'))
    if os.path.isfile(p + '.py'): return p + '.py'
    if os.path.isdir(p): return os.path.join(p, '__init__.py') if os.path.isfile(os.path.join(p, '__init__.py')) else p
    return None


def check():
    files = tracked()
    py = [f for f in files if f.endswith('.py')]
    bybase = {}
    for f in py: bybase.setdefault(os.path.basename(f)[:-3], []).append(f)
    errs = []
    cache = {}
    for f in py:
        src = open(os.path.join(ROOT, f), encoding='utf-8', errors='replace').read()
        try: t = ast.parse(src)
        except SyntaxError as e:
            errs.append(f'{f}: syntax error {e}'); continue
        aliases = {}       # local name -> package module (module objects imported by name)
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname and a.name.split('.')[0] == 'simcode' and (mod_path(a.name) or '').endswith('.py'): aliases[a.asname] = a.name
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.split('.')[0] == 'simcode':
                for a in n.names:
                    p_ = mod_path(n.module + '.' + a.name)
                    if p_ and p_.endswith('.py'): aliases[a.asname or a.name] = n.module + '.' + a.name
        for n in ast.walk(t):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                m = aliases[n.value.id]; p_ = mod_path(m)
                if p_ not in cache: cache[p_] = names_of(p_)
                if n.attr not in cache[p_] and '__getattr__' not in cache[p_] and not mod_path(m + '.' + n.attr) and not n.attr.startswith('__'):
                    errs.append(f'{f}:{n.lineno}: {n.value.id}.{n.attr}: {m} has no name {n.attr}')
        for n in ast.walk(t):
            mods = []
            if isinstance(n, ast.Import): mods = [(a.name, None) for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module: mods = [(n.module, [a.name for a in n.names])]
            for m, names in mods:
                top = m.split('.')[0]
                if top == 'simcode':
                    if top not in PKGS: errs.append(f'{f}:{n.lineno}: import of {m}: package {top} does not exist'); continue
                    p = mod_path(m)
                    if p is None: errs.append(f'{f}:{n.lineno}: module {m} does not exist'); continue
                    if names and p.endswith('.py'):
                        if p not in cache: cache[p] = names_of(p)
                        for nm in names:
                            if nm == '*' or nm in cache[p]: continue
                            if mod_path(m + '.' + nm): continue
                            errs.append(f'{f}:{n.lineno}: {m} has no name {nm}')
                    elif names:
                        for nm in names:
                            if not mod_path(m + '.' + nm): errs.append(f'{f}:{n.lineno}: {m}.{nm} does not exist')
                elif top in bybase or top in ('analysis',) or (top in PAPER_MODULES and not HAS_ANALYSIS):
                    continue     # sibling analysis script (found through sys.path of the script)
                else:
                    try: found = importlib.util.find_spec(top) is not None
                    except Exception: found = False
                    if not found and top not in OPTIONAL: errs.append(f'{f}:{n.lineno}: import {m}: not a file of this repository and not installed')
    # mentions of scripts in shell scripts and docs
    allb = {os.path.basename(f) for f in files}
    for f in files:
        if not f.endswith(('.sh', '.md')) or f in SKIP_MENTIONS or f.startswith('figures/'): continue   # figures/ (paper_figures branch): copies of figure outputs
        for i, line in enumerate(open(os.path.join(ROOT, f), encoding='utf-8', errors='replace'), 1):
            for mm in re.finditer(r'(?<![\w~/:-])((?:simcode|analysis|tests)/[\w/.-]+\.(?:py|sh))\b', line):
                s = mm.group(1)
                if s.startswith('analysis/') and not HAS_ANALYSIS: continue      # a script of the paper_figures branch
                if not os.path.exists(os.path.join(ROOT, s)): errs.append(f'{f}:{i}: mentions {s}, which does not exist')
            for mm in re.finditer(r'python\s+(?:-u\s+)?-m\s+(simcode[\w.]*)', line):
                if mod_path(mm.group(1)) is None: errs.append(f'{f}:{i}: python -m {mm.group(1)}: no such module')
    return errs


# imports that are optional or come from external checkouts (old code base, DME, DECODE, spline PSF, published 4Pi code)
OPTIONAL = {'smlmtorch', 'dme', 'decode', 'torch_convspline', 'h5py', 'setproctitle', 'fastpsf', 'photonpy', 'picasso', 'cupy',
            'joblib', 'sklearn', 'pywt', 'napari', 'zarr', 'tqdm', 'yaml', 'pearson_corr'}
# master has no analysis/ (it is on the paper_figures branch): mentions of analysis/ scripts and the imports of these analysis
# modules (through sys.path of the analysis folder) are not checked there
HAS_ANALYSIS = os.path.isdir(os.path.join(ROOT, 'analysis'))
PAPER_MODULES = {'frc_fair'}
# historical documents: they describe removed code on purpose (mentions written archive-<date>:<path> name a file of that git tag)
SKIP_MENTIONS = {'docs/CLEANUP_LOG.md', 'docs/CLEANUP_CANDIDATES.md', 'docs/ARCHIVE.md', 'docs/RESULTS.md', 'analysis/results/README.md'}


def test_tree():
    errs = check()
    assert not errs, '\n'.join(errs)


if __name__ == '__main__':
    e = check()
    print('\n'.join(e) if e else 'tree ok')
    sys.exit(1 if e else 0)
