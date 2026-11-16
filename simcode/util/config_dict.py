"""
ConfigDict: a nested settings dictionary with attribute access, for defining runs in Python code (the v1 config_dict,
with errors instead of silent None).

  from simcode.util.config_dict import ConfigDict
  base = ConfigDict(optics=dict(pixel_um=0.10833, NA=1.49), camera=dict(offset_adu=99.3), gpus=[1, 2])
  base.optics.NA, base['optics']['NA'], base['optics.NA']     attribute, item and dotted-path access (all the same value)
  base.optics.na                                               AttributeError naming the key and the close matches
  base.get('hints'), base.get('optics.na', 1.4)                .get never raises
  cfg = base.updated(name='a', psf=dict(iters=300))            a NEW object; nested dicts are merged key by key, every
  cfg = base.updated({'psf.iters': 300, 'gpus': [0]})          other value (lists too) is replaced; dotted keys allowed
  cfg.psf.iters = 300; cfg['spots.thr'] = 5.0                  in-place edits (not on a frozen object)
  base.freeze()                                                read-only from here on (recursive); .clone() is editable again
  cfg.to_dict()                                                plain dicts / lists / numbers: numpy arrays and scalars,
                                                               torch tensors, tuples, sets and paths converted
  cfg.save('x.yaml' | 'x.json'); ConfigDict.load('x.yaml' | 'x.json' | 'x.py')
  load_python('x.py') -> [ConfigDict, ...]                     a Python file that defines `config` or a list `configs`

Nested dicts (also inside lists / tuples) become ConfigDict objects. Keys are strings without a dot. A key with the
name of a method (keys, items, get, update, ...) is only reachable as cfg['keys'].
No torch import here; the schema of the automatic pipeline is in simcode/pipeline/acquisition_config.py.
"""
import copy, difflib, json, os
from collections.abc import Mapping, MutableMapping

__all__ = ['ConfigDict', 'load_python']


def _wrap(v):
    if isinstance(v, ConfigDict): return v.clone()
    if isinstance(v, Mapping): return ConfigDict(v)
    if isinstance(v, list): return [_wrap(x) for x in v]
    if isinstance(v, tuple): return tuple(_wrap(x) for x in v)
    return v


def _plain(v):
    """value -> plain python (dict / list / str / number / bool / None), for yaml and json"""
    if isinstance(v, Mapping): return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_plain(x) for x in v]
    if isinstance(v, (set, frozenset)): return sorted(_plain(x) for x in v)
    if isinstance(v, os.PathLike): return os.fspath(v)
    if hasattr(v, 'detach') and hasattr(v, 'tolist'): return v.detach().cpu().tolist()   # torch tensor
    if hasattr(v, 'tolist') and hasattr(v, 'dtype'): return v.tolist()                    # numpy array / scalar
    return v


class ConfigDict(MutableMapping):
    __slots__ = ('_d', '_frozen')

    def __init__(self, d=None, **kwargs):
        object.__setattr__(self, '_d', {})
        object.__setattr__(self, '_frozen', False)
        for src in (d, kwargs):
            if src is None: continue
            if not isinstance(src, Mapping): raise TypeError(f'ConfigDict: expected a mapping, got {type(src).__name__}')
            for k, v in src.items(): self[k] = v

    # ---- access
    def _missing(self, key, exc):
        low = {}
        for k in self._d: low.setdefault(k.lower(), k)
        near = [low[m] for m in difflib.get_close_matches(str(key).lower(), list(low), n=3, cutoff=0.5)]
        hint = f'; did you mean {", ".join(map(repr, near))}?' if near else ''
        return exc(f'no config key {key!r}{hint} (keys: {", ".join(self._d) or "none"})')

    def __getitem__(self, key):
        if key in self._d: return self._d[key]
        if isinstance(key, str) and '.' in key:
            head, rest = key.split('.', 1)
            if head in self._d:
                sub = self._d[head]
                if not isinstance(sub, ConfigDict): raise KeyError(f'config key {head!r} is not a section (path {key!r})')
                return sub[rest]
            key = head
        raise self._missing(key, KeyError)

    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)        # copy / pickle protocol probes
        d = object.__getattribute__(self, '_d')
        if name in d: return d[name]
        raise self._missing(name, AttributeError)

    def __setitem__(self, key, val):
        if self._frozen: raise TypeError(f'this ConfigDict is frozen (setting {key!r}); use .updated(...) or .clone()')
        if not isinstance(key, str) or not key: raise TypeError(f'config keys are non-empty strings, got {key!r}')
        if '.' in key:
            head, rest = key.split('.', 1)
            if head not in self._d: self._d[head] = ConfigDict()
            if not isinstance(self._d[head], ConfigDict): raise KeyError(f'config key {head!r} is not a section (path {key!r})')
            self._d[head][rest] = val
        else:
            self._d[key] = _wrap(val)

    def __setattr__(self, name, val):
        if name in ConfigDict.__slots__ or hasattr(type(self), name):
            raise AttributeError(f'{name!r} is a ConfigDict method / internal name; use cfg[{name!r}] = ...')
        self[name] = val

    def __delitem__(self, key):
        if self._frozen: raise TypeError(f'this ConfigDict is frozen (deleting {key!r})')
        if key not in self._d and isinstance(key, str) and '.' in key:
            head, rest = key.rsplit('.', 1)
            del self[head][rest]; return
        if key not in self._d: raise self._missing(key, KeyError)
        del self._d[key]

    def __delattr__(self, name):
        try: del self[name]
        except KeyError as e: raise AttributeError(*e.args) from None

    def __contains__(self, key):
        if key in self._d: return True
        if isinstance(key, str) and '.' in key:
            head, rest = key.split('.', 1)
            return isinstance(self._d.get(head), ConfigDict) and rest in self._d[head]
        return False

    def get(self, key, default=None):
        """value of a key or dotted path; default when missing (never raises)"""
        try: return self[key]
        except KeyError: return default

    def __iter__(self): return iter(self._d)
    def __len__(self): return len(self._d)
    def __dir__(self): return list(super().__dir__()) + list(self._d)
    def __eq__(self, other):
        return isinstance(other, Mapping) and self.to_dict() == _plain(other)
    __hash__ = None

    # ---- copies, merge
    def clone(self):
        """deep, editable copy"""
        return ConfigDict({k: copy.deepcopy(v) if not isinstance(v, (ConfigDict, list, tuple)) else v for k, v in self._d.items()})

    def __copy__(self): return self.clone()
    def __deepcopy__(self, memo): return self.clone()
    def __getstate__(self): return (dict(self._d), self._frozen)
    def __setstate__(self, s):
        object.__setattr__(self, '_d', s[0]); object.__setattr__(self, '_frozen', s[1])

    def merge(self, *others, **kwargs):
        """in-place deep merge: sections (nested dicts) are merged key by key, other values replaced; dotted keys allowed.
        A section given as None, or a non-dict value for an existing section, replaces the section. Returns self."""
        for u in others + (kwargs,):
            if u is None: continue
            if not isinstance(u, Mapping): raise TypeError(f'ConfigDict.merge: expected a mapping, got {type(u).__name__}')
            for k, v in u.items():
                cur = self.get(k)
                if isinstance(cur, ConfigDict) and isinstance(v, Mapping): cur.merge(v)
                else: self[k] = v
        return self

    def updated(self, *others, **kwargs):
        """a new ConfigDict = this one with the given values merged in (see merge); self is not changed"""
        return self.clone().merge(*others, **kwargs)

    def replaced(self, **kwargs):
        """a new ConfigDict with the given top-level keys replaced as a whole (no merging into existing sections)"""
        r = self.clone()
        for k, v in kwargs.items(): r[k] = v
        return r

    def freeze(self):
        """make this object and its sections read-only; returns self"""
        def fz(v):
            if isinstance(v, ConfigDict): v.freeze()
            elif isinstance(v, (list, tuple)):
                for x in v: fz(x)
        for v in self._d.values(): fz(v)
        object.__setattr__(self, '_frozen', True)
        return self

    @property
    def frozen(self): return self._frozen

    # ---- plain data, files
    def to_dict(self):
        """plain nested dict: numpy arrays / scalars, torch tensors, tuples and sets -> lists / numbers, paths -> str"""
        return _plain(self._d)

    @staticmethod
    def from_dict(d): return ConfigDict(d)

    def to_yaml(self):
        import yaml
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None, width=120)

    def to_json(self, indent=1):
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path):
        """write .yaml / .yml or .json (by extension)"""
        path = os.fspath(path)
        if path.endswith(('.yaml', '.yml')): txt = self.to_yaml()
        elif path.endswith('.json'): txt = self.to_json() + '\n'
        else: raise ValueError(f'ConfigDict.save: extension of {path!r} must be .yaml, .yml or .json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f: f.write(txt)
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        """read .yaml / .yml, .json, or a Python file that defines exactly one config (load_python for several)"""
        path = os.fspath(path)
        if path.endswith('.py'):
            L = load_python(path)
            if len(L) != 1: raise ValueError(f'{path} defines {len(L)} configs; use load_python(path) for the list')
            return L[0]
        with open(path) as f: txt = f.read()
        if path.endswith(('.yaml', '.yml')):
            import yaml
            d = yaml.safe_load(txt)
        elif path.endswith('.json'): d = json.loads(txt)
        else: raise ValueError(f'ConfigDict.load: extension of {path!r} must be .yaml, .yml, .json or .py')
        if not isinstance(d, Mapping): raise ValueError(f'{path}: the top level must be a mapping')
        return ConfigDict(d)

    def __repr__(self):
        return 'ConfigDict(\n' + self._fmt(1) + ')' if self._d else 'ConfigDict()'

    def _fmt(self, lvl):
        pad = '  ' * lvl; s = ''
        for k, v in self._d.items():
            if isinstance(v, ConfigDict) and len(v): s += f'{pad}{k}:\n' + v._fmt(lvl + 1)
            else: s += f'{pad}{k}: {_plain(v)!r}\n'
        return s

    def _repr_pretty_(self, p, cycle): p.text(repr(self))        # IPython / Jupyter


def load_python(path, names=('configs', 'config')):
    """Run a Python config file and return its configs as a list of ConfigDict: the file defines `configs` (a list / tuple
    of ConfigDict or dict, or a dict name -> config) or one `config`. The file runs as ordinary Python (imports, loops,
    derived values) with __file__ set; it is not imported as a module, so `if __name__ == '__main__':` blocks do not run."""
    import runpy
    path = os.path.abspath(os.fspath(path))
    ns = runpy.run_path(path, run_name='__simcode_config__')
    for n in names:
        if ns.get(n) is None: continue
        v = ns[n]
        if isinstance(v, Mapping) and n == 'configs':
            v = [ConfigDict(c).updated(name=k) if 'name' not in c else c for k, c in v.items()]
        elif isinstance(v, Mapping): v = [v]
        elif not isinstance(v, (list, tuple)): raise TypeError(f'{path}: `{n}` must be a ConfigDict / dict or a list of them')
        out = []
        for i, c in enumerate(v):
            if not isinstance(c, Mapping): raise TypeError(f'{path}: `{n}`[{i}] is a {type(c).__name__}, expected ConfigDict / dict')
            out.append(c.clone() if isinstance(c, ConfigDict) else ConfigDict(c))
        if not out: raise ValueError(f'{path}: `{n}` is empty')
        return out
    raise ValueError(f'{path}: a Python config file must define `config` (ConfigDict or dict) or a list `configs`')
