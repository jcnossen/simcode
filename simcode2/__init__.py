"""Compatibility alias (2026-11-13): the package was renamed simcode2 -> simcode.

`import simcode2.x.y` and `python -m simcode2.x` resolve to the SAME module objects as `simcode.x.y` (so monkeypatches of
either name apply to both). Kept for older analysis / run scripts; new code imports `simcode`. Remove when no script uses
the old name (tests/check_imports.py lists them)."""
import sys
import importlib
import importlib.abc
import importlib.util

_OLD, _NEW = 'simcode2', 'simcode'


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(_OLD + '.'):
            new = _NEW + name[len(_OLD):]
            real = importlib.util.find_spec(new)
            if real is None:
                return None
            spec = importlib.util.spec_from_loader(name, self, is_package=real.submodule_search_locations is not None)
            spec.origin = real.origin
            return spec
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        pass

    # runpy (python -m simcode2.x) needs the code object of the real module
    def get_code(self, fullname):
        real = importlib.util.find_spec(_NEW + fullname[len(_OLD):])
        return real.loader.get_code(real.name)

    def is_package(self, fullname):
        real = importlib.util.find_spec(_NEW + fullname[len(_OLD):])
        return real.submodule_search_locations is not None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
import simcode as _pkg  # noqa: E402
sys.modules[_OLD] = _pkg
