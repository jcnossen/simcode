"""
Optional torch.compile support for the fast inference paths (2026-11-10, throughput work).

torch.compile on CUDA needs Triton; PyTorch's Windows wheels ship without it (triton-windows is a separate optional
package). Every fast path therefore runs in eager mode by default-capable code, and compilation is an optional extra:
  mode 'off'  : never compile (eager)
  mode 'on'   : compile; raise if Triton / the compile smoke test fails
  mode 'auto' : compile only if Triton is importable and a tiny compile smoke test succeeds, else eager (one log line)
The compiled and eager functions are the same Python code, so they give the same results up to float rounding.
"""
import os, sys

_STATE = {}


def _fix_ptxas():
    """Triton looks for ptxas in its own backends/nvidia/bin, then on PATH; a conda env run by its full python path (not
    activated) has its ptxas in <env>/bin, which is then not found (and triton 3.7 raises a TypeError instead of falling
    back). If TRITON_PTXAS_PATH is unset and triton ships no ptxas, point it at <env>/bin/ptxas when that exists
    (this process only)."""
    if os.environ.get('TRITON_PTXAS_PATH'):
        return
    try:
        import shutil, triton
        own = os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'bin', 'ptxas')
        if os.path.exists(own) or shutil.which('ptxas'):
            return
        cand = os.path.join(os.path.dirname(sys.executable), 'ptxas')
        if os.path.exists(cand):
            os.environ['TRITON_PTXAS_PATH'] = cand
    except Exception:
        pass


def compile_available(verbose=True):
    """True if torch.compile works here (Triton importable + a tiny CUDA compile smoke test). Cached per process."""
    if 'ok' in _STATE:
        return _STATE['ok']
    ok, why = False, ''
    _fix_ptxas()
    try:
        import torch
        import triton  # noqa: F401  (inductor's CUDA backend)
        if not torch.cuda.is_available():
            raise RuntimeError('no CUDA')
        f = torch.compile(lambda a: (a * 2.0 + 1.0).sin().sum(-1), dynamic=False)
        x = torch.arange(8, device='cuda', dtype=torch.float32).reshape(2, 4)
        y = f(x)
        ref = (x * 2.0 + 1.0).sin().sum(-1)
        ok = bool(torch.allclose(y, ref, atol=1e-5))
        if not ok: why = 'smoke test mismatch'
    except Exception as e:      # ImportError (no Triton), compiler errors, ...
        why = f'{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ""}'
    _STATE['ok'] = ok
    if verbose:
        print(f'[simcode.util.compile] torch.compile {"available" if ok else "unavailable -> eager (" + why + ")"}', file=sys.stderr, flush=True)
    return ok


def resolve(mode):
    """'auto' | 'on' | 'off' (or bool) -> bool use-compile (the pipeline always uses 'auto'; 'on' / 'off' for tests)"""
    if isinstance(mode, bool):
        mode = 'on' if mode else 'off'
    mode = (mode or 'auto').lower()
    if mode == 'off':
        return False
    if mode == 'on':
        if not compile_available():
            raise RuntimeError('compile=on requested but torch.compile/Triton is not usable here')
        return True
    return compile_available()


def maybe_compile(fn, mode='auto', **kw):
    """fn compiled with torch.compile(**kw) if resolve(mode), else fn itself"""
    if not resolve(mode):
        return fn
    import torch
    cf = torch.compile(fn, **kw)
    strict = (mode if isinstance(mode, str) else 'on').lower() == 'on'
    state = {'ok': True}

    def call(*a, **k):
        # a compile / codegen error at the first calls (e.g. a host C compiler problem while Triton builds its launcher)
        # -> eager for the rest of the process (one log line); with mode 'on' the error is raised
        if state['ok']:
            try:
                return cf(*a, **k)
            except Exception as e:
                if strict: raise
                state['ok'] = False
                print(f'[simcode.util.compile] torch.compile failed at run time -> eager ({type(e).__name__})', file=sys.stderr, flush=True)
        return fn(*a, **k)
    call.__wrapped__ = fn
    return call
