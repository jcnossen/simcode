"""Atomic checkpoint save with a disk guard, used by every trainer (2026-11-12: the root disk filled to 0 and a checkpoint
was destroyed while being overwritten). save(obj, path): writes to path + '.tmp' and renames over path only after a
complete write; skips the save with a warning when the target file system has less than `min_free_gb` free, so the
previous checkpoint survives. Training itself is not interrupted (a trainer keeps running and saves at its next step)."""
import os, shutil, sys
import torch


def save(obj, path, min_free_gb=10.0):
    d = os.path.dirname(os.path.abspath(path))
    free = shutil.disk_usage(d).free / 1e9
    if free < min_free_gb:
        print(f'WARNING: only {free:.1f} GB free on {d}; checkpoint {path} NOT written (previous one kept)', file=sys.stderr, flush=True)
        return False
    tmp = path + '.tmp'
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f'WARNING: checkpoint {path} failed ({e}); previous one kept', file=sys.stderr, flush=True)
        try: os.remove(tmp)
        except OSError: pass
        return False
