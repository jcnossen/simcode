"""
One resumable work driver for both pipelines. A "unit" is a piece of work with one output file: a chunk of a movie (2D
pipeline) or a file of a multi-file acquisition (4Pi).

  atomic_savez(path, **arrays)       np.savez into path + '.tmp', then os.replace: a killed run never leaves a truncated
  atomic_save(path, array)           output that looks finished (same for .npy, torch checkpoints, json and text)
  atomic_torch_save(obj, path)
  atomic_json(path, obj, **kw)
  disk_guard(path, min_free_gb)      raises DiskFull BEFORE a unit starts when the target file system is nearly full

  U = Units(out_dir, stage, pattern='{stage}_{key}.npz', claim=False, redo=False, min_free_gb=0.0)
  U.path(key)                        the unit's output file
  U.done(key)                        output exists (complete by construction, because writes are atomic)
  U.claim(key) / U.release(key)      marker '<output>.claim' (O_CREAT | O_EXCL) holding host, pid, process start time;
                                     a claim whose process no longer exists on this host is taken over
  for key in U.todo(keys):           the loop of a worker: yields every unit that is not done (and, with claim=True, that
      ... ; U.save(key, **arrays)    this process could claim); the claim is released when the loop moves on. With
                                     claim=True the loop only ends when ALL units are done (it waits for units claimed
                                     by other workers and takes over those whose worker died), so the code after the
                                     loop can rely on every output (the stage is a barrier).
  U.save(key, **arrays)              atomic_savez of the unit's output
  U.record(key, info)                one small json per unit ('<output without extension>.json'): timing, counts, ...
  U.records(keys)                    {key: info} of the units that have a record: summaries are built by merging these,
                                     so no file is rewritten by two processes
  U.fail(key, err)                   marker '<output>.failed' with the error text; failed units are skipped by todo()
                                     until the marker is deleted or redo is set

Several GPUs = the same command started once per GPU with claiming on (python -m simcode.io.launch --gpus 1,2 -- <command>).
"""
import json, os, shutil, socket, time
import numpy as np


class DiskFull(RuntimeError):
    pass


def _replace(tmp, path):
    os.replace(tmp, path)


def atomic_savez(path, **arrays):
    """np.savez(path, **arrays) through a temporary file in the same directory (path + '.tmp'; no '.npz' suffix, so that
    a leftover of a killed run never matches a '*.npz' glob)"""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    _replace(tmp, path)
    return path


def atomic_save(path, array):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, array)
    _replace(tmp, path)
    return path


def atomic_torch_save(obj, path):
    import torch
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    _replace(tmp, path)
    return path


def atomic_json(path, obj, **kw):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, **kw)
    _replace(tmp, path)
    return path


def atomic_text(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    _replace(tmp, path)
    return path


def free_gb(path):
    d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    return shutil.disk_usage(d).free / 1e9


def disk_guard(path, min_free_gb):
    """raise DiskFull if the file system of `path` has less than min_free_gb free (0 = no check)"""
    if min_free_gb and free_gb(path) < min_free_gb:
        raise DiskFull(f'only {free_gb(path):.1f} GB free on the file system of {path} (need {min_free_gb:g} GB): not starting')


def _proc_start(pid):
    """start time of a process (clock ticks since boot; /proc/<pid>/stat field 22), None if it does not exist"""
    try:
        with open(f'/proc/{pid}/stat') as f:
            return int(f.read().rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _me():
    return dict(host=socket.gethostname(), pid=os.getpid(), start=_proc_start(os.getpid()), t=time.strftime('%Y-%m-%d %H:%M:%S'))


def claim_owner_alive(c):
    """True / False for a claim of this host (the process with that pid and start time exists / does not); None for a
    claim of another host or an unreadable claim (cannot be checked: treated as alive)"""
    if not isinstance(c, dict) or c.get('host') != socket.gethostname() or 'pid' not in c: return None
    st = _proc_start(c['pid'])
    return st is not None and (c.get('start') is None or st == c['start'])


class Units:
    def __init__(self, out_dir, stage, pattern='{stage}_{key}.npz', claim=False, redo=False, min_free_gb=0.0, poll_s=10.0, title=True,
                 log=None, **fmt):
        """pattern: file name of a unit's output, formatted with stage, key and the extra keyword arguments (fmt).
        title: True = the process title is set to 'simcode:<stage> <key>' while a unit is processed; a string replaces the
        stage in it; False = the title is left alone"""
        self.out, self.stage, self.pattern, self.fmt = out_dir, stage, pattern, fmt
        self.use_claim, self.redo, self.min_free_gb, self.poll_s, self.title = bool(claim), bool(redo), min_free_gb, poll_s, title
        self.log = log or (lambda *s: print(*s, flush=True))
        self._redone = set()
        os.makedirs(out_dir, exist_ok=True)

    # ---- names
    def path(self, key):
        return os.path.join(self.out, self.pattern.format(stage=self.stage, key=key, **self.fmt))

    def record_path(self, key):
        return os.path.splitext(self.path(key))[0] + '.json'

    # ---- state
    def done(self, key):
        return os.path.exists(self.path(key)) and not (self.redo and key not in self._redone)

    def failed(self, key):
        return os.path.exists(self.path(key) + '.failed') and not self.redo

    def claimed(self, key):
        """the claim of a unit (dict) if a live worker holds it, else None"""
        try:
            c = json.load(open(self.path(key) + '.claim'))
        except (OSError, ValueError):
            return None if not os.path.exists(self.path(key) + '.claim') else {}
        return None if claim_owner_alive(c) is False else c

    def claim(self, key):
        """try to claim a unit -> True if this process now holds it. A claim left by a dead process of this host is
        taken over (renamed away first, so that only one of several waiting workers gets it)."""
        cf = self.path(key) + '.claim'
        for _ in range(2):
            try:
                fd = os.open(cf, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w') as f: json.dump(_me(), f)
                return True
            except FileExistsError:
                try: c = json.load(open(cf))
                except (OSError, ValueError): c = None
                if c is None:                       # being written by another worker right now, or unreadable
                    time.sleep(0.2)
                    try: c = json.load(open(cf))
                    except (OSError, ValueError): return False
                if c.get('host') == socket.gethostname() and c.get('pid') == os.getpid(): return True
                if claim_owner_alive(c) is not False: return False
                try:
                    os.rename(cf, f'{cf}.stale.{os.getpid()}'); os.remove(f'{cf}.stale.{os.getpid()}')
                    self.log(f'{self.stage} {key}: took over the claim of the dead process {c.get("pid")} ({c.get("t")})')
                except OSError:
                    return False                    # another worker took it over first
        return False

    def release(self, key):
        cf = self.path(key) + '.claim'
        try:
            c = json.load(open(cf))
            if c.get('host') == socket.gethostname() and c.get('pid') == os.getpid(): os.remove(cf)
        except (OSError, ValueError):
            pass

    # ---- the worker loop
    def todo(self, keys, order='fwd'):
        keys = list(keys) if order == 'fwd' else list(keys)[::-1]
        told = False; seen = set()                  # a unit is given to this loop once, also if its output was not written
        while True:
            worked = False
            for key in keys:
                if key in seen or self.done(key) or self.failed(key): continue
                if self.use_claim and not self.claim(key): continue
                if self.done(key):                  # finished by another worker between the test and the claim
                    self.release(key); continue
                try:
                    disk_guard(self.out, self.min_free_gb)
                    if self.title:                  # process title 'simcode:<stage or the given title> <unit>'
                        from simcode.util.proctitle import set_title
                        set_title(self.title if isinstance(self.title, str) else self.stage, key)
                    worked = True; seen.add(key)
                    yield key
                    self._redone.add(key)
                finally:
                    if self.use_claim: self.release(key)
            if not self.use_claim: return
            left = [k for k in keys if k not in seen and not self.done(k) and not self.failed(k)]
            if not left: return
            if not worked:
                if not told:
                    self.log(f'{self.stage}: waiting for {len(left)} unit(s) held by other workers ({left[:6]}{" ..." if len(left) > 6 else ""})')
                    told = True
                time.sleep(self.poll_s)

    def missing(self, keys):
        return [k for k in keys if not os.path.exists(self.path(k))]

    # ---- outputs
    def save(self, key, **arrays):
        return atomic_savez(self.path(key), **arrays)

    def record(self, key, info):
        return atomic_json(self.record_path(key), info, indent=1, default=float)

    def records(self, keys):
        out = {}
        for k in keys:
            try: out[k] = json.load(open(self.record_path(k)))
            except (OSError, ValueError): pass
        return out

    def fail(self, key, err):
        atomic_text(self.path(key) + '.failed', f'{time.strftime("%Y-%m-%d %H:%M:%S")} {err!r}\n')
