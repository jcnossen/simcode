"""Recognizable process names for long jobs (shown by nvidia-smi, ps, top), e.g. 'simcode:pass1 6000' (stage and unit; set
per unit by simcode/io/work.Units) or 'simcode:run' (set once by an entry script).
Short on purpose: at most MAXLEN characters, no command-line arguments.
Needs the optional package `setproctitle`; without it this is a no-op, so no job depends on it."""
MAXLEN = 32


def set_title(name, detail=''):
    """Set the process title to 'simcode:<name>' (plus ' <detail>' if it fits), cut to MAXLEN characters. Never raises."""
    try:
        import setproctitle
        setproctitle.setproctitle(('simcode:%s %s' % (name, detail)).strip()[:MAXLEN].rstrip())
    except Exception:
        pass
    return name


def auto(name):
    """= set_title(name). Kept only for the scripts of the open branches decodev3 / wsm3d that call it (analysis/baselines/decode/
    train_v3.py, analysis/decode_checks/v3/eval_watch.py, analysis/papersim/common_rule.py); remove once they are merged and use
    set_title. Nothing on main calls it (2026-11-15)."""
    return set_title(name)
