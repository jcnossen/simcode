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
    """Set the title 'simcode:<name>' (kept for the existing call sites; the arguments are no longer appended)."""
    return set_title(name)
