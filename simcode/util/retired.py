"""
Command-line options that were removed because their production value became the only behaviour (options clean-up
2026-11-14, docs/CLEANUP_LOG.md). Older scripts may still pass them: strip() drops such an option when it is given with
its fixed value and stops with an error naming the option otherwise, so no run silently changes meaning.

  table: {'--name': fixed value as a string, or None for a former store_true flag that is now always on}
"""
import sys


def strip(argv, table, prog=''):
    argv = list(sys.argv[1:] if argv is None else argv)
    out, i, dropped = [], 0, []
    while i < len(argv):
        tok = argv[i]
        name, eq, val = tok.partition('=')
        if name in table:
            fixed = table[name]
            if fixed is None:                                    # former flag (store_true), now always on
                dropped.append(name); i += 1; continue
            if not eq:
                if i + 1 >= len(argv): raise SystemExit(f'{prog}: {name} needs a value')
                val = argv[i + 1]; i += 1
            if not _same(val, fixed):
                raise SystemExit(f'{prog}: option {name} was removed (2026-11-14); its value is fixed to {fixed!r}, got {val!r} '
                                 '(docs/CLEANUP_LOG.md)')
            dropped.append(name); i += 1; continue
        out.append(tok); i += 1
    if dropped:
        print(f'{prog}: ignored removed option(s) given with their fixed value: {" ".join(dropped)}', file=sys.stderr, flush=True)
    return out


def _same(a, b):
    try:
        return float(a) == float(b)
    except ValueError:
        return str(a).strip().lower() == str(b).strip().lower()
