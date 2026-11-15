"""
Several GPUs = the same command once per GPU. Each copy claims its units (simcode.fileio.work.Units with claiming on), so the
copies share the work without a list being split beforehand, and a copy that dies is replaced by restarting the launcher.

  python -m simcode.fileio.launch --gpus 1,2 [--log <prefix>] -- <command ...>

Starts <command> once per GPU with CUDA_VISIBLE_DEVICES set, waits for all copies and returns the largest exit code.
--log <prefix>: stdout / stderr of each copy go to <prefix>_gpu<g>.log (appended); default: inherited.
The command must itself turn claiming on (e.g. `--claim 1` of simcode.run and of the 4Pi drivers).
"""
import argparse, os, subprocess, sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv: raise SystemExit(__doc__)
    i = argv.index('--'); cmd = argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='comma-separated GPU indices, one copy of the command per GPU')
    ap.add_argument('--log', default='', help='log file prefix')
    a = ap.parse_args(argv[:i])
    procs = []
    for g in a.gpus.split(','):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g.strip())
        out = open(f'{a.log}_gpu{g.strip()}.log', 'a') if a.log else None
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None)))
        print(f'GPU {g}: pid {procs[-1][1].pid}: {" ".join(cmd)}', flush=True)
    rc = 0
    for g, p in procs:
        r = p.wait(); rc = max(rc, abs(r))
        print(f'GPU {g}: exit code {r}', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
