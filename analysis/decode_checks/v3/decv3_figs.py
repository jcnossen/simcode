"""Tagged copies (_decv3) of Fig 2 (fig_sim_common.py: fig_sim_Dexp_v2psf_wsm3d) and of the density-sweep SI figures
(fig_sim_main.py --v2key v2c: fig_sim_{D,Dexp}_v2psf_wsm3d_common) with DECODE = dec_m1809v3 (<run>/decode/locs_v3.npz), made with
the unchanged figure scripts and the common rule. Nothing of the paper's own outputs is written or changed:
  shadow movie dirs out/decode_checks/v3/shadow/<movie>/ (symlinks to the movie files and to every method's run output, with
  full/decode/locs.npz -> locs_v3.npz), metrics.py --only decode and gtfrc_run.py recomputed there, common_rule.py into
  shadow/common_rule, the figure scripts run with common.OUT = the shadow dir; the figures are then copied with the suffix _decv3
  into out/paper_figs_v2 (Fig 2) and out/papersim/figures (SI), and the caption numbers old vs new -> decv3_caption_diff.md.
usage: python analysis/decode_checks/v3/decv3_figs.py [--steps shadow,metrics,rule,figs,diff]"""
import os, sys, json, glob, shutil, subprocess, argparse, runpy, difflib
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAIN = os.path.expanduser('~/simcode2')
PSD = os.path.join(MAIN, 'out', 'papersim')
V3 = os.path.join(MAIN, 'out', 'decode_checks', 'v3')
SH = os.path.join(V3, 'shadow')
SETS = [f'{p}{d}_v2psf_wsm3d' for p in ('D', 'Dexp') for d in ('0.5', '1', '2', '4')]
MODEL = 'm1809v2'
PY = sys.executable
HERE_PS = os.path.join(REPO, 'analysis', 'papersim')


def shadow():
    for s in SETS:
        src, dst = os.path.join(PSD, s), os.path.join(SH, s)
        os.makedirs(os.path.join(dst, 'full', 'decode'), exist_ok=True)
        for f in os.listdir(src):
            if f == 'full': continue
            if not os.path.lexists(os.path.join(dst, f)): os.symlink(os.path.join(src, f), os.path.join(dst, f))
        for f in os.listdir(os.path.join(src, 'full')):
            p = os.path.join(src, 'full', f)
            # directories only: every file the scripts write into the run dir (json, png) must be a new file in the shadow, never
            # a symlink to the paper's file
            if f == 'decode' or f == 'logs' or not os.path.isdir(p): continue
            if not os.path.lexists(os.path.join(dst, 'full', f)): os.symlink(p, os.path.join(dst, 'full', f))
        ln = os.path.join(dst, 'full', 'decode', 'locs.npz')
        if not os.path.lexists(ln): os.symlink(os.path.join(src, 'full', 'decode', 'locs_v3.npz'), ln)
        for j in (f'metrics_{MODEL}.json', f'gtfrc_{MODEL}.json'):
            if not os.path.exists(os.path.join(dst, 'full', j)): shutil.copyfile(os.path.join(src, 'full', j), os.path.join(dst, 'full', j))
        print('shadow', dst, flush=True)


def metrics():
    procs = []
    for s in SETS:
        md = os.path.join(SH, s); run = os.path.join(md, 'full'); lg = open(os.path.join(V3, 'logs', f'decv3_metrics_{s}.log'), 'w')
        cmd = f'{PY} {HERE_PS}/metrics.py {run} --movie {md} --model {MODEL} --only decode && {PY} {HERE_PS}/gtfrc_run.py {run} --movie {md} --model {MODEL}'
        procs.append(subprocess.Popen(cmd, shell=True, stdout=lg, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}))
        if len(procs) >= 4:
            for p in procs: p.wait()
            procs = []
    for p in procs: p.wait()


def patch():
    sys.path[:0] = [HERE_PS, os.path.join(REPO, 'analysis', 'paper'), REPO]
    import common as C
    C.OUT = SH
    import common_rule as CR
    CR.OUTD = os.path.join(SH, 'common_rule')
    return C, CR


def rule():
    C, CR = patch()
    for s in SETS: CR.run_set(s, MODEL, CR.OUTD)


def run_script(path, argv):
    sys.argv = [path] + argv
    runpy.run_path(path, run_name='__main__')


def figs():
    C, CR = patch()
    fo = os.path.join(SH, 'figs'); os.makedirs(fo, exist_ok=True)
    run_script(os.path.join(HERE_PS, 'fig_sim_common.py'), ['--set', 'Dexp1_v2psf_wsm3d', '--out', fo])
    for p in ('D', 'Dexp'):
        run_script(os.path.join(HERE_PS, 'fig_sim_main.py'), ['--prefix', p, '--suffix', '_v2psf_wsm3d', '--model', MODEL, '--v2key', 'v2c'])
    # copies with the _decv3 tag next to the paper's files
    pairs = [(os.path.join(fo, 'fig_sim_Dexp_v2psf_wsm3d'), os.path.join(MAIN, 'out', 'paper_figs_v2', 'fig_sim_Dexp_v2psf_wsm3d'))]
    pairs += [(os.path.join(SH, 'figures', f'fig_sim_{p}_v2psf_wsm3d_common'), os.path.join(PSD, 'figures', f'fig_sim_{p}_v2psf_wsm3d_common')) for p in ('D', 'Dexp')]
    for a, b in pairs:
        for f in glob.glob(a + '*'):
            if '_internal' in f[len(a):]: continue
            dst = b + '_decv3' + f[len(a):]
            shutil.copyfile(f, dst); print('->', dst, flush=True)


def diff():
    L = ['# Caption / table numbers: current paper files vs the _decv3 copies (DECODE = dec_m1809v3)', '']
    for base in (os.path.join(MAIN, 'out', 'paper_figs_v2', 'fig_sim_Dexp_v2psf_wsm3d'),
                 os.path.join(PSD, 'figures', 'fig_sim_D_v2psf_wsm3d_common'), os.path.join(PSD, 'figures', 'fig_sim_Dexp_v2psf_wsm3d_common')):
        for ext in ('_caption.txt', '_table.md'):
            a, b = base + ext, base + '_decv3' + ext
            if not (os.path.exists(a) and os.path.exists(b)): continue
            d = [l for l in difflib.unified_diff(open(a).read().splitlines(), open(b).read().splitlines(), os.path.basename(a), os.path.basename(b), n=0, lineterm='')]
            L += [f'## {os.path.basename(b)}', '', '```diff'] + d + ['```', '']
    open(os.path.join(V3, 'decv3_caption_diff.md'), 'w').write('\n'.join(L) + '\n'); print(os.path.join(V3, 'decv3_caption_diff.md'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', default='shadow,metrics,rule,figs,diff'); a = ap.parse_args()
    for s in a.steps.split(','): globals()[s]()
