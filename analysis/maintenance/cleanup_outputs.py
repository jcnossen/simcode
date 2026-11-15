"""Clean-up of the output directory and the models directory after the 2026-11-15 reorganisation (docs/CLEANUP_CANDIDATES.md,
the user's decisions of 2026-11-15). Run it after the merge, when no worker uses these files any more.

  python analysis/maintenance/cleanup_outputs.py                 dry run: every action with its size (nothing is changed)
  python analysis/maintenance/cleanup_outputs.py --apply         do it
  options: --out <dir> (default: config root 'out'), --models <dir> (default: config models dir), --targets 1 (for a symlinked
           entry also delete the data it points to; default 0 = only the link), --only delete|archive|models|keys|links

Actions:
  delete   out/ entries: development / temporary outputs of earlier networks, one-off logs and scripts, zimflux
  archive  out/ entries moved to <out>/archive/ (results kept, not in the working set): nanoruler, tput, the high-density /
           evaluation-protocol comparison and the finished studies with a STATUS record
  links    the old names out/sc4 and out/p2 (symlinks; the data are reached through out/emitter_heldout, out/gate_dumps,
           out/pass2_gate), and the reorganisation's own check outputs out/rename
  models   weight files in <models>: byte-identical copies of files in the model folders / archives (checked and listed),
           superseded sets (psf3dmix, ablation heads, hd_sc7bg, p2_det / p2_head_long, gnr80, p2det_g2ss)
  keys     the model.json files below <models> rewritten with the present key names (analysis/maintenance/rename_model_keys.py)
Entries not listed are kept. Nothing under the raw data root is touched.
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

DELETE = """ablate_frr_hd_b.json ablate.log ablate_polish_hd_b.json ablate_prior_hd_b.json ablate_r11.json ablate_r41.json
bench bench_hd_a_quick.log bright calib_sc_hd_sc1.json calib_sc_hd_sc1.json.p1 calib_sc_hd_sc2b15.json calib_sc_hd_sc2b15.json.p1
calib_sc_hd_sc2.json calib_sc_hd_sc2.json.p1 calib_sc_hd_sc2Lb15.json calib_sc_hd_sc2Lb15.json.p1 calib_sc_hd_sc2Ln1b15.json
calib_sc_hd_sc2Ln1b15.json.p1 calib_sc_hd_sc3b15.json calib_sc_hd_sc3b15.json.p1 calib_sc_hd_sc5db15.json calib_sc_hd_sc5db15.json.p1
calib_sc_hd_sc5nb15.json calib_sc_hd_sc5nb15.json.p1 calib_sc_hd_sc5nb15_pm.json calib_sc_hd_sc5nbb15.json calib_sc_hd_sc5nbb15.json.p1
calib_sc_hd_sc5nbb15_pm_autoc1.json calib_sc_hd_sc5nbb15_pm_autoc.json calib_sc_hd_sc5nbb15_pm_bright.json
calib_sc_hd_sc5nbb15_pm_bright_real.json calib_sc_hd_sc5nbb15_pm.json calib_sc_hd_sc6nb15.json calib_sc_hd_sc6nb15.json.p1
calib_sc_hd_sc6nb15_pm.json calib_sc_hd_sc7bgb15.json calib_sc_hd_sc7bgb15.json.p1 calib_sc_hd_sc7bgb15_pm.json
critic34 critic35 decomp decomp_sc2L decomp_sc2Ln1 decomp_sc3 decomp_sc5d decomp_sc5n decomp_sc5n_pm det_up2 drift drift25 ext
field full_0.5nM_p2L_real full_0.5nM_retr_real full_0.5nM_sc5nb_autoc full_0.5nM_sc5nb_autoc_v1s full_0.5nM_sc5nb_bright_real
full_20nM_m2_fast full_20nM_m2_fast_autoc full_2.5nM_m2_p2L_real full_2.5nM_m2_retr_real full_2.5nM_m2_sc5nb_autoc
full_2.5nM_m2_sc5nb_bright_real gate_gpu1.log gate_gpu1.pid gate_gpu2.log gate_gpu2.pid gate_report_sc1.json gate_report_sc.json
gate_report_v1.json gate_report_v2.json logs_move_paper2.sh miss_anatomy.py move_paper2_b.log move_paper2_b.sh move_paper2.log n20
nuis p2prior p2real plots plots_check plots_check2 post_hd_b.log post_hd_b.pid psf_v2 real20 realgap realism refs resolvability
retrain_accept.txt retrain_r run_baselines.log run_baselines.pid run_baselines.sh run_chain_b.sh run_gate.sh run_gpu2_a.pid
run_gpu2_a.sh run_post.sh run_sc1.log run_sc1.pid run_sc2.log run_sc.sh sc3 sc6 sc7 scratch simcode_base simcode_base2 simreal2
smoke_0.5nM_sc5n smoke_0.5nM_sc5nb smoke_0.5nM_sc5nb_bright smoke_0.5nM_sc5nb_bright_real speed static25 summary_baselines.json
time_sc.py validate_0.5nM_final.json validate_0.5nM_final.txt validate_0.5nM.json validate_0.5nM_v2.json validate_0.5nM_v3.json
validate_0.5nM_v4.json validate_2.5nM_final.json validate_2.5nM_final.txt validate_2.5nM.json validate_2.5nM_v2.json
validate_2.5nM_v3.json validate_2.5nM_v4.json zimflux""".split()
ARCHIVE = """nanoruler tput hd_compare evalproto psfshape autoc calib_inputs hdunbias detband solvefix cleanpass optclean""".split()
LINKS = ['sc4', 'p2', 'rename']                      # sc4 / p2: old names (symlinks) of emitter_heldout / gate_dumps + pass2_gate
MODELS_DELETE = """det_m1809.pt det_m25.pt det_g13.pt hd_m1809.pt hd_m25.pt hd_g13.pt p2det_m1809.pt p2det_m25.pt p2det_g13.pt
p2headL_m1809.pt p2headL_m25.pt p2headL_g13.pt p2det_g2ss.pt p2det_gnr80.pt p2headL_gnr80.pt det_psf3dmix.pt hd_psf3dmix.pt
p2det_psf3dmix.pt p2headL_psf3dmix.pt p2det_psf3dmix_pair.pt p2headL_psf3dmix_pair.pt hd_abl1809_avg.pt hd_abl1809_ax.pt
hd_sc7bg.pt p2_det.pt p2_head_long.pt""".split()
# generic pass-2 study variants (out/p2generic/STATUS.md): moved to <models>/archive_p2generic/
MODELS_ARCHIVE = """p2det_g1809.pt p2det_g2c.pt p2det_g2s.pt p2det_g2w.pt p2det_g2cc.pt p2headL_g1809.pt p2headL_g2c.pt
p2headL_g2cc.pt p2headL_g2s.pt p2headL_g2ss.pt p2headL_g2w.pt""".split()


def size(p):
    try:
        r = subprocess.run(['du', '-sh', '--apparent-size', p], capture_output=True, text=True, timeout=600)
        return r.stdout.split('\t')[0] or '?'
    except Exception:
        return '?'


def copies(models, name, path):
    """weight files below <models> (not the file itself, not one to be deleted) byte-identical to it"""
    import hashlib
    h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
    me, n = h(path), os.path.getsize(path); out = []
    for dp, _, fs in os.walk(models):
        for f in fs:
            q = os.path.join(dp, f)
            if q == path or (dp == models and f in MODELS_DELETE) or not f.endswith('.pt') or os.path.getsize(q) != n: continue
            if h(q) == me: out.append(os.path.relpath(q, models))
    return out


def remove(p, targets):
    if os.path.islink(p):
        t = os.path.realpath(p); os.unlink(p)
        if targets and os.path.isdir(t): shutil.rmtree(t)
        elif targets and os.path.exists(t): os.remove(t)
    elif os.path.isdir(p): shutil.rmtree(p)
    elif os.path.exists(p): os.remove(p)


def main():
    from simcode import config
    ap = argparse.ArgumentParser(); ap.add_argument('--apply', action='store_true'); ap.add_argument('--targets', type=int, default=0)
    ap.add_argument('--out', default=config.root('out')); ap.add_argument('--models', default=config.models_dir())
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    do = lambda part: not a.only or part in a.only.split(',')
    sys.stdout.reconfigure(line_buffering=True)
    print(('APPLY' if a.apply else 'DRY RUN (nothing is changed; --apply does it)') + f'; out {a.out}; models {a.models}; '
          f'symlink targets {"deleted too" if a.targets else "kept (only the link is removed)"}')
    if do('delete'):
        print('\n## delete (out/)')
        for n in DELETE:
            p = os.path.join(a.out, n)
            if not os.path.lexists(p): continue
            t = f' -> {os.path.realpath(p)}' if os.path.islink(p) else ''
            print(f'  delete  {n:45s} {size(p):>8s}{t}')
            if a.apply: remove(p, a.targets)
    if do('archive'):
        print('\n## archive (out/ -> out/archive/)')
        for n in ARCHIVE:
            p = os.path.join(a.out, n)
            if not os.path.lexists(p): continue
            print(f'  archive {n:45s} {size(p):>8s}' + (f' -> {os.path.realpath(p)} (the link moves)' if os.path.islink(p) else ''))
            if a.apply:
                os.makedirs(os.path.join(a.out, 'archive'), exist_ok=True); shutil.move(p, os.path.join(a.out, 'archive', n))
    if do('links'):
        print('\n## old names / check outputs (out/)')
        for n in LINKS:
            p = os.path.join(a.out, n)
            if not os.path.lexists(p): continue
            if n in ('sc4', 'p2') and not os.path.islink(p): print(f'  skip    {n} (not a symlink)'); continue
            print(f'  remove  {n:45s} {"(symlink; the data stay)" if os.path.islink(p) else size(p)}')
            if a.apply: os.unlink(p) if os.path.islink(p) else shutil.rmtree(p)
    if do('models'):
        print('\n## models')
        for n in MODELS_DELETE:
            p = os.path.join(a.models, n)
            if os.path.exists(p):
                same = copies(a.models, n, p)
                print(f'  delete  {n:45s} {size(p):>8s}  ' + (f'identical copy: {same[0]}' if same else 'no identical copy (superseded set)'))
                if a.apply: os.remove(p)
        for n in MODELS_ARCHIVE:
            p = os.path.join(a.models, n)
            if os.path.exists(p):
                print(f'  archive {n:45s} {size(p):>8s} -> archive_p2generic/')
                if a.apply:
                    os.makedirs(os.path.join(a.models, 'archive_p2generic'), exist_ok=True); shutil.move(p, os.path.join(a.models, 'archive_p2generic', n))
    if do('keys'):
        print('\n## model.json key names')
        cmd = [sys.executable, os.path.join(HERE, 'rename_model_keys.py')] + ([] if a.apply else ['--check']) + [a.models]
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
