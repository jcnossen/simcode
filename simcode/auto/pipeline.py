"""SIMCODE v2 automatic per-dataset pipeline (docs/AUTO_PIPELINE.md): one config file -> camera calibration, optional bead
fit, spot finding, isolated tracks, in-situ PSF, model, processing, report.

  python -m simcode.auto <config.yaml> [--stages all|camera,spots,...] [--redo psf] [--quick] [--gpus 1,2] [--force]

  python -m simcode.auto <setup.py> [--only name1,name2] [--show]      a Python file that defines `config` or a list `configs`

  from simcode.auto import LocalizationPipeline
  pipe = LocalizationPipeline(cfg)      # cfg: ConfigDict / dict / yaml / json / Python file (simcode/auto/configs/)
  pipe.run()                            # every stage, finished ones skipped
  pipe.estimate_psf()                   # one stage: determine_frame_order, calibrate_camera, find_spots, estimate_camera_maps,
                                        # extract_tracks, fit_beads, estimate_psf, train_model, process, report
  ds = pipe.dataset_pass2()             # simcode.dataset.Dataset of the processed movie (dataset_pass1(): pass 1); ds.select('main')

Stages (run order): order, camera, spots, cammaps, tracks, beads, psf, train, process, report. Each writes
<out>/<stage>/summary.json last (atomic) = its done marker; a finished stage is skipped unless redone (--redo <stage> redoes
it and every later stage). Each summary has 'checks' = [{name, value, ok, warn_only}]; a failed check that is not warn_only
stops the run unless --force."""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
# as runs/psfmatch/env.sh: triton needs the env's bin/ (ptxas) on PATH; no native-JIT kernels (set before torch is imported)
os.environ['PATH'] = os.path.dirname(sys.executable) + ':' + os.environ.get('PATH', '')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
from simcode.auto import config as C
from simcode.fileio.work import atomic_json

# stage name (directory under <out>, command line) -> the public method, in run order
METHODS = dict(order='determine_frame_order', camera='calibrate_camera', spots='find_spots', cammaps='estimate_camera_maps',
               tracks='extract_tracks', beads='fit_beads', psf='estimate_psf', train='train_model', process='process', report='report')
STAGES = list(METHODS)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StageFailed(RuntimeError):
    pass


class LocalizationPipeline:
    def __init__(self, cfg, gpus=None, log=None, quick=None):
        """cfg: ConfigDict (simcode/util/cfgdict.py), dict, yaml / json path, or a Python file that defines one `config`;
        validated against the schema of simcode/auto/config.py in every form. self.c = the resolved plain dict."""
        self.c = C.load(cfg)
        if gpus is not None: self.c['gpus'] = list(gpus)
        if quick is not None: self.c['train']['quick'] = bool(quick)
        self.out = self.c['out']
        os.makedirs(self.out, exist_ok=True)
        link = os.path.join(REPO, 'out', 'auto', self.c['name'])
        if os.path.realpath(self.out) != os.path.realpath(link) and not os.path.lexists(link):
            os.makedirs(os.path.dirname(link), exist_ok=True); os.symlink(os.path.abspath(self.out), link)
        self._log = log or self._default_log
        C.dump(self.c, os.path.join(self.out, 'config_resolved.json'))

    @classmethod
    def from_file(cls, cfg, only=None, **kw):
        """every acquisition of a config (a Python file with a list `configs`, or any single form) -> [LocalizationPipeline,
        ...]; only = names to keep. All configs are validated before the first object (and output directory) is made."""
        L = C.load_all(cfg)
        if only:
            miss = [n for n in only if n not in [c['name'] for c in L]]
            if miss: raise KeyError(f'no acquisition named {miss} in {cfg} (has: {[c["name"] for c in L]})')
            L = [c for c in L if c['name'] in only]
        return [cls(c, **kw) for c in L]

    @property
    def cfg(self):
        """the resolved config as a read-only ConfigDict (pipe.cfg.optics.NA)"""
        from simcode.util.cfgdict import ConfigDict
        return ConfigDict({k: v for k, v in self.c.items() if not k.startswith('_')}).freeze()

    # ---- helpers
    def _default_log(self, *a):
        msg = time.strftime('%H:%M:%S ') + ' '.join(str(v) for v in a)
        print(msg, flush=True)
        with open(os.path.join(self.out, 'auto.log'), 'a') as f: f.write(msg + '\n')

    def log(self, *a): self._log(*a)

    def dir(self, stage):
        d = os.path.join(self.out, stage); os.makedirs(d, exist_ok=True); return d

    def summary(self, stage):
        p = os.path.join(self.out, stage, 'summary.json')
        return json.load(open(p)) if os.path.exists(p) else None

    def done(self, stage): return self.summary(stage) is not None

    @property
    def movie(self):
        s = self.summary('order')
        return s['movie'] if s else self.c['movie']

    def gpu(self, need_gb=8):
        """first GPU of the config list with >= need_gb free"""
        try:
            q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
            free = {int(l.split(',')[0]): float(l.split(',')[1]) / 1024 for l in q.strip().splitlines()}
        except Exception:
            free = {}
        for g in self.c['gpus']:
            if free.get(g, 1e9) >= need_gb: return g
        raise StageFailed(f'no GPU of {self.c["gpus"]} has {need_gb} GB free ({free})')

    def spots_file(self):
        """spots used for the tracks (linking, isolation) and the camera-map mask: 'all' detections (default; the earlier
        recipe used the whole smlm table) or only the 'good' fits"""
        return os.path.join(self.out, 'spots', 'spots.hdf5' if self.c['spots']['tracks_from'] == 'all' else 'spots_good.hdf5')

    def torch_dev(self, need_gb=8):
        """device for the in-process GPU stages (spots, psf): CUDA_VISIBLE_DEVICES is set once, before CUDA is initialised"""
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu(need_gb))
        return 'cuda'

    def env(self, gpu=None, wrap=None):
        e = dict(os.environ, PYTHONPATH=REPO + (':' + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
        e['PATH'] = os.path.dirname(sys.executable) + ':' + e.get('PATH', '')
        if gpu is not None: e['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if wrap is not None: e['AUTO_WRAP'] = wrap
        return e

    def sh(self, args, log, gpu=None, wrap=None, check=True, env=None):
        """run [python, ...args] with the repo on PYTHONPATH, output to log; returns the log text"""
        cmd = [sys.executable, '-u'] + [str(a) for a in args]
        self.log('$', ' '.join(cmd), '>', log)
        with open(log, 'w') as f:
            f.write('# ' + ' '.join(cmd) + '\n'); f.flush()
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(self.env(gpu, wrap), **(env or {})), cwd=REPO)
        txt = open(log).read()
        if check and r.returncode != 0:
            raise StageFailed(f'command failed ({r.returncode}), see {log}:\n' + txt[-2000:])
        return txt

    def wrapfile(self, extra=None, name='wrap'):
        """json of the in-memory overrides (runwrap.py) for this dataset"""
        c = self.c; W = {}
        if tuple(c['pattern']['pitch_px']) != (1.6, 2.8): W['pitch_px'] = c['pattern']['pitch_px']
        if extra: W.update(extra)
        p = os.path.join(self.out, name + '.json'); atomic_json(p, W, indent=1)
        return p

    # ---- driver
    @staticmethod
    def _stage_name(x):
        """a stage given as one of the stage methods (pipe.estimate_psf), its name, or the stage name -> stage name"""
        n = getattr(x, '__name__', x)
        for k, m in METHODS.items():
            if n in (k, m): return k
        raise ValueError(f'unknown stage {x!r} (methods: {list(METHODS.values())}; command-line names: {STAGES})')

    def _invalidate(self, first, todo=STAGES):
        """redo: the done markers of stage `first` and of every later stage in todo are moved to summary.json.old"""
        for s in STAGES[STAGES.index(first):]:
            p = os.path.join(self.out, s, 'summary.json')
            if os.path.exists(p) and s in todo: os.replace(p, p + '.old')

    def run(self, only=None, redo=(), force=False):
        """Every stage in order, finished ones skipped. only / redo: stage methods (pipe.run(redo=[pipe.estimate_psf])) or
        their names; redo = run these again and every later stage; force = continue after failed checks.
        Returns {stage name: summary}."""
        if isinstance(only, str) or callable(only): only = [only]
        if isinstance(redo, str) or callable(redo): redo = [redo]
        sel = None if only is None or list(only) in ([], ['all']) else {self._stage_name(x) for x in only}
        todo = [s for s in STAGES if sel is None or s in sel]
        if redo:
            self._invalidate(min((self._stage_name(r) for r in redo), key=STAGES.index), todo)
        for s in todo:
            self._stage(s, force=force)
        return {s: self.summary(s) for s in todo}

    # ---- the stages: each returns its summary (also <out>/<stage>/summary.json = the done marker; a finished stage is
    # skipped). force = accept failed checks; redo = run it again (the later stages are then redone by the next run()).
    def determine_frame_order(self, force=False, redo=False):
        """frame order of the acquisition; writes / verifies the blocked-order movie when the order is not blocked"""
        return self._stage('order', force, redo)

    def calibrate_camera(self, force=False, redo=False):
        """gain, offset, read noise from the movie (stage a1)"""
        return self._stage('camera', force, redo)

    def find_spots(self, force=False, redo=False):
        """pattern-free Gaussian spot finding on moving 6-frame sums (stage c1)"""
        return self._stage('spots', force, redo)

    def estimate_camera_maps(self, force=False, redo=False):
        """per-pixel offset / read-noise maps on the frames without spots (stage a2)"""
        return self._stage('cammaps', force, redo)

    def extract_tracks(self, force=False, redo=False):
        """isolated emitter tracks for the PSF fit (stage c2)"""
        return self._stage('tracks', force, redo)

    def fit_beads(self, force=False, redo=False):
        """bead z-stack fit (stage b; skipped when the config has no beads)"""
        return self._stage('beads', force, redo)

    def estimate_psf(self, force=False, redo=False):
        """in-situ PSF + pattern calibration on the tracks (stage d)"""
        return self._stage('psf', force, redo)

    def train_model(self, force=False, redo=False):
        """PSF-matched networks and calibrations (stage e; skipped when train.model names an existing model)"""
        return self._stage('train', force, redo)

    def process(self, force=False, redo=False):
        """localize the whole movie with simcode.run (stage f)"""
        return self._stage('process', force, redo)

    def report(self, force=False, redo=False):
        """report page (stage g)"""
        return self._stage('report', force, redo)

    def run_dir(self):
        """directory of the processing run (simcode.run output: locs_p1 / locs_p2 tables, run_config.json)"""
        S = self.summary('process')
        if S is None: raise FileNotFoundError(f'{self.c["name"]}: not processed yet ({self.out}/process/summary.json is missing); call process() or run()')
        return S['run']

    def dataset_pass1(self, **kw):
        """the pass-1 localizations of the processed movie as a simcode.dataset.Dataset (selection: ds.select('main'))"""
        from simcode.dataset import Dataset
        return Dataset.from_run(self.run_dir(), pass_=1, **kw)

    def dataset_pass2(self, **kw):
        """the pass-2 (pattern-aware, final) localizations of the processed movie as a simcode.dataset.Dataset"""
        from simcode.dataset import Dataset
        return Dataset.from_run(self.run_dir(), pass_=2, **kw)

    def _stage(self, name, force=False, redo=False):
        if redo: self._invalidate(name)
        if self.done(name):
            self.log(f'[{name}] done (summary.json exists), skipped'); return self.summary(name)
        from simcode.util.proctitle import set_title
        set_title('auto-' + name[:7])
        self.log(f'[{name}] start')
        t0 = time.time()
        S = getattr(self, '_st_' + name)()
        S['stage'] = name; S['time_s'] = time.time() - t0; S['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        fails = [k for k in S.get('checks', []) if not k.get('ok') and not k.get('warn_only')]
        warns = [k for k in S.get('checks', []) if not k.get('ok') and k.get('warn_only')]
        for k in warns: self.log(f'[{name}] WARNING check: {k["name"]} = {k.get("value")}')
        if fails and not force:
            atomic_json(os.path.join(self.dir(name), 'summary_failed.json'), S, indent=1, default=str)
            raise StageFailed(f'[{name}] failed checks: ' + '; '.join(f'{k["name"]} = {k.get("value")}' for k in fails))
        atomic_json(os.path.join(self.dir(name), 'summary.json'), S, indent=1, default=str)
        self.log(f'[{name}] done in {S["time_s"]:.0f} s')
        return S

    # ---- stage: frame order
    def _st_order(self):
        c = self.c
        if not C.needs_reorder(c):
            return dict(movie=c['movie'], reordered=False, checks=[])
        from simcode.auto import order
        cache = c['pattern']['cache'] or os.path.join(self.out, 'movie_blocked.tif')
        inv = list(np.argsort(C.slot_of_frame(c)))           # cache slot s <- acquisition frame inv[s]
        if not os.path.exists(cache):
            order.write_blocked(c['movie'], cache, inv, log=self.log)
        chk = order.verify(c['movie'], cache, inv, log=self.log)
        return dict(movie=cache, reordered=True, perm_slot_from_frame=inv, **chk['info'], checks=chk['checks'])

    # ---- stage a1: camera
    def _st_camera(self):
        from simcode.auto import camera
        r = camera.estimate(self.movie, self.c['camera'], self.c['camera']['gain_frames'], log=self.log)
        return r

    # ---- stage c1: spots
    def _st_spots(self):
        import torch
        from simcode.auto import spots
        c, sp = self.c, self.c['spots']; cam = self.summary('camera')
        d = self.dir('spots')
        self.torch_dev(4)
        sig0 = 1.15 * C.sigma_px(c)
        import hashlib
        key = hashlib.sha1(json.dumps([self.movie, cam['gain'], cam['offset'], sp['maxframe'], sp['sum'], sp['thr'], sp['roi'], sig0]).encode()).hexdigest()[:10]
        cd = os.path.join(d, 'chunks_' + key); os.makedirs(cd, exist_ok=True)     # chunk files keyed by the settings (no stale reuse)
        L = spots.run(self.movie, cd, cam['gain'], cam['offset'], sp['maxframe'], sp['sum'], sp['thr'], sp['roi'], sig0, log=lambda *a: None)
        m, s_med = spots.good(L, sp['chi2_max'], sp['snr_min'], sp['sigma_range'])
        spots.write_hdf5(L, os.path.join(d, 'spots.hdf5')); spots.write_hdf5(L[m], os.path.join(d, 'spots_good.hdf5'))
        nf = int(L['frame'].max()) + 1 if len(L) else 0
        G = L[m]
        o = c['optics']; area = None
        from simcode.fileio.movie import open_movie
        H, W = open_movie(self.movie).shape[-2:]
        area = H * W * o['pixel_um'] ** 2
        stats = dict(n=int(len(L)), n_good=int(m.sum()), n_windows=nf, per_window=len(L) / max(nf, 1), good_per_window=float(m.sum()) / max(nf, 1),
                     photons_window_p5_50_95=np.percentile(G['photons'], [5, 50, 95]).tolist() if len(G) else None,
                     bg_photons_px_frame=float(np.median(G['bg']) / sp['sum']) if len(G) else None,
                     sigma_px_median=s_med, sigma_px_diffraction=C.sigma_px(c),
                     density_per_um2_window=float(m.sum()) / max(nf, 1) / area)
        chk = [dict(name='good spots per window > 1', value=stats['good_per_window'], ok=stats['good_per_window'] > 1),
               dict(name='median sigma / diffraction estimate in [0.9, 1.6]', value=s_med / C.sigma_px(c), ok=0.9 <= s_med / C.sigma_px(c) <= 1.6)]
        if c['camera']['map_frames'] > sp['maxframe']:
            chk.append(dict(name='camera.map_frames <= spots.maxframe (masks)', value=c['camera']['map_frames'], ok=False, warn_only=True))
        self.log('spots:', json.dumps(stats))
        return dict(stats=stats, sigma0=sig0, files=['spots.hdf5', 'spots_good.hdf5'], checks=chk)

    # ---- stage a2: camera maps
    def _st_cammaps(self):
        c = self.c; cam = self.summary('camera'); d = self.dir('cammaps')
        args = ['-m', 'simcode.insitu.camera_maps', '--movie', self.movie, '--nframes', c['camera']['map_frames'], '--gain', cam['gain'],
                '--offset', cam['offset'], '--locs', self.spots_file(), '--out', os.path.join(d, 'camera_maps.npz')]
        if c['camera']['readnoise_from']: args += ['--readnoise_from', c['camera']['readnoise_from']]
        txt = self.sh(args, os.path.join(d, 'camera_maps.log'))
        corr = float(re.search(r'corr\(offset_dev, loc density\) = ([-\d.]+)', txt).group(1))
        M = np.load(os.path.join(d, 'camera_maps.npz'))
        st = dict(offset_dev_sd=float(np.std(M['offset_adu'] - cam['offset'])), rn_median=float(np.median(M['readnoise_e'])),
                  rn_p99=float(np.percentile(M['readnoise_e'], 99)), corr_offset_density=corr, readnoise_from=c['camera']['readnoise_from'])
        return dict(stats=st, file='camera_maps.npz', checks=[dict(name='corr(offset_dev, loc density) < 0.05', value=corr, ok=abs(corr) < 0.05, warn_only=True)])

    # ---- stage c2: tracks
    def _st_tracks(self):
        sp = self.c['spots']; d = self.dir('tracks')
        txt = self.sh(['-m', 'simcode.insitu.extract_tracks', '--movie', self.movie, '--locs', self.spots_file(),
                       '--maxframe', sp['maxframe'], '--roi', sp['track_roi'], '--isolation', sp['isolation_px'], '--link', sp['link_px'],
                       '--out', os.path.join(d, 'tracks.npz')], os.path.join(d, 'tracks.log'))
        g = lambda pat: int(re.search(pat, txt).group(1))
        st = dict(tracks=g(r'(\d+) tracks spanning'), isolated=g(r'(\d+) isolated tracks'), samples=g(r'(\d+) samples of up to'))
        need = 2 * self.c['psf']['nsamples']
        return dict(stats=st, file='tracks.npz', checks=[dict(name=f'samples >= 2 x psf.nsamples ({need})', value=st['samples'], ok=st['samples'] >= need)])

    # ---- stage b: beads
    def _st_beads(self):
        b = self.c['beads']
        if not b:
            return dict(skipped=True, note='no bead z-stacks in the config: stage d starts from zero aberrations', checks=[])
        o, cam = self.c['optics'], self.summary('camera'); d = self.dir('beads')
        g = self.gpu(6)
        pre = os.path.join(d, 'beads')
        args = ['-m', 'simcode.psf.fit_beads', '--stacks', *b['stacks'], '--dz', b['dz_um'], '--sign', b['sign'],
                '--gain', b['gain_e_per_adu'] or cam['gain'], '--offset', b['offset_adu'] if b['offset_adu'] is not None else cam['offset'],
                '--na', o['NA'], '--wl', o['wavelength_um'], '--nmed', o['n_med'], '--pixelsize', o['pixel_um'], '--nball', b['nball'],
                '--iters', b['iters'], '--holdout', b['holdout'], '--roi', b['roi'], '--out', pre, '--tag', '']
        txt = self.sh(args, os.path.join(d, 'beads.log'), gpu=g)
        f = f'{pre}_sign{int(b["sign"])}'
        J = json.load(open(f + '.json'))
        Z = np.array(list(J['zernike'].values()))
        hd = re.search(r'holdout: mean Poisson deviance/pixel ([\d.]+)', txt)
        return dict(pt=f + '.pt', json=f + '.json', zernike_rms=float(np.sqrt((Z ** 2).sum())), NA=J['NA'],
                    holdout_deviance=float(hd.group(1)) if hd else None,
                    checks=[dict(name='|Z| < 0.15 waves', value=float(np.sqrt((Z ** 2).sum())), ok=float(np.sqrt((Z ** 2).sum())) < 0.15)])

    # ---- stage d: in-situ PSF
    def _st_psf(self):
        from simcode.auto import stage_psf
        return stage_psf.run(self)

    # ---- stage e: model (trained networks + calibrations)
    def _st_train(self):
        from simcode.auto import train
        return train.run(self)

    # ---- stage f: processing
    def _st_process(self):
        from simcode.auto import process
        return process.run(self)

    # ---- stage g: report
    def _st_report(self):
        from simcode.auto import report
        return report.run(self)


def main(argv=None):
    ap = argparse.ArgumentParser(description='SIMCODE v2 automatic per-dataset pipeline (docs/AUTO_PIPELINE.md)')
    ap.add_argument('config', help='yaml / json file, or a Python file that defines `config` or a list `configs`')
    ap.add_argument('--stages', default='all', help='comma list of ' + ','.join(STAGES) + ' or all')
    ap.add_argument('--redo', default='', help='comma list: redo these stages and every later one')
    ap.add_argument('--quick', action='store_true', help='stage e: few steps per trainer (plumbing test)')
    ap.add_argument('--gpus', default='', help='override the config GPU list, e.g. 1,2')
    ap.add_argument('--force', action='store_true', help='continue after failed checks')
    ap.add_argument('--only', default='', help='comma list of acquisition names (a Python file with several `configs`)')
    ap.add_argument('--show', action='store_true', help='print the resolved config(s) and exit (nothing is written)')
    a = ap.parse_args(argv)
    if a.show:
        from simcode.util.cfgdict import ConfigDict
        only = [n for n in a.only.split(',') if n]
        for c in C.load_all(a.config):
            if not only or c['name'] in only: print(ConfigDict(c))
        return
    L = LocalizationPipeline.from_file(a.config, only=[n for n in a.only.split(',') if n],
                              gpus=[int(v) for v in a.gpus.split(',')] if a.gpus else None, quick=True if a.quick else None)
    failed = []
    for ds in L:                    # one after the other; a failed acquisition does not stop the next ones
        try:
            ds.run(only=a.stages.split(','), redo=[r for r in a.redo.split(',') if r], force=a.force)
        except StageFailed as e:
            if len(L) == 1: raise
            ds.log('FAILED:', e); failed.append(ds.c['name'])
    if failed: raise SystemExit(f'failed acquisitions: {failed}')


if __name__ == '__main__':
    main()
