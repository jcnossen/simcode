"""4Pi set-ups. FourPiSetup = 4Pi-SIMFLUX: four interference channels of one sCMOS camera (two of them mirrored), Hamamatsu dcimg files (a movie
= several files; every file starts at pattern step 0), 6-frame windows (2 angles x 3 steps), the coherent two-arm 4Pi PSF,
the 4Pi detector (registered 4-channel sum) and emitter network (24 unwarped channel patches), and per file: the channel
registration (stage registration), the SIM pattern and the 4Pi phase field (simcode/processing/fourpi_pattern_phase.py)
and the explicit 4Pi solve (simcode/solve/fourpi.py). The acquisitions: simcode/pipeline/configs/fourpi.py.
FourPiPlainSetup (setup fourpi_plain) = plain 4Pi (4Pi-SMS, no SIM pattern): the same channels, front end, networks and
solve with windows of fourpi.window frames (1: single frames; the networks are trained for that T), no pattern stage,
no per-angle 4Pi offset; the 4Pi phase field per file as for 4Pi-SIMFLUX. Acquisition: simcode/pipeline/configs/fourpi_plain.py.

Config section 'fourpi' (defaults below), besides the common keys (camera.gain_e_per_adu, camera.readnoise_e,
camera.offset_map, optics.pixel_um, channels, registration):
  frame_dt_s    frame time (s);  xlim  illuminated columns [x0, x1) of a channel crop;  patch  candidate patch (px)
  window        frames per window (4Pi-SIMFLUX: 6);  pattern  SIM pattern (4Pi-SIMFLUX: true)
  delta_deg     4Pi phase offset of the second pattern angle (deg)
  z_period_um   4Pi z period (um): spacing of the solve's z starts (Lambda / 2 apart)
  sb            prior sd (e/px) of the solve's residual background per channel (training: the same value)
  file_field    which '_'-separated field of a movie file name is the file number (-1: Cell03_561_000_005.dcimg -> 5)
  det_batch     windows per detector batch
  zstack, bead_align   file globs of the bead z-stack / bead-align recordings of the acquisition (PSF / registration
                calibration samples; not used by the processing)
  psf           the 4Pi PSF model json (key 'psf': a psf4pi state);  pattern_ref  reference SIM pattern json (k start)
  detector, emitter    checkpoints of the 4Pi detector and emitter network (emitter: a path or a tag of
                <4Pi work dir>/models/head4pi_<tag>.pt)
  files         file numbers to process (null: all);  nframes  frames per file (0: all)
  and the processing settings of simcode/processing/fourpi.py (chunk, p_min, tile, tblk, dnll, chi2_max, coh, iters,
  batch, min_photons, nphi, regions, tag; claim / redo / order / min_free_gb of the file driver)

  acquisition(name)   the resolved config of a 4Pi acquisition of simcode/pipeline/configs/fourpi.py ('MT', 'NPC' or
                      the full name 'fourpi_MT')
  file_index(path)    the file number of a dcimg file (Cell03_561_000_005.dcimg -> 5)
"""
import json, math, os
import numpy as np
from simcode.setups.base import Setup

DEFAULTS = dict(frame_dt_s=5.575e-3, xlim=[55, 305], patch=15, window=6, pattern=True, delta_deg=None, z_period_um=0.257, sb=0.2,
                file_field=-1, det_batch=20, zstack=None, bead_align=None, psf=None,
                pattern_ref=None, detector=None, emitter='v10', files=None, nframes=0,
                chunk=600, p_min=0.5, tile=12.0, tblk=50, dnll=9.21, chi2_max=2.5, coh=1.0, iters=15, batch=1500,
                min_photons=300.0, regions='', nphi=60000, tag='', claim=0, redo=0, order='fwd', min_free_gb=10.0)


def file_index(fn, field=-1):
    return int(os.path.basename(fn).split('.')[0].split('_')[field])


def acquisition(name):
    """resolved config of a 4Pi acquisition of simcode/pipeline/configs/fourpi.py"""
    from simcode.pipeline import acquisition_config as C
    L = C.load_all(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipeline', 'configs', 'fourpi.py'))
    for c in L:
        if c['name'] in (name, f'fourpi_{name}'): return c
    raise KeyError(f'no 4Pi acquisition {name!r} (have {[c["name"] for c in L]})')


class FourPiSetup(Setup):
    name = 'fourpi'

    def defaults(self):
        return dict(fourpi=dict(DEFAULTS))

    def validate(self, c):
        f = c['fourpi']
        if not c.get('channels') or len(c['channels']) != 4:
            raise ValueError('setup fourpi: four channels (the interference channels) are required in channels')
        if f['delta_deg'] is None: raise ValueError('setup fourpi: fourpi.delta_deg is required')
        if c['pattern']['orientations'] * c['pattern']['steps'] != 6 or f['window'] != 6 or not f['pattern']:
            raise ValueError('setup fourpi: 6-frame windows (2 angles x 3 steps) with the SIM pattern (plain 4Pi: setup fourpi_plain)')

    def stages(self, c):
        return ['registration', 'process', 'report']

    # ---- files and readers
    def movie_files(self, c):
        import glob
        return sorted(glob.glob(c['movie']))

    def files(self, c):
        """the file numbers to process"""
        fs = sorted(file_index(f, c['fourpi']['file_field']) for f in self.movie_files(c))
        return fs if c['fourpi']['files'] is None else [f for f in fs if f in set(c['fourpi']['files'])]

    def movie_file(self, c, fi):
        fs = [f for f in self.movie_files(c) if file_index(f, c['fourpi']['file_field']) == fi]
        if not fs: raise FileNotFoundError(f'{c["name"]}: no movie file with index {fi} ({c["movie"]})')
        return fs[0]

    def layout(self, c):
        from simcode.channels.layout import ChannelLayout
        return ChannelLayout.from_config(c)

    def reader(self, c, path):
        from simcode.io.movie import DcimgMovie
        from simcode.channels.layout import ChannelReader
        return ChannelReader(DcimgMovie(path), self.layout(c))

    # ---- calibration inputs
    def camera(self, c):
        """offset (ADU: map [C, H, W] of the channel crops or a scalar), gain (e / ADU), read noise (e)"""
        cam = c['camera']
        off = np.load(cam['offset_map'])['off'] if cam.get('offset_map') else cam['offset_adu']
        if off is None: raise ValueError(f'{c["name"]}: camera.offset_map or camera.offset_adu is required')
        return dict(offset=off, gain=cam['gain_e_per_adu'], readnoise_e=cam['readnoise_e'])

    def regmap_file(self, c, fi):
        """the registration map of file fi: the per-file in-situ map of the registration stage, or registration.map"""
        r = c['registration']
        if r['mode'] == 'file': return r['map']
        return os.path.join(c['out'], 'registration', f'map_{fi:03d}.json')

    def regmap(self, c, fi):
        from simcode.channels.registration import PolyMap
        L = self.layout(c)
        return PolyMap.load(self.regmap_file(c, fi), size=(L.W, L.H))

    def front(self, c, fi=None, dev='cuda', map_file=None):
        """ChannelFront of file fi (its registration map), or with the map json map_file"""
        from simcode.channels.front import ChannelFront
        from simcode.channels.registration import PolyMap
        L = self.layout(c); cam = self.camera(c)
        pm = PolyMap.load(map_file, size=(L.W, L.H)) if map_file else self.regmap(c, fi)
        return ChannelFront(pm, cam['offset'], cam['gain'], L.C, L.H, L.W, P=c['fourpi']['patch'],
                            xlim=c['fourpi']['xlim'], weights=L.weights, dev=dev)

    def delta(self, c):
        """per-angle 4Pi phase offsets (rad)"""
        return (0.0, math.radians(c['fourpi']['delta_deg']))

    def solve_kw(self, c, net):
        """FourPiSolve arguments of the set-up (besides the PSF table and the windows of the emitter network net)"""
        return dict(delta=self.delta(c))

    # ---- stages
    def st_registration(self, pipe):
        """per file: the in-situ registration map (registration.mode insitu) -> <out>/registration/map_<file>.json, or
        the configured map for every file (mode file)"""
        c = pipe.c; r = c['registration']
        files = self.files(c)
        if r['mode'] == 'file':
            from simcode.channels.registration import PolyMap
            self.regmap(c, files[0])          # readable
            return dict(mode='file', map=r['map'], files=files, checks=[])
        from simcode.channels import registration as REG
        d = pipe.dir('registration'); per = {}
        dev = pipe.torch_dev(6)
        import torch
        if not torch.cuda.is_available(): dev = 'cpu'           # CUDA_VISIBLE_DEVICES='' : the registration runs on the CPU
        for fi in files:
            fn = self.regmap_file(c, fi)
            if os.path.exists(fn):
                per[fi] = json.load(open(fn)).get('report', {}); continue
            rd = self.reader(c, self.movie_file(c, fi))
            pm, rep = REG.register_insitu(rd, self.camera(c), r, xlim=c['fourpi']['xlim'], pixel_um=c['optics']['pixel_um'],
                                          dev=dev, log=pipe.log)
            pm.save(fn, source=f'in situ: {c["name"]} file {fi}', report=rep)
            per[fi] = rep
        return dict(mode='insitu', files=files, maps=[os.path.basename(self.regmap_file(c, fi)) for fi in files], per_file=per,
                    checks=REG.checks(per))

    def st_process(self, pipe):
        """every file through simcode.processing.fourpi (one worker per GPU of the config, files claimed)"""
        c = pipe.c; d = pipe.dir('process')
        cfgf = os.path.join(pipe.out, 'config_resolved.json')
        gpus = c['gpus']
        if len(gpus) > 1:
            args = ['-m', 'simcode.io.launch', '--gpus', ','.join(str(g) for g in gpus), '--', sys_python(), '-m', 'simcode.processing.fourpi',
                    cfgf, '--claim', '1']
            pipe.sh(args, os.path.join(d, 'process.log'))
        else:
            pipe.sh(['-m', 'simcode.processing.fourpi', cfgf], os.path.join(d, 'process.log'), gpu=pipe.gpu(10))
        from simcode.processing.fourpi import output_units
        U = output_units(c)
        files = self.files(c); miss = U.missing(files)
        return dict(files=files, out=d, records={str(k): v for k, v in U.records(files).items()},
                    checks=[dict(name='every file processed', value=len(files) - len(miss), ok=not miss)])

    def st_report(self, pipe):
        S = pipe.summary('process') or {}
        R = S.get('records', {})
        rows = {k: dict(n=v.get('n'), good_frac=v.get('good_frac'), kin_frac=v.get('kin_frac'), field_R=v.get('field_R'),
                        chi2_dof_median=v.get('chi2_dof_median'), sig_med_nm_2_4k=v.get('sig_med_nm_2_4k')) for k, v in R.items()}
        return dict(files=rows, checks=[])


# training recipe of the plain-4Pi networks (scene simulator MultiChannelAcquisitionSim with patterns=False; trainers
# analysis/fourpi/train_det4pi.py / train_head4pi_w.py --acq <config>): emitters log-normal F_med (sd log F_sd, clipped
# to F_rng) e per full frame, on-through with p_through else starting / ending inside the frame; density log-uniform
# dens / um^2 / frame (uniform, filaments, NPC-like rings, close pairs); z uniform in zr (um) with the PSF tables over
# zlut and the networks' prior range +-zmax; background bg (e/px per channel) x log-uniform bg_rng, structured;
# read noise camera.readnoise_e, offset error offerr (e); registration error per scene (shift / linear, px); PSF bank
# of n_psf variants of the fitted state (simulate/multichannel.perturbed_states)
TRAIN_DEFAULTS = dict(bg=None, dens=[0.005, 0.5], zr=[-0.9, 0.6], zlut=[-1.0, 1.0], zmax=1.0, F_med=2500.0, F_sd=0.75,
                      F_rng=[150.0, 30000.0], p_through=0.5, offerr=0.2, bg_rng=[0.4, 2.0], reg_shift_sd=0.08, reg_lin_sd=0.05,
                      n_psf=6, det_steps=30000, det_init='det_n1.pt', emitter_steps=25000, emitter_scenes=48, emitter_size=64)


class FourPiPlainSetup(FourPiSetup):
    """plain 4Pi (4Pi-SMS): no SIM pattern, windows of fourpi.window frames"""
    name = 'fourpi_plain'

    def defaults(self):
        return dict(fourpi=dict(DEFAULTS, window=1, pattern=False, delta_deg=0.0, emitter=None, det_batch=120, tblk=200),
                    fourpi_train=dict(TRAIN_DEFAULTS))

    def cell_spec(self, c):
        """the acquisition as a CellSpec of the scene simulator (simcode/simulate/multichannel.py): the registration map,
        the channel crop size, the field of fourpi.xlim x all rows, fourpi_train.bg"""
        from simcode.simulate.multichannel import CellSpec
        L = self.layout(c)
        if c['registration']['mode'] != 'file': raise ValueError('fourpi_plain training: registration.mode file (one map)')
        t = c['fourpi_train']
        if not t['bg']: raise ValueError('fourpi_plain training: fourpi_train.bg (background per channel, e/px/frame) is required')
        return CellSpec(c['name'], c['registration']['map'], L.W, L.H, tuple(c['fourpi']['xlim']), (0, L.H), tuple(t['bg']))

    def sim_kw(self, c):
        """MultiChannelAcquisitionSim constructor and sample() arguments of the training recipe"""
        t = c['fourpi_train']
        init = dict(cells=[self.cell_spec(c)], patterns=False, T=c['fourpi']['window'], zlut=tuple(t['zlut']),
                    reg_shift_sd=t['reg_shift_sd'], reg_lin_sd=t['reg_lin_sd'], n_psf=t['n_psf'])
        smp = dict(dens=tuple(t['dens']), zr=tuple(t['zr']), F_med=t['F_med'], F_sd=t['F_sd'], F_rng=tuple(t['F_rng']),
                   p_through=t['p_through'], rn=c['camera']['readnoise_e'], offerr=t['offerr'], bg_rng=tuple(t['bg_rng']))
        return init, smp

    def validate(self, c):
        f = c['fourpi']
        if not c.get('channels') or len(c['channels']) != 4:
            raise ValueError('setup fourpi_plain: four channels (the interference channels) are required in channels')
        if f['pattern']: raise ValueError('setup fourpi_plain: no SIM pattern (fourpi.pattern false; 4Pi-SIMFLUX: setup fourpi)')
        if f['window'] < 1: raise ValueError('setup fourpi_plain: fourpi.window >= 1')
        for k in ('psf', 'detector', 'emitter'):
            if not f[k]: raise ValueError(f'setup fourpi_plain: fourpi.{k} is required')

    def delta(self, c):
        return (0.0,)

    def solve_kw(self, c, net):
        zm = net.prior.zmax
        return dict(k=None, delta=(0.0,), ang=(0,) * c['fourpi']['window'], zr=(-zm, zm))


def sys_python():
    import sys
    return sys.executable
