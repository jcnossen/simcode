"""4Pi-SIMFLUX set-up: four interference channels of one sCMOS camera (two of them mirrored), Hamamatsu dcimg files (a movie
= several files; every file starts at pattern step 0), 6-frame windows (2 angles x 3 steps), the coherent two-arm 4Pi PSF,
the 4Pi detector (registered 4-channel sum) and emitter network (24 unwarped channel patches), and per file: the channel
registration (stage registration), the SIM pattern and the 4Pi phase field (simcode/processing/fourpi_pattern_phase.py)
and the explicit 4Pi solve (simcode/solve/fourpi.py). The acquisitions: simcode/pipeline/configs/fourpi.py.

Config section 'fourpi' (defaults below), besides the common keys (camera.gain_e_per_adu, camera.readnoise_e,
camera.offset_map, optics.pixel_um, channels, registration):
  frame_dt_s    frame time (s);  xlim  illuminated columns [x0, x1) of a channel crop;  patch  candidate patch (px)
  delta_deg     4Pi phase offset of the second pattern angle (deg)
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

DEFAULTS = dict(frame_dt_s=5.575e-3, xlim=[55, 305], patch=15, delta_deg=None, zstack=None, bead_align=None, psf=None,
                pattern_ref=None, detector=None, emitter='v10', files=None, nframes=0,
                chunk=600, p_min=0.5, tile=12.0, tblk=50, dnll=9.21, chi2_max=2.5, coh=1.0, iters=15, batch=1500,
                min_photons=300.0, regions='', nphi=60000, tag='', claim=0, redo=0, order='fwd', min_free_gb=10.0)


def file_index(fn):
    return int(os.path.basename(fn).split('_')[-1].split('.')[0])


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
        if c['pattern']['orientations'] * c['pattern']['steps'] != 6:
            raise ValueError('setup fourpi: 6-frame windows (2 angles x 3 steps)')

    def stages(self, c):
        return ['registration', 'process', 'report']

    # ---- files and readers
    def movie_files(self, c):
        import glob
        return sorted(glob.glob(c['movie']))

    def files(self, c):
        """the file numbers to process"""
        fs = sorted(file_index(f) for f in self.movie_files(c))
        return fs if c['fourpi']['files'] is None else [f for f in fs if f in set(c['fourpi']['files'])]

    def movie_file(self, c, fi):
        fs = [f for f in self.movie_files(c) if file_index(f) == fi]
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


def sys_python():
    import sys
    return sys.executable
