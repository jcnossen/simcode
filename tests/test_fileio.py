"""
CPU tests of the shared file handling (simcode/io) and of the configuration names. Run: python tests/test_fileio.py
(or pytest tests). No GPU, no models, no raw data: every input is written into a temporary directory.
  movie readers : multi-part TIFF frame index across the part boundary and crops; dcimg header, channel crops with
                  mirroring, footer; array / .npy reader; Prefetch hits, misses and identical data
  work driver   : atomic writes (a leftover temporary file is neither 'done' nor matched by '*.npz'), resume skips finished
                  units, claims are exclusive between processes, a claim of a dead process is taken over, a killed writer
                  leaves no output, per-unit records, the claim loop ends only when all units are done
  config        : models resolve (a name of models.json or a model folder); the pre-rename names (SIMCODE2_* variables,
                  simcode2.json, netset) are no longer read
"""
import glob, json, multiprocessing as mp, os, signal, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode.io import movie as MV, work as WK
from simcode.psf import ArrayMovie
from simcode.channels.layout import Channel, ChannelLayout, ChannelReader


def _tif(path, a):
    import tifffile
    tifffile.imwrite(path, a, photometric='minisblack')


def test_tiff_movie():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4000, (17, 12, 10)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        _tif(f'{d}/m.tif', a[:7]); _tif(f'{d}/m_X1.tif', a[7:13]); _tif(f'{d}/m_X2.tif', a[13:])
        m = MV.open_movie(f'{d}/m.tif')
        assert m.n == 17 and tuple(m.shape) == (12, 10) and len(m.parts) == 3
        assert np.array_equal(m.read(0, 17), a.astype(np.float32))                      # whole frames
        assert np.array_equal(m.read(5, 15, 2, 1, 8), a[5:15, 2:10, 1:9].astype(np.float32))   # across both part boundaries, cropped
        assert m.read(6, 8, 0, 0, 4).dtype == np.float32
    print('  ok TIFF reader (3 parts, boundaries, crop)')


def _write_dcimg(path, frames, footer_words=128):
    """a minimal dcimg file with the header fields the reader uses (format 0x2000000)"""
    n, nrow, ncol = frames.shape
    bpi = nrow * ncol * 2; bpf = bpi + 4 * footer_words; s0 = 800; off_data = 400
    h = np.zeros(200, '<u4'); h[2] = 0x2000000; h[9] = n; h[10] = s0
    s = np.zeros(100, '<u4'); s[18] = ncol; s[19] = nrow; s[20] = 2 * ncol; s[21] = bpi; s[24] = off_data; s[29] = bpf
    with open(path, 'wb') as f:
        f.write(h.tobytes()); f.write(s.tobytes())
        assert f.tell() == s0 + off_data
        for i in range(n):
            f.write(frames[i].astype('<u2').tobytes())
            ft = np.zeros(footer_words, '<u4'); ft[0] = 1000 + i; ft[1] = 1700000000 + i; ft[2] = 250000
            f.write(ft.tobytes())


def test_dcimg_movie():
    rng = np.random.default_rng(1)
    fr = rng.integers(0, 60000, (9, 6, 40)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        _write_dcimg(f'{d}/x.dcimg', fr)
        m = MV.open_movie(f'{d}/x.dcimg')
        assert m.n == 9 and m.shape == (6, 40)
        assert np.array_equal(m.raw(2, 5), fr[2:7]) and m.raw(7, 100).shape[0] == 2          # raw frames, clipped at the end
        assert np.array_equal(m.read(0, 9), fr.astype(np.float32))
        assert int(m.footer(4)[0]) == 1004 and abs(m.time_s(4) - (1700000004 + 0.25)) < 1e-6
        lay = ChannelLayout([Channel(name=f'c{i}', crop=(x0, 0, 8, 6), mirror=m) for i, (x0, m) in enumerate(zip((2, 12, 22, 31), ('none', 'none', 'lr', 'lr')))])
        c = ChannelReader(MV.DcimgMovie(f'{d}/x.dcimg'), lay)
        assert c.shape == (4, 6, 8) and c.n == 9
        x = c.read(1, 6)
        assert x.shape == (5, 4, 6, 8) and x.dtype == np.float32
        assert np.array_equal(x[:, 0], fr[1:6, :, 2:10]) and np.array_equal(x[:, 1], fr[1:6, :, 12:20])
        assert np.array_equal(x[:, 2], fr[1:6, :, 22:30][..., ::-1]) and np.array_equal(x[:, 3], fr[1:6, :, 31:39][..., ::-1])   # mirrored
        assert np.array_equal(lay.split(fr[1:6]), x)
        # interleaved pages (two-colour TIFFs): channel k = page k of each frame, own crop, up-down mirror
        l2 = ChannelLayout([Channel(name='a', crop=(0, 0, 8, 6)), Channel(name='b', crop=(30, 0, 8, 6), mirror='ud', page=1)])
        y = l2.split(fr[2:8])
        assert y.shape == (3, 2, 6, 8) and np.array_equal(y[:, 0], fr[2:8:2, :, :8]) and np.array_equal(y[:, 1], fr[3:8:2, ::-1, 30:38])
        assert ChannelReader(ArrayMovie(fr), l2).n == 4
    print('  ok dcimg reader (header, raw, footer) and the channel layout (crops, mirror, interleaved pages)')


def test_array_movie_and_prefetch():
    a = np.arange(40 * 9 * 9, dtype=np.float64).reshape(40, 9, 9)
    with tempfile.TemporaryDirectory() as d:
        np.save(f'{d}/a.npy', a)
        for m in (ArrayMovie(a), MV.open_movie(f'{d}/a.npy')):
            assert m.n == 40 and m.shape == (9, 9)
            assert np.array_equal(m.read(3, 9, 1, 2, 5), a[3:9, 1:6, 2:7].astype(np.float32))
            plan = [(0, 10, 0, 0, 9), (10, 20, 0, 0, 9), (20, 30, 0, 0, 9), (30, 40, 0, 0, 9)]
            pm = MV.Prefetch(m, plan)
            for k in plan + [plan[1]]:                                                  # 4 hits, then 1 miss
                assert np.array_equal(pm.read(*k), m.read(*k))
            assert pm.stats['hits'] == 4 and pm.stats['misses'] == 1 and pm.n == 40
            pm = MV.Prefetch(m, plan); assert np.array_equal(pm.read(*plan[2]), m.read(*plan[2])); pm.close()   # miss with a read in flight
    print('  ok array / .npy reader, Prefetch (hits, misses, identical data)')


def test_atomic_and_resume():
    with tempfile.TemporaryDirectory() as d:
        U = WK.Units(d, 'pass1', title=False)
        assert U.path(6000) == f'{d}/pass1_6000.npz' and U.record_path(6000) == f'{d}/pass1_6000.json'
        open(U.path(0) + '.tmp', 'wb').write(b'truncated')                      # what a killed writer leaves behind
        assert not U.done(0) and glob.glob(f'{d}/pass1_*.npz') == []
        ran = []
        for k in U.todo([0, 6000, 12000]):
            ran.append(k); U.save(k, x=np.arange(3) + k); U.record(k, dict(n=3, t=0.5))
        assert ran == [0, 6000, 12000] and all(U.done(k) for k in ran)
        assert np.array_equal(np.load(U.path(6000))['x'], np.arange(3) + 6000)
        assert list(U.todo([0, 6000, 12000, 18000])) == [18000]                 # resume: only the missing unit
        assert set(U.records([0, 6000, 12000, 18000])) == {0, 6000, 12000} and U.records([0])[0]['n'] == 3
        R = WK.Units(d, 'pass1', redo=True, title=False)
        assert list(R.todo([0, 6000])) == [0, 6000] and list(R.todo([0, 6000])) == []     # redo: every unit once
        U.fail(18000, RuntimeError('x')); assert list(U.todo([18000])) == []    # failed units are skipped
        V = WK.Units(d, 'cl', pattern='{stage}_{cell}_{key:03d}{tag}.npz', cell='NPC', tag='_wv', title=False)
        assert os.path.basename(V.path(7)) == 'cl_NPC_007_wv.npz'
        try:
            WK.disk_guard(d, 1e9); assert False
        except WK.DiskFull:
            pass
        WK.disk_guard(d, 0)
        WK.atomic_json(f'{d}/a.json', dict(a=1)); assert json.load(open(f'{d}/a.json')) == dict(a=1) and not os.path.exists(f'{d}/a.json.tmp')
    print('  ok atomic writes, resume, redo, failed marker, records, disk guard')


def _worker(d, keys, q):
    U = WK.Units(d, 's', claim=True, poll_s=0.05, title=False, log=lambda *s: None)
    mine = []
    for k in U.todo(keys):
        time.sleep(0.03); mine.append(k); U.save(k, pid=np.array(os.getpid()))
    q.put(mine)


def test_claims_two_processes():
    keys = list(range(24))
    with tempfile.TemporaryDirectory() as d:
        ctx = mp.get_context('fork'); q = ctx.Queue()
        ps = [ctx.Process(target=_worker, args=(d, keys, q)) for _ in range(3)]
        for p in ps: p.start()
        got = [q.get(timeout=60) for _ in ps]
        for p in ps: p.join()
        flat = sorted(k for g in got for k in g)
        assert flat == keys, f'units processed {flat}'                          # every unit exactly once
        assert sum(1 for g in got if g) >= 2, 'only one worker did any work'
        assert glob.glob(f'{d}/*.claim') == []
    print(f'  ok claims: 3 processes, 24 units, each exactly once ({[len(g) for g in got]})')


def test_stale_claim_and_barrier():
    with tempfile.TemporaryDirectory() as d:
        U = WK.Units(d, 's', claim=True, poll_s=0.05, title=False, log=lambda *s: None)
        p = subprocess.Popen([sys.executable, '-c', 'pass']); p.wait()          # a pid that no longer exists
        json.dump(dict(host=__import__('socket').gethostname(), pid=p.pid, start=1, t='x'), open(U.path(1) + '.claim', 'w'))
        assert U.claimed(1) is None and U.claim(1)                              # dead owner: taken over
        U.release(1); assert not os.path.exists(U.path(1) + '.claim')
        # a live owner (this test's parent shell is not us: use a sleeping child) blocks the claim; the loop waits for its output
        ch = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        st = WK._proc_start(ch.pid)
        json.dump(dict(host=__import__('socket').gethostname(), pid=ch.pid, start=st, t='x'), open(U.path(2) + '.claim', 'w'))
        assert not U.claim(2) and U.claimed(2)['pid'] == ch.pid
        t0 = time.time(); ran = []
        import threading
        threading.Timer(0.5, lambda: (ch.kill(), ch.wait())).start()            # the owner dies after 0.5 s without an output
        for k in U.todo([2]): ran.append(k); U.save(k, x=np.zeros(1))
        assert ran == [2] and time.time() - t0 > 0.4                            # waited, then took the unit over
    print('  ok stale claim taken over; live claim respected; the claim loop waits until every unit is done')


def test_killed_writer_leaves_no_output():
    with tempfile.TemporaryDirectory() as d:
        code = ("import sys, numpy as np; sys.path.insert(0, %r)\nfrom simcode.io.work import atomic_savez\n"
                "a = np.random.default_rng(0).random((60, 1000, 1000))\nprint('go', flush=True)\n"
                "atomic_savez(%r, a=a, b=a, c=a, d=a)\n" % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'{d}/big.npz'))
        p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE)
        p.stdout.readline(); time.sleep(0.3); p.send_signal(signal.SIGKILL); p.wait()
        assert not os.path.exists(f'{d}/big.npz'), 'the writer finished before it was killed (make the array larger)'
        assert WK.Units(d, 'big', pattern='big.npz', title=False).done('') is False
    print('  ok kill -9 during a write: no output file')


def test_config_names():
    """no path settings: no config file / environment lookup; model and calibration names resolve under <repo>/models"""
    from simcode import config
    for k in ('root', 'out_dir', 'env', 'models_dir'): assert not hasattr(config, k), k
    assert config.MODELS == os.path.join(str(config.REPO), 'models')
    f = sorted(os.listdir(os.path.join(config.MODELS, 'calib')))[0]
    assert config.resolve('calib/' + f) == config.resolve(f) == os.path.join(config.MODELS, 'calib', f)
    assert config.resolve(config.CAL_SIM) == os.path.join(config.MODELS, config.CAL_SIM)
    with tempfile.TemporaryDirectory() as d:
        assert config.resolve(d) == d and config.resolve('') == ''
    from simcode import config
    S = config.model_sets()
    sets = [k for k in S if not k.startswith('_')]
    assert S['_default'] in sets and config.model_set()['name'] == S['_default']
    for n in sets:
        N = config.model_set(n)
        assert all(k in N for k in ('detector', 'emitter', 'cal_sim', 'detector_pass2', 'emitter_pass2', 'cal_sim_pass2', 'psf')) and 'sig_u' not in N
        assert not set(N) & set(config.REMOVED_KEYS), (n, N)                        # no real-data sigma calibrations (removed 2026-11-17)
        for k in ('cal_sim', 'cal_sim_pass2'):                                       # the calibration json files are in the repository
            assert os.path.exists(os.path.join(config.REPO, 'models', N[k])), (n, N[k])
    for ds, D in config.production()['datasets'].items():
        assert D['model'] is None or D['model'] in sets
        assert os.path.exists(os.path.join(config.REPO, 'models', D['calib'])), D['calib']
    try:
        config.model_set('default'); assert False
    except KeyError:
        pass
    # a model folder: model.json with one model, file names relative to the folder (absolute ones kept)
    with tempfile.TemporaryDirectory() as d:
        M = dict(config.model_set('m1809')); M.pop('name')
        json.dump({'mfold': dict(M, detector='det_x.pt', emitter=os.path.abspath(config.model(M['emitter']))), '_recipe': {}}, open(f'{d}/model.json', 'w'))
        for p in (d, f'{d}/model.json'):
            F = config.model_set(p)
            assert F['name'] == 'mfold' and F['detector'] == os.path.join(d, 'det_x.pt') and F['emitter'] == os.path.abspath(config.model(M['emitter'])) and F['folder'] == d
        json.dump({'old': dict(M, det='det_x.pt')}, open(f'{d}/old.json', 'w'))      # the key names before 2026-11-15 are refused
        try: config.model_set(f'{d}/old.json'); assert False
        except ValueError: pass
        json.dump({'rr': dict(M, cal_real='calib/x_real.json')}, open(f'{d}/rr.json', 'w'))   # the real-data sigma calibrations (removed 2026-11-17) are refused
        try: config.model_set(f'{d}/rr.json'); assert False
        except ValueError: pass
        json.dump({'a': M, 'b': M}, open(f'{d}/two.json', 'w'))
        try: config.model_set(f'{d}/two.json'); assert False
        except ValueError: pass
    try: config.model_set('/nonexistent/model'); assert False
    except KeyError: pass
    print(f'  ok config: models {sets}, default {S["_default"]}; model folders; names resolve under {config.MODELS}')


if __name__ == '__main__':
    for f in (test_tiff_movie, test_dcimg_movie, test_array_movie_and_prefetch, test_atomic_and_resume, test_claims_two_processes,
              test_stale_claim_and_barrier, test_killed_writer_leaves_no_output, test_config_names):
        print(f.__name__, flush=True); f()
    print('ALL OK')
