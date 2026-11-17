"""
Tests of the run record (simcode/util/runconfig.py: <out>/run_config.json).
Run: python tests/test_runconfig.py            CPU tests (no GPU, no models, no raw data)
     python tests/test_runconfig.py --gpu <data root>   + the real check (needs a GPU, the m1809 model files and the 1809_2 movie
                                               under <data root> (/mnt/data/dmdsf/bn_dnapaint), ~4 min):
                                               simcode.run on 2 chunks of 1809_2 writes the record; the re-run from it
                                               (python -m simcode.run --config <run_config.json> --out <new dir>) gives
                                               bit-identical chunk files, pattern and tables
  settings   the settings of a run turned into arguments (rerun_argv) resolve to the same settings again
  hashes     arrays_sha256 is equal for two .npz files with the same arrays (their file hashes differ: zip time stamps)
  steps      simcode.table appends its settings to run_config.json 'steps', replaces the outputs / selection entries of the
             files it wrote, and logs a plain line in command.txt
  start      a second invocation with the same settings is added to 'invocations'; changed settings are kept under
             'earlier_settings'
"""
import json, os, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode import run as RUN, table as SP
from simcode.util import runconfig as RC
from test_tables import _run_dir
DR = '/nonexistent/data_root'           # --data-root of the CPU tests (the movie is not read)


def _settings(argv):
    a, N, cal = RUN.resolve(RUN.build_parser(), argv); a.model_name = N['name']
    return RUN.settings_of(a, cal)


def test_settings_roundtrip():
    S = _settings(['--dataset', '1809_2', '--data-root', DR, '--out', 'x', '--nframes', '12000'])
    assert S['movie'] == os.path.join(DR, RUN.PROD['datasets']['1809_2']['movie'])                # the dataset's movie under --data-root
    try: _settings(['--dataset', '1809_2', '--out', 'x']); raise AssertionError('--dataset needs --data-root')
    except SystemExit: pass
    assert not set(S) & {'sigma_max', 'subset', 'format', 'prefetch', 'u_max', 'strict_q'} and os.path.isabs(S['detector']) and os.path.isabs(S['cal'])
    assert S['crop'] == '5,5,110' and 'sig_u' not in S                                                 # the dataset's crop; no sigma rescale
    assert _settings(['--movie', 'm.tif', '--calib', 'c.json', '--model', 'm1809', '--out', 'x'])['crop'] == ''   # the whole frame
    S2 = _settings(RC.rerun_argv(dict(settings=S)) + ['--out', 'y'])
    for k in S:
        if k not in ('out', 'dataset'): assert S[k] == S2[k], (k, S[k], S2[k])
    S = _settings(['--movie', 'm.tif', '--calib', 'g13/calib_gauss1.3.psf.zip', '--model', 'g13', '--crop', '5,5,110', '--out', 'x'])
    assert S['model'] == 'g13' and 'sig_u' not in S
    S2 = _settings(RC.rerun_argv(dict(settings=S)) + ['--out', 'y', '--chunk', '3000'])      # an argument after the record overrides
    assert S2['chunk'] == 3000 and all(S[k] == S2[k] for k in S if k not in ('out', 'dataset', 'chunk'))
    # --model is required; a record's settings that are no longer options are not passed on (rerun_argv takes RERUN_KEYS only)
    try: _settings(['--movie', 'm.tif', '--calib', 'c.json', '--out', 'x']); raise AssertionError('--model is required')
    except SystemExit: pass
    old = dict(settings=dict(S, subset='main,u,strict', sigma_max=12.0, format='npz', prefetch=1, sig_u='', u_max=0.45, strict_q=None, det='/x/det.pt', cal_kind='real'))
    av = RC.rerun_argv(old)
    assert not {'--subset', '--sigma_max', '--format', '--prefetch', '--sig_u', '--u_max', '--det', '--cal_kind'} & set(av) and av[av.index('--crop') + 1] == '5,5,110'
    assert 'cal_kind' not in S and '_sim' in S['cal'] and '_sim' in S['cal_pass2']                  # one calibration per pass (2026-11-17)
    # a record of format 1 (file keys det / head / p2_* before 2026-11-15) is read with the present names
    R1 = dict(format='simcode.run_config/1', settings=dict(det='/a.pt', head='/b.pt', p2_det='/c.pt', p2_head='/d.pt', p2_cal='/e.json'),
              fixed=dict(head_batch=1024), model=dict(name='m', definition=dict(det='/a.pt', p2_cal_real='/e.json'), files=dict(det={}, sig_u_p1={})))
    U = RC.upgrade(R1)
    assert U['format'] == RC.FORMAT and U['settings'] == dict(detector='/a.pt', emitter='/b.pt', detector_pass2='/c.pt', emitter_pass2='/d.pt', cal_pass2='/e.json')
    assert U['model']['definition'] == dict(detector='/a.pt', cal_real_pass2='/e.json') and set(U['model']['files']) == {'detector', 'sig_u_pass1'}
    assert U['fixed'] == dict(emitter_batch=1024) and RC.upgrade(U) == U
    F = RUN.fixed_values()
    assert F['cand_pmin'] == 0.3 and F['pass2'] is True and F['det_batch'] == 64 and F['pattern_pitch_search_px'] == [1.6, 2.8]
    print('  ok settings -> arguments -> the same settings; fixed values')


def test_hashes():
    with tempfile.TemporaryDirectory() as d:
        a = dict(x=np.arange(10.0), y=np.arange(5, dtype=np.int32))
        np.savez(f'{d}/a.npz', **a); time.sleep(2.1); np.savez(f'{d}/b.npz', y=a['y'], x=a['x'])
        np.savez(f'{d}/c.npz', x=a['x'] + 1, y=a['y'])
        assert RC.arrays_sha256(f'{d}/a.npz') == RC.arrays_sha256(f'{d}/b.npz') != RC.arrays_sha256(f'{d}/c.npz')
        r = RC.file_record(f'{d}/a.npz')
        assert r['size'] > 0 and len(r['sha256']) == 64 and RC.file_record(f'{d}/none')['missing']
    print('  ok arrays_sha256 (content hash of an npz), file_record')


def test_steps_and_start():
    with tempfile.TemporaryDirectory() as d:
        _run_dir(d)
        S = _settings(['--dataset', '1809_2', '--data-root', DR, '--out', d])
        R = RC.start(d, S, dict(cand_pmin=0.3), dict(name='m', files={}), dict(path=''), dict(path=''), argv=['--out', d])
        assert R['status'] == 'running' and len(R['invocations']) == 1 and os.path.exists(f'{d}/run_config.json')
        R = RC.start(d, S, dict(cand_pmin=0.3), dict(name='m', files={}), dict(path=''), dict(path=''), argv=['--out', d])
        assert len(R['invocations']) == 2 and 'earlier_settings' not in R                    # resumed / second copy
        R = RC.start(d, dict(S, crop='0,0,120'), dict(cand_pmin=0.3), dict(name='m', files={}), dict(path=''), dict(path=''), argv=['--out', d])
        assert len(R['invocations']) == 3 and R['earlier_settings'][0]['settings']['crop'] == '5,5,110'
        SP.main([d])
        R = RC.load(d); st = R['steps'][-1]
        assert st['step'] == 'table' and st['settings']['tables'] == list(SP.TABLES) and st['settings']['crop'] == [5.0, 5.0, 110.0]
        assert 'sig_u' not in st['settings']                                                            # no sigma rescale
        assert R['outputs']['locs_p2_net.hdf5']['written_by'] == 'steps[0]' and not [k for k in R['outputs'] if 'sig' in k]
        assert R['selection']['locs_p2_net']['main']['n'] == st['result']['locs_p2_net']['n_main'] and set(R['selection']) == set(SP.TABLES)
        assert R['outputs']['locs_p2_net.hdf5']['arrays_sha256'] == RC.arrays_sha256(f'{d}/locs_p2_net.hdf5')
        assert 'simcode.table' in open(f'{d}/command.txt').read()
        R = RC.finish(d, error='X: test'); assert R['status'] == 'failed'
        if os.path.exists(f'{d}/run_config.yaml'):
            import yaml
            assert yaml.safe_load(open(f'{d}/run_config.yaml'))['steps'][0]['step'] == 'table'
    with tempfile.TemporaryDirectory() as d:                                             # a directory without a record
        _run_dir(d); SP.main([d, '--table', 'locs_p2_net'])
        R = RC.load(d); assert R['status'] == 'no run record' and len(R['steps']) == 1
    print('  ok start (invocations, earlier_settings), table step (settings, outputs, selection), command.txt, failed status')


def gpu_rerun(data_root, keep=''):
    """2 chunks of 1809_2 -> run_config.json; re-run from it -> bit-identical outputs"""
    import torch
    from compare_npz import compare_files
    with tempfile.TemporaryDirectory(dir=keep or None) as d:
        py = [sys.executable, '-m', 'simcode.run']; env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.run(py + ['--dataset', '1809_2', '--data-root', data_root, '--nframes', '12000', '--out', f'{d}/a'], check=True, env=env, stdout=subprocess.DEVNULL)
        A = RC.load(f'{d}/a')
        assert A['status'] == 'done' and A['movie']['chunks'] == [0, 6000] and A['model']['name'] == 'm1809' and A['code']['git_commit']
        assert all(len(v['sha256']) == 64 for v in A['model']['files'].values()) and len(A['calib']['sha256']) == 64 and len(A['movie']['parts'][0]['sha256']) == 64
        assert A['selection']['locs_p2']['counts']['n_rows'] > 0 and A['pattern']['angles']['0']['pitch_px'] > 0 and A['environment']['torch'] == torch.__version__
        subprocess.run(py + ['--config', f'{d}/a/run_config.json', '--out', f'{d}/b'], check=True, env=env, stdout=subprocess.DEVNULL)
        B = RC.load(f'{d}/b')
        assert B['rerun_of']['sha256'] == RC.sha256(f'{d}/a/run_config.json')
        fs = sorted(f for f in os.listdir(f'{d}/a') if f.endswith('.npz'))
        assert len(fs) >= 8                                  # 8 chunk files
        for f in fs:
            bad = compare_files(f'{d}/a/{f}', f'{d}/b/{f}', skip=('pattern',)); assert not bad, (f, bad)
        for f in ('locs_p1.hdf5', 'locs_p2.hdf5'):           # the tables
            assert RC.arrays_sha256(f'{d}/a/{f}') == RC.arrays_sha256(f'{d}/b/{f}'), f
        Pa, Pb = torch.load(f'{d}/a/pattern_est.pt'), torch.load(f'{d}/b/pattern_est.pt')
        assert all(np.array_equal(np.asarray(Pa[k]), np.asarray(Pb[k])) for k in Pa)
        assert all(A['outputs'][k]['arrays_sha256'] == B['outputs'][k]['arrays_sha256'] for k in A['outputs'])
        assert {k: v for k, v in A['settings'].items() if k not in ('out', 'dataset')} == {k: v for k, v in B['settings'].items() if k not in ('out', 'dataset')}
        print(f'  ok run -> run_config.json -> re-run with --config: {len(fs)} npz files + pattern bit-identical, recorded table hashes equal')


if __name__ == '__main__':
    for f in (test_settings_roundtrip, test_hashes, test_steps_and_start):
        print(f.__name__); f()
    if '--gpu' in sys.argv:
        print('gpu_rerun'); gpu_rerun(sys.argv[sys.argv.index('--gpu') + 1])
    print('ALL OK')
