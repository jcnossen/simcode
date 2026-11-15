"""
Tests of the run record (simcode/util/runconfig.py: <out>/run_config.json).
Run: python tests/test_runconfig.py            CPU tests (no GPU, no models, no raw data)
     python tests/test_runconfig.py --gpu      + the real check (needs a GPU, the m1809 model files and the 1809_2 movie, ~4 min):
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


def _settings(argv):
    a, N, cal = RUN.resolve(RUN.build_parser(), argv); a.model_name = N['name']
    return RUN.settings_of(a, cal)


def test_settings_roundtrip():
    S = _settings(['--dataset', '1809_2', '--out', 'x', '--nframes', '12000', '--sigma_max', '14', '--sig_u', ''])
    assert S['sigma_max'] == 14 and 'u_max' not in S and 'strict_q' not in S and S['subset'] == '14' and S['sig_u'] == '' and os.path.isabs(S['det']) and os.path.isabs(S['cal'])
    S2 = _settings(RC.rerun_argv(dict(settings=S)) + ['--out', 'y'])
    for k in S:
        if k not in ('out', 'dataset'): assert S[k] == S2[k], (k, S[k], S2[k])
    S = _settings(['--dataset', '2.5nM_2', '--out', 'x'])                 # defaults: the model's sig_u, both passes
    assert S['model'] == 'm25' and S['sig_u'].count(',') == 1 and 'sigu_p2_m25' in S['sig_u']
    S2 = _settings(RC.rerun_argv(dict(settings=S)) + ['--out', 'y', '--chunk', '3000'])      # an argument after the record overrides
    assert S2['chunk'] == 3000 and all(S[k] == S2[k] for k in S if k not in ('out', 'dataset', 'chunk'))
    # --model is required (no single-file options); an old record's rule-named subsets and removed keys are translated / dropped
    for bad in (['--movie', 'm.tif', '--calib', 'c.json', '--out', 'x'], ['--dataset', '1809_2', '--out', 'x', '--det', 'd.pt'],
                ['--dataset', '1809_2', '--out', 'x', '--netset', 'm25'], ['--dataset', '1809_2', '--out', 'x', '--pass2']):
        try: _settings(bad); raise AssertionError(bad)
        except SystemExit: pass
    old = dict(settings=dict(S, subset='main,u,strict', u_max=0.45, strict_q=None, det='/x/det.pt'))
    av = RC.rerun_argv(old); assert av[av.index('--subset') + 1] == '12' and '--u_max' not in av and '--det' not in av
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
        S = _settings(['--dataset', '1809_2', '--out', d])
        R = RC.start(d, S, dict(cand_pmin=0.3), dict(name='m', files={}), dict(path=''), dict(path=''), argv=['--out', d])
        assert R['status'] == 'running' and len(R['invocations']) == 1 and os.path.exists(f'{d}/run_config.json')
        R = RC.start(d, S, dict(cand_pmin=0.3), dict(name='m', files={}), dict(path=''), dict(path=''), argv=['--out', d])
        assert len(R['invocations']) == 2 and 'earlier_settings' not in R                    # resumed / second copy
        R = RC.start(d, dict(S, crop='0,0,120'), dict(cand_pmin=0.3), dict(name='m', files={}), dict(path=''), dict(path=''), argv=['--out', d])
        assert len(R['invocations']) == 3 and R['earlier_settings'][0]['settings']['crop'] == '5,5,110'
        SP.main([d, '--tag', 'est2', '--sigma_max', '14', '--subset', '14,12'])
        R = RC.load(d); st = R['steps'][-1]
        assert st['step'] == 'strict' and st['settings']['sigma_max'] == 14 and st['settings']['tag'] == 'est2' and st['settings']['crop'] == [5.0, 5.0, 110.0]
        assert R['outputs']['locs_p2.npz']['written_by'] == 'steps[0]' and 'locs_p2_sig14.npz' in R['outputs'] and 'locs_p2_sig12.npz' in R['outputs']
        assert R['selection']['locs_p2']['main']['sigma_max_nm'] == 14 and R['selection']['locs_p2']['main']['n'] == st['result']['n_main']
        assert R['outputs']['locs_p2.npz']['arrays_sha256'] == RC.arrays_sha256(f'{d}/locs_p2.npz')
        assert 'simcode.table' in open(f'{d}/command.txt').read()
        R = RC.finish(d, error='X: test'); assert R['status'] == 'failed'
        if os.path.exists(f'{d}/run_config.yaml'):
            import yaml
            assert yaml.safe_load(open(f'{d}/run_config.yaml'))['steps'][0]['step'] == 'strict'
    with tempfile.TemporaryDirectory() as d:                                             # a directory without a record
        _run_dir(d); SP.main([d, '--tag', 'est2'])
        R = RC.load(d); assert R['status'] == 'no run record' and len(R['steps']) == 1
    print('  ok start (invocations, earlier_settings), strict step (settings, outputs, selection), command.txt, failed status')


def gpu_rerun(keep=''):
    """2 chunks of 1809_2 -> run_config.json; re-run from it -> bit-identical outputs"""
    import torch
    from compare_npz import compare_files
    with tempfile.TemporaryDirectory(dir=keep or None) as d:
        py = [sys.executable, '-m', 'simcode.run']; env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.run(py + ['--dataset', '1809_2', '--nframes', '12000', '--out', f'{d}/a'], check=True, env=env, stdout=subprocess.DEVNULL)
        A = RC.load(f'{d}/a')
        assert A['status'] == 'done' and A['movie']['chunks'] == [0, 6000] and A['model']['name'] == 'm1809' and A['code']['git_commit']
        assert all(len(v['sha256']) == 64 for v in A['model']['files'].values()) and len(A['calib']['sha256']) == 64 and len(A['movie']['parts'][0]['sha256']) == 64
        assert A['selection']['locs_p2']['counts']['n_rows'] > 0 and A['pattern']['angles']['0']['pitch_px'] > 0 and A['environment']['torch'] == torch.__version__
        subprocess.run(py + ['--config', f'{d}/a/run_config.json', '--out', f'{d}/b'], check=True, env=env, stdout=subprocess.DEVNULL)
        B = RC.load(f'{d}/b')
        assert B['rerun_of']['sha256'] == RC.sha256(f'{d}/a/run_config.json')
        fs = sorted(f for f in os.listdir(f'{d}/a') if f.endswith('.npz'))
        assert len(fs) >= 12                                 # 8 chunk files, 2 tables, 2 subset files (sig12)
        for f in fs:
            bad = compare_files(f'{d}/a/{f}', f'{d}/b/{f}', skip=('pattern',)); assert not bad, (f, bad)
        Pa, Pb = torch.load(f'{d}/a/pattern_est.pt'), torch.load(f'{d}/b/pattern_est.pt')
        assert all(np.array_equal(np.asarray(Pa[k]), np.asarray(Pb[k])) for k in Pa)
        assert all(A['outputs'][k]['arrays_sha256'] == B['outputs'][k]['arrays_sha256'] for k in A['outputs'])
        assert {k: v for k, v in A['settings'].items() if k not in ('out', 'dataset')} == {k: v for k, v in B['settings'].items() if k not in ('out', 'dataset')}
        print(f'  ok run -> run_config.json -> re-run with --config: {len(fs)} npz files + pattern bit-identical, recorded table hashes equal')


if __name__ == '__main__':
    for f in (test_settings_roundtrip, test_hashes, test_steps_and_start):
        print(f.__name__); f()
    if '--gpu' in sys.argv:
        print('gpu_rerun'); gpu_rerun()
    print('ALL OK')
