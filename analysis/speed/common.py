"""Shared helpers of the analysis/speed scripts: datasets, GPU-state record, NVML-free utilisation sampler (nvidia-smi)."""
import os, subprocess, threading, time, json
import numpy as np

DATA = {
    # sparse: 1809/2 (the 0.5 nM-like DNA-PAINT movie); calib + pattern of the production run (read-only)
    '1809': dict(movie='/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simflux_5.5ms_500mW.tif', calib='insitu/calib_1809_2.json',
                 pattern='/home/jc/simcode2/out/paper/1809_2/v2/run_default/pattern_est.pt', f0=12000),
    # dense: 20 nM movie 2
    '20nM': dict(movie='/mnt/data/dmdsf/bn_dnapaint/dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif', calib='insitu/calib_20nM_m2_sc.json',
                 pattern='/home/jc/simcode2/out/full_20nM_m2_fast/pattern_est.pt', f0=12000),
}
SC_CAL = 'calib/calib_sc_hd_sc5nbb15_pm_bright_real.json'
P2_CAL = 'calib/calib_p2L_b10_real.json'
OUT = '/home/jc/simcode2/out/speed'


def gpu_index():
    """nvidia-smi index of the (first) visible GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID assumed, as nvidia-smi)"""
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    return int(v)


def gpu_state():
    """all GPUs: utilisation, memory, compute pids (a snapshot to store with each measurement)"""
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return {int(l.split(',')[0]): dict(util=int(l.split(',')[1]), mem_mb=int(l.split(',')[2])) for l in q}
    except Exception as e:
        return {'error': str(e)}


class UtilSampler:
    """samples utilization.gpu of one GPU every `ms` ms (nvidia-smi -lms) during a with-block; .summary() -> mean / samples"""
    def __init__(self, idx=None, ms=100):
        self.idx = gpu_index() if idx is None else idx; self.ms = ms; self.vals = []

    def __enter__(self):
        self.p = subprocess.Popen(['nvidia-smi', '-i', str(self.idx), '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                                   '-lms', str(self.ms)], stdout=subprocess.PIPE, text=True)
        self.th = threading.Thread(target=self._rd, daemon=True); self.th.start(); return self

    def _rd(self):
        for l in self.p.stdout:
            try: self.vals.append(int(l.strip()))
            except ValueError: pass

    def __exit__(self, *a):
        self.p.terminate(); self.p.wait(); self.th.join(timeout=2)

    def summary(self):
        v = np.array(self.vals[1:], float)
        if not len(v): return dict(mean=None, n=0)
        return dict(mean=round(float(np.nanmean(v)), 1), p10=float(np.nanpercentile(v, 10)), n=len(v))


def canon(R):
    o = np.lexsort((R['y_det'], R['x_det'], R['wstart']))
    return {k: (v[o] if np.ndim(v) and len(v) == len(o) else v) for k, v in R.items()}


def qstats(d):
    d = np.abs(np.asarray(d, np.float64)).ravel()
    if not len(d): return {}
    return dict(med=float(np.median(d)), p99=float(np.quantile(d, 0.99)), p999=float(np.quantile(d, 0.999)), max=float(d.max()),
                frac_gt_1e3=float((d > 1e-3).mean()))


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, default=float)
