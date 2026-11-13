"""Summary of the published localization tables (Picasso-style hdf5 'locs') and pick files of the 20 nm / 40 nm nanoruler
movies. Reads only the small tables in results/. Output: out/nanoruler/r2040/tables.json"""
import sys, json, os, h5py, yaml, numpy as np
B = '/mnt/data/SIMFLUX/data/fig2/'
PX = 65.0
SETS = {'nr20': B + '20nm-panel-s-t/results/1_2_MMStack_Pos0.ome/', 'nr40': B + '40nm-panel-q-r/results/3Phase_3_MMStack_Pos0.ome/'}
FILES = ['g2d-nodriftc.hdf5', 'g2d.hdf5', 'g2d_render.hdf5', 'g2d-linked.hdf5', 'simflux-nodriftc.hdf5', 'simflux.hdf5', 'simflux_render.hdf5', 'simflux-linked.hdf5', 'smlm.hdf5']
def load_picks(p):
    # the pick files store numpy float64 scalars as tagged binary; decode them without unpickling
    import re, base64, struct
    t = open(p).read()
    if 'numpy.core.multiarray.scalar' not in t: return yaml.safe_load(t)
    vals = [struct.unpack('<d', base64.b64decode(m))[0] for m in re.findall(r'!!binary \|\s+(\S+=)\s', t) if len(base64.b64decode(m)) == 8]
    y = {'Centers': np.array(vals).reshape(-1, 2).tolist()}
    for k in ('Diameter', 'Shape'):
        m = re.search(r'^%s: (.*)$' % k, t, re.M)
        if m: y[k] = m.group(1)
    return y
out = {}
for name, d in SETS.items():
    o = {}
    ref = {}
    for f in FILES:
        if not os.path.exists(d + f): o[f] = 'MISSING'; continue
        with h5py.File(d + f, 'r') as h:
            keys = list(h.keys()); L = h['locs'][:]
        e = {'h5keys': keys, 'n': int(len(L)), 'fields': list(L.dtype.names)}
        fr = L['frame']; e['frame_min'] = int(fr.min()); e['frame_max'] = int(fr.max()); e['frames_unique'] = int(len(np.unique(fr)))
        e['per_frame_value'] = float(len(L) / (fr.max() + 1))
        for k in L.dtype.names:
            if k in ('frame',): continue
            v = L[k].astype(float); e[k] = [float(q) for q in np.nanpercentile(v, [5, 50, 95])]
        for k in ('lpx', 'lpy'):
            if k in L.dtype.names: e[k + '_nm_median'] = float(np.nanmedian(L[k]) * PX)
        e['x_range'] = [float(L['x'].min()), float(L['x'].max())]; e['y_range'] = [float(L['y'].min()), float(L['y'].max())]
        ref[f] = L
        o[f] = e
        print(name, f, e['n'], 'frames %d..%d' % (e['frame_min'], e['frame_max']), 'photons med %.0f' % e.get('photons', [0, 0, 0])[1],
              'lpx med %.4f px = %.2f nm' % (e.get('lpx', [0, 0, 0])[1], e.get('lpx_nm_median', 0)), 'lpy %.4f' % e.get('lpy', [0, 0, 0])[1], 'bg med %.2f' % e.get('bg', [0, 0, 0])[1], e['fields'], flush=True)
    # drift applied between the tables (same rows?)
    for a_, b_ in (('g2d-nodriftc.hdf5', 'g2d.hdf5'), ('g2d.hdf5', 'g2d_render.hdf5'), ('simflux-nodriftc.hdf5', 'simflux.hdf5'), ('simflux.hdf5', 'simflux_render.hdf5'), ('g2d.hdf5', 'simflux.hdf5')):
        if a_ in ref and b_ in ref and len(ref[a_]) == len(ref[b_]):
            dx = ref[b_]['x'].astype(float) - ref[a_]['x']; dy = ref[b_]['y'].astype(float) - ref[a_]['y']
            same = bool((ref[a_]['frame'] == ref[b_]['frame']).all())
            o['diff %s -> %s' % (a_, b_)] = {'same_frames': same, 'dx_rms_px': float(dx.std()), 'dy_rms_px': float(dy.std()), 'dx_range': [float(dx.min()), float(dx.max())], 'dy_range': [float(dy.min()), float(dy.max())]}
            print(name, 'diff', a_, b_, o['diff %s -> %s' % (a_, b_)])
        elif a_ in ref and b_ in ref: print(name, 'len differs', a_, len(ref[a_]), b_, len(ref[b_]))
    for f in sorted(os.listdir(d)):
        if 'pick' in f and f.endswith('.yaml'):
            y = load_picks(d + f)
            e = {k: v for k, v in y.items() if k != 'Centers'} if isinstance(y, dict) else {}
            c = np.array(y.get('Centers', [])) if isinstance(y, dict) else np.zeros((0, 2))
            e['n_picks'] = int(len(c))
            if len(c): e['x_range'] = [float(c[:, 0].min()), float(c[:, 0].max())]; e['y_range'] = [float(c[:, 1].min()), float(c[:, 1].max())]
            if isinstance(y, dict): e['keys'] = list(y.keys())
            if len(c): e['centers'] = c.round(3).tolist()
            o[f] = e; print(name, f, {k: v for k, v in e.items() if k != 'centers'}, flush=True)
    for f in ('drift.txt',):
        if os.path.exists(d + f):
            D = np.loadtxt(d + f); o[f] = {'shape': list(D.shape), 'range_px': [[float(D[:, i].min()), float(D[:, i].max())] for i in range(D.shape[1])]}; print(name, f, o[f])
    for f in sorted(os.listdir(d)):
        if f.endswith('_drift.txt'):
            D = np.loadtxt(d + f); o[f] = {'shape': list(D.shape), 'range_px': [[float(D[:, i].min()), float(D[:, i].max())] for i in range(D.shape[1])]}; print(name, f, o[f])
    out[name] = o
json.dump(out, open('/home/jc/simcode2/out/nanoruler/r2040/tables.json', 'w'), indent=1)
