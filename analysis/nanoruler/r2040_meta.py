"""Micro-Manager metadata and page counts of the 20 nm / 40 nm nanoruler movies (and, first part only, of two
supplementary movies). tifffile with is_ome=False (no multi-file OME series). Reads only IFDs / metadata blocks.
Output: out/nanoruler/r2040/meta.json"""
import sys, json, glob, os, tifffile
sys.path.insert(0, '/home/jc/simcode2')
from simcode.util.proctitle import set_title
set_title('nr-facts')
B = '/mnt/data/SIMFLUX/data/'
SETS = {
    'nr20': (B + 'fig2/20nm-panel-s-t/', ['1_2_MMStack_Pos0.ome.tif'] + ['1_2_MMStack_Pos0_%d.ome.tif' % i for i in range(1, 8)], True),
    'nr40': (B + 'fig2/40nm-panel-q-r/', ['3Phase_3_MMStack_Pos0.ome.tif'] + ['3Phase_3_MMStack_Pos0_%d.ome.tif' % i for i in range(1, 4)], True),
    'nr80': (B + 'fig2/80nm-panel-a-e/', ['sim4_1_MMStack_Pos0.ome.tif'], False),
    'sf10': (B + 'Supplementary Figure 10/', ['4Phase_3_MMStack_Pos0.ome.tif'], False),
    'sf19a': (B + 'Supplementary Figure 19/', ['sfig19a_rawframes.tif'], False),
    'sf19d': (B + 'Supplementary Figure 19/', ['sfig19d_rawframes.tif'], False),
}
KEYS = ('Elapsed', 'Exposure', 'Binning', 'Camera', 'ROI', 'Frame', 'Time', 'PixelSize', 'Core-', 'TRIGGER', 'Trigger', 'ScanMode', 'Sensor', 'CONVERSION', 'Bit', 'Temperature', 'Readout', 'Defect', 'Version', 'Serial')
out = {}
for name, (d, parts, allparts) in SETS.items():
    o = {'dir': d, 'parts': {}}
    for k, fn in enumerate(parts):
        if not os.path.exists(d + fn): o['parts'][fn] = 'MISSING'; continue
        with tifffile.TiffFile(d + fn, is_ome=False) as t:
            p = t.pages[0]
            e = {'bytes': os.path.getsize(d + fn), 'shape': list(p.shape), 'dtype': str(p.dtype), 'compression': str(p.compression)}
            def pm(i):
                tg = t.pages[i].tags.get('MicroManagerMetadata'); m = tg.value if tg is not None else {}
                return {q: m[q] for q in m if any(s in q for s in KEYS)}
            e['page0'] = pm(0)
            if k == 0:
                try:
                    mm = t.micromanager_metadata or {}
                    e['summary'] = {q: v for q, v in (mm.get('Summary') or {}).items() if q != 'StagePositions'}
                    im = mm.get('IndexMap'); e['indexmap_len'] = None if im is None else int(len(im['Frame'] if isinstance(im, dict) else im))
                    e['mm_keys'] = list(mm.keys())
                except Exception as ex: e['summary_err'] = repr(ex)
                e['desc'] = (p.description or '')[:600]
                e['page1'] = pm(1)
            if allparts or name == 'nr80':
                if allparts:
                    n = len(t.pages); e['pages'] = n; e['pagelast'] = pm(n - 1)
            print(name, fn, e.get('pages'), e['shape'], e['dtype'], e['page0'].get('FrameIndex'), flush=True)
            o['parts'][fn] = e
    out[name] = o
json.dump(out, open('/home/jc/simcode2/out/nanoruler/r2040/meta.json', 'w'), indent=1, default=str)
