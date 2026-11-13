"""Micro-Manager metadata and frame counts of the sim4_1 nanoruler movie (tifffile; no smlmtorch reader)."""
import sys, json, tifffile, numpy as np
D = '/mnt/data/SIMFLUX/data/fig2/80nm-panel-a-e/'
out = {}
for fn in ['sim4_1_MMStack_Pos0.ome.tif', 'sim4_1_MMStack_Pos0_1.ome.tif', 'sim4_1_MMStack_Pos0_2.ome.tif']:
    with tifffile.TiffFile(D + fn, is_ome=False) as t:
        n = len(t.pages)
        p = t.pages[0]
        print(fn, 'pages', n, p.shape, p.dtype)
        out[fn] = n
        if fn.endswith('Pos0.ome.tif'):
            mm = t.micromanager_metadata
            s = mm.get('Summary', {}) if mm else {}
            print(json.dumps({k: v for k, v in s.items() if k not in ('StagePositions',)}, indent=0, default=str)[:4000])
            for i in (0, 1, 6, n - 1):
                tg = t.pages[i].tags.get('MicroManagerMetadata')
                if tg is not None:
                    m = tg.value
                    print(i, {k: m[k] for k in m if any(q in k for q in ('Elapsed', 'Exposure', 'Binning', 'Camera', 'Gain', 'Offset', 'PixelSize', 'ROI', 'Sensitivity', 'ReadoutRate', 'Readout', 'Frame', 'Time', 'EM', 'Temperature', 'BitDepth', 'Core-'))})
            print('desc', (p.description or '')[:1500])
json.dump(out, open('out/nanoruler/frames.json', 'w'))
