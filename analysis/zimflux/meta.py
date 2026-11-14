"""ZIMFLUX server data: Micro-Manager metadata per folder (tifffile only). Pages are counted (IFD walk) only for the 6 ms
nanoruler movie and the calibration stacks (user scope 2026-11-13); other folders: Summary Frames only."""
import os, glob, json, tifffile, numpy as np
R = '/mnt/data/SIMFLUX Z/'
OUT = '/home/jc/simcode2/out/zimflux/inventory/meta.json'
KEYS = ('Exposure-ms', 'ROI', 'Camera-1-ChipName', 'Camera-1-SerialNumber', 'Camera-1-Port', 'Camera-1-ReadoutRate',
        'Camera-1-Gain', 'Camera-1-Offset', 'Camera-1-TriggerMode', 'Camera-1-PP  1   ENABLED', 'Camera-1-ActualInterval-ms')
res = {}
for d in sorted(os.listdir(R)):
    p = R + d
    if not os.path.isdir(p):
        continue
    tifs = sorted(glob.glob(p + '/*.tif')) + sorted(glob.glob(p + '/*.ome'))
    e = {'files': [os.path.basename(f) for f in tifs], 'other': sorted(set(os.listdir(p)) - set(os.path.basename(f) for f in tifs))}
    pages = 0
    count = d in ('24_12_DNA_nanorulers_6ms_1',) or 'gain' in d or 'offset' in d or 'astig' in d or 'z_scan' in d
    for f in (tifs if count else tifs[:1]):
        try:
            with tifffile.TiffFile(f, is_ome=False) as t:
                n = len(t.pages) if count else 0
                pages += n
                e.setdefault('pages_per_file', []).append(n)
                if 'shape' not in e:
                    e['shape'] = list(t.pages[0].shape); e['dtype'] = str(t.pages[0].dtype)
                    mm = t.micromanager_metadata or {}
                    s = mm.get('Summary', {})
                    e['summary'] = {k: s.get(k) for k in ('Frames', 'Slices', 'Channels', 'Positions', 'StartTime', 'z-step_um', 'Interval_ms', 'IntendedDimensions', 'UserData', 'MicroManagerVersion', 'UserName')}
                    tg = t.pages[0].tags.get('MicroManagerMetadata')
                    if tg is not None and isinstance(tg.value, dict):
                        m = tg.value
                        e['frame0'] = {k: m.get(k) for k in KEYS}
                        e['frame0']['zpos'] = {k: m[k] for k in m if 'Z' in k and ('Position' in k or 'Stage' in k or k.endswith('-Z'))}
                    if count and n > 1:
                        tg = t.pages[min(n - 1, 50)].tags.get('MicroManagerMetadata')
                        if tg is not None and isinstance(tg.value, dict):
                            e['frame50_zpos'] = {k: tg.value[k] for k in tg.value if 'Z' in k and ('Position' in k or 'Stage' in k)}
                            e['frame50_elapsed'] = tg.value.get('ElapsedTime-ms')
        except Exception as ex:
            e.setdefault('errors', []).append(f'{os.path.basename(f)}: {ex}'[:200])
    e['pages_total'] = pages
    res[d] = e
    print(d, pages, e.get('shape'), (e.get('frame0') or {}).get('Exposure-ms'), (e.get('summary') or {}).get('StartTime'), (e.get('summary') or {}).get('Frames'), (e.get('summary') or {}).get('z-step_um'), (e.get('frame0') or {}).get('ROI'), e.get('errors'), flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=str)
