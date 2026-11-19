"""add the fit's sampling (npupil 64 / oversample 2: simcode.pipeline.psf_fit FIT_NPUPIL / FIT_OVERSAMPLE) to the v3 calib jsons
(right after focus_um, as psf_stage.calib_dict writes it since the fix) and rewrite their psf.zip; the old files are copied to
calib_pre_sampling/<dir tag>/ first. usage: python add_sampling.py <json> ..."""
import json, os, shutil, sys
from simcode import psf as spsf
from simcode.pipeline import psf_fit
for fn in sys.argv[1:]:
    tag = fn.replace('/', '_').strip('_')
    bk = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib_pre_sampling', tag)
    os.makedirs(bk, exist_ok=True)
    zf = fn[:-5] + '.psf.zip'
    for f in (fn, zf):
        if not os.path.exists(os.path.join(bk, os.path.basename(f))): shutil.copy2(f, bk)
    c = json.load(open(fn))
    if c.get('oversample') == psf_fit.FIT_OVERSAMPLE and c.get('npupil') == psf_fit.FIT_NPUPIL:
        print('has sampling', fn); continue
    d = {}
    for k, v in c.items():
        if k in ('npupil', 'oversample'): continue
        d[k] = v
        if k == 'focus_um': d['npupil'] = psf_fit.FIT_NPUPIL; d['oversample'] = psf_fit.FIT_OVERSAMPLE
    json.dump(d, open(fn, 'w'), indent=1)
    spsf.write_calib_zip(fn, device='cpu')
    z = spsf.read_calib(zf)
    print(fn, 'zip sampling', z.get('npupil'), z.get('oversample'), flush=True)
