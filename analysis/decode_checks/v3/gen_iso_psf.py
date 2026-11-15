"""The isolated-emitter test movie of analysis/decode_checks/gen_iso.py (main tree) with another PSF v2 calibration (PSF and
heights by its v2 rule; camera v2 crop, pattern, bg, layout unchanged), for the progress evaluation of DECODE model 2 (m25).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/decode_checks/v3/gen_iso_psf.py <calib.json> --out <dir> [gen_iso args]"""
import os, sys, functools, importlib.util
MAIN = os.path.expanduser('~/simcode2')
sys.path.insert(0, os.path.join(MAIN, 'analysis', 'papersim'))
import common as C
calib = os.path.abspath(sys.argv.pop(1))
C.PSF_CALIB_V2 = calib
C.heights_v2 = functools.partial(C.heights_v2, calib=calib)
s = importlib.util.spec_from_file_location('gen_iso', os.path.join(MAIN, 'analysis', 'decode_checks', 'gen_iso.py'))
G = importlib.util.module_from_spec(s); s.loader.exec_module(G)
G.main()
