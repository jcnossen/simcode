"""Export an in-situ 4Pi PSF for SIMCODE: a PSF json {'psf': state} that simcode/fourpi/lut4pi.py LUT4Pi and the
simulator read (fourpi.psf / fourpi.lut_psf / the training recipe).

  python analysis/fourpi_plain/p4export.py --state <Model4Pi state.json> --out <psf json> [--knots <refined knots .pt>]
      vectorial: the FourPiPSF state + 'arms' (NA, n_med, apod, pupil maps per arm), 'reg' (per-channel offsets),
      'beta'; with --knots also 'spline' = the knots file (copied next to the json) -> the refined table
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
from insitu4pi import Model4Pi, simcode_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True); p.add_argument('--out', required=True); p.add_argument('--knots', default='')
    p.add_argument('--note', default='')
    a = p.parse_args()
    m = Model4Pi.from_state(json.load(open(a.state)), roisize=17)
    st = simcode_state(m)
    if a.knots:
        import torch
        k = torch.load(a.knots, weights_only=False)
        if json.dumps(k['vector_state'], sort_keys=True) != json.dumps(st, sort_keys=True):
            raise SystemExit(f'{a.knots}: refined from another vectorial state than {a.state}')
        dst = os.path.splitext(a.out)[0] + '_knots.pt'
        shutil.copy(a.knots, dst)
        st['spline'] = os.path.abspath(dst)
    json.dump(dict(psf=st, source=dict(state=os.path.abspath(a.state), knots=a.knots, note=a.note)), open(a.out, 'w'))
    print('->', a.out)


if __name__ == '__main__':
    main()
