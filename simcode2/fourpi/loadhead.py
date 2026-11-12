"""Loader of the 4Pi head checkpoints (out/4pi/models/head4pi_<tag>.pt): the learned-window head Head4PiW (arch 'w',
v10 ...) and the plain Head4Pi (no arch; v3L = the frozen prior inside Head4PiW, also used alone as the candidate front of
the simulated movies). Returns (net in eval mode on cuda, checkpoint dict)."""
import os
import torch
from simcode2.fourpi.lut4pi import LUT4Pi

REPO = '/home/jc/simcode2'


def load_head(tag_or_path):
    f = tag_or_path if os.path.sep in tag_or_path else f'{REPO}/out/4pi/models/head4pi_{tag_or_path}.pt'
    ck = torch.load(f, weights_only=False)
    a = ck['args']
    if a.get('arch') == 'w':
        from simcode2.fourpi.head4pi_w import Head4PiW
        net = Head4PiW(ch=a['ch'], nblk=a['nblk'], wins=tuple(tuple(w) for w in a['wins']), bound=a.get('bound', 1.0)).cuda().eval()
        net.load_state_dict(ck['state']); net.lut = LUT4Pi(ck['psf'], P=15)
        return net, ck
    if a.get('arch'):
        raise ValueError(f"{f}: head arch {a.get('arch')!r} is not supported (only 'w' = learned-window head, or the plain Head4Pi)")
    from simcode2.fourpi.head4pi import Head4Pi
    net = Head4Pi(ch=a['ch'], nblk=a['nblk']).cuda().eval()
    net.load_state_dict(ck['state'])
    if a.get('template') == 'lut':
        net.tlut = LUT4Pi(ck['psf'], P=15, sub=1)
    return net, ck
