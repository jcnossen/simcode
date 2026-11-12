"""One loader for all 4Pi head checkpoints (out/4pi/models/head4pi_<tag>.pt): Head4Pi (v2/v3/v7), Head4PiP1 (v4/v6),
Head4PiS (v8, structured). Returns (net in eval mode on cuda, checkpoint dict)."""
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
    if a.get('arch') == 'k':
        from simcode2.fourpi.head4pi_k import Head4PiK
        net = Head4PiK(ch=a['ch'], nblk=a['nblk'], wins=tuple(tuple(w) for w in a['wins'])).cuda().eval()
        net.load_state_dict(ck['state']); net.lut = LUT4Pi(ck['psf'], P=15)
        return net, ck
    if a.get('arch') == 's':
        from simcode2.fourpi.head4pi_s import Head4PiS
        net = Head4PiS(ch=a['ch'], nblk=a['nblk']).cuda().eval()
        net.load_state_dict(ck['state']); net.tlut = LUT4Pi(ck['psf'], P=15, sub=1)
        return net, ck
    if a.get('arch') == 'p1':
        from simcode2.fourpi.head4pi_p1 import Head4PiP1
        net = Head4PiP1(ch=a['ch'], nblk=a['nblk'], bg_mode=a.get('bg_mode', 'frame')).cuda().eval()
    else:
        from simcode2.fourpi.head4pi import Head4Pi
        net = Head4Pi(ch=a['ch'], nblk=a['nblk']).cuda().eval()
    net.load_state_dict(ck['state'])
    if a.get('template') == 'lut':
        net.tlut = LUT4Pi(ck['psf'], P=15, sub=1)
    return net, ck
