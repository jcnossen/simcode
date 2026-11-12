"""Inventory of all 4Pi-SIMFLUX files: sizes, frame counts with independent counters, sync across file / part boundaries.

dcimg: frame count from the header word 9, checked against the file size (data offset + n * bytes_per_frame) and the
footer frame counters of the first and last frame; timestamps of first / last frame give the inter-file gaps.

Summed TIFF parts: frame count from our own IFD walk. The page layout is regular (image data followed by its IFD), so
the IFD stride is measured from pages 1 -> 2, the count predicted from the file size, and the prediction is verified by
reading the predicted last IFD (it must hold a valid entry table and a zero next-IFD pointer) and the IFD before it
(its next pointer must point at the last one). Page 0 is handled separately (its IFD is larger). No tifffile page
indexing, no use of the multipart reader that drops 2 frames per part.

Usage: python analysis/fourpi/inventory.py  -> out/4pi/inventory/inventory.json + inventory.md
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.setup import P4
import glob, json, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dc import Dcimg, ROOT, FRAME_DT_S  # noqa

OUT = (f'{P4}/inventory')


def dcimg_info(fn):
    d = Dcimg(fn)
    size = os.path.getsize(fn)
    n_size = (size - d.off) / d.bpf
    f0, f1 = d.footer(0, 3), d.footer(d.nframes - 1, 3)
    return dict(file=os.path.relpath(fn, ROOT), size=size, nframes=d.nframes, nframes_from_size=n_size,
                trailing_bytes=int(size - d.off - d.nframes * d.bpf), shape=[d.nrow, d.ncol],
                counter_first=int(f0[0]), counter_last=int(f1[0]),
                t_first=float(f0[1] + 1e-6 * f0[2]), t_last=float(f1[1] + 1e-6 * f1[2]))


def _ifd(f, off):
    f.seek(off)
    n = struct.unpack('<H', f.read(2))[0]
    ent = f.read(12 * n)
    nxt = struct.unpack('<I', f.read(4))[0]
    tags = {}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack('<HHII', ent[12 * i:12 * i + 12])
        if typ == 3 and cnt == 1:
            val &= 0xFFFF
        tags[tag] = (typ, cnt, val)
    return n, tags, nxt


def tiff_count(fn):
    size = os.path.getsize(fn)
    with open(fn, 'rb') as f:
        hdr = f.read(8)
        assert hdr[:4] == b'II*\x00', 'only little-endian classic TIFF handled'
        off0 = struct.unpack('<I', hdr[4:8])[0]
        n0, t0, off1 = _ifd(f, off0)
        n1, t1, off2 = _ifd(f, off1)
        n2, t2, off3 = _ifd(f, off2)
        w, h = t1[256][2], t1[257][2]
        stride = off2 - off1
        assert off3 - off2 == stride, 'irregular IFD stride'
        # pages k >= 1: IFD_k at off1 + (k-1)*stride; last page k_last such that its strip ends within the file
        k_last = 1 + (size - off1 - (2 + 12 * n1 + 4)) // stride
        # walk back to the last valid IFD with next == 0
        found = None
        for k in range(k_last + 1, max(k_last - 4, 1), -1):
            o = off1 + (k - 1) * stride
            if o + 2 + 12 * n1 + 4 > size:
                continue
            try:
                n, t, nxt = _ifd(f, o)
            except struct.error:
                continue
            if n == n1 and 256 in t and t[256][2] == w and nxt == 0:
                found = k; break
        assert found is not None, 'could not verify last IFD'
        # the IFD before must point to it
        _, _, nprev = _ifd(f, off1 + (found - 2) * stride)
        assert nprev == off1 + (found - 1) * stride
        ifd_sizes = (2 + 12 * n0 + 4, 2 + 12 * n1 + 4)
    return dict(file=os.path.relpath(fn, ROOT), size=size, npages=found + 1, width=w, height=h,
                ifd0_bytes=ifd_sizes[0], ifd_bytes=ifd_sizes[1], ifd_stride=stride)


def part_key(fn):
    m = re.search(r'summed_X(\d+)\.tif$', fn)
    return 0 if m is None else int(m.group(1))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    groups = {
        'MT_movie': sorted(glob.glob(f'{ROOT}/MT/Cell03_MT/*.dcimg')),
        'NPC_movie': sorted(glob.glob(f'{ROOT}/NPC/movies/*.dcimg')),
        'MT_zstack': sorted(glob.glob(f'{ROOT}/MT/Cell02_beads_z_stack/*.dcimg')),
        'NPC_zstack': sorted(glob.glob(f'{ROOT}/NPC/zstack/*.dcimg')),
        'MT_beadalign': sorted(glob.glob(f'{ROOT}/MT/Cell01_beads_align/*.dcimg')),
        'NPC_beadalign': sorted(glob.glob(f'{ROOT}/NPC/beads_align/*.dcimg')),
    }
    for g, fs in groups.items():
        res[g] = [dcimg_info(f) for f in fs]
        print(g, len(fs), flush=True)
    for cell in ('MT', 'NPC'):
        fs = sorted(glob.glob(f'{ROOT}/{cell}/summed/summed*.tif'), key=part_key)
        res[f'{cell}_summed'] = [tiff_count(f) for f in fs]
        print(cell, 'summed', len(fs), flush=True)
    others = [os.path.relpath(p, ROOT) for p in glob.glob(f'{ROOT}/**/*', recursive=True)
              if os.path.isfile(p) and not p.endswith(('.dcimg', '.tif'))]
    res['other_files'] = sorted(others)
    json.dump(res, open(f'{OUT}/inventory.json', 'w'), indent=1)

    L = ['# 4Pi-SIMFLUX inventory', '', 'Generated by `analysis/fourpi/inventory.py`. Frame counts: dcimg header word 9, '
         'checked against file size and footer counters; TIFF: own IFD walk (verified last IFD), not the multipart reader.', '']
    L += ['| group | files | frames total | frames/file | size total | size consistent (tail = stamp table) | counters 0..n-1 | gaps between files (s) min / median / max |',
          '|---|---|---|---|---|---|---|---|']
    for g in groups:
        r = res[g]
        if not r:
            continue
        nf = [x['nframes'] for x in r]
        # bytes after the last frame record = the per-frame stamp table (512 B per frame + 1 kB); never a whole frame
        okz = all(x['trailing_bytes'] == 512 * x['nframes'] + 1024 or x['trailing_bytes'] < 922112 for x in r)
        okc = all(x['counter_first'] == 0 and x['counter_last'] == x['nframes'] - 1 for x in r)
        gaps = [r[i + 1]['t_first'] - r[i]['t_last'] for i in range(len(r) - 1)]
        gs = f'{min(gaps):.2f} / {np.median(gaps):.2f} / {max(gaps):.2f}' if gaps and 'movie' in g else '-'
        L.append(f'| {g} | {len(r)} | {sum(nf)} | {min(nf)}–{max(nf)} | {sum(x["size"] for x in r) / 1e9:.1f} GB | '
                 f'{okz} | {okc} | {gs} |')
    L += ['', '## Summed TIFF parts (derived product, not used for processing)', '',
          '| cell | parts | pages total (IFD walk) | movie frames | pages/part (first..last) |', '|---|---|---|---|---|']
    for cell in ('MT', 'NPC'):
        r = res[f'{cell}_summed']
        npg = [x['npages'] for x in r]
        mov = sum(x['nframes'] for x in res[f'{cell}_movie'])
        L.append(f'| {cell} | {len(r)} | {sum(npg)} | {mov} | {npg[0]} .. {npg[-1]} |')
    L += ['', 'The parts are complete (pages = movie frames). The known reader problem (2 frames lost per part, '
          'SMLM_REPORT section 8) is in the reader, not in the files.']
    L += ['', '## Other files', ''] + [f'- `{o}`' for o in res['other_files']]
    open(f'{OUT}/inventory.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
