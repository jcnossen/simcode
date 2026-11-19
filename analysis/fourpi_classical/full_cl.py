"""Full classical 4Pi-SIMFLUX reconstruction from the raw dcimg files: one sequential reader, detection like the
published analysis, patches, fits, tables. Resumable per file (files with a table are skipped).

Detection (published code: W4PiSMS_sumim_seg.m, called from W4PiSMS_RM_analysisv8stack.m; det_thresh GUI default 4):
  image   = registered sum of the 4 channels and the 6 frames of the window (ch1-3 resampled into ch0 coordinates with the
            in-situ registration, bilinear; the paper: affine FMT alignment then the 4-channel, 6-frame sum)
  filter  = a-trous wavelet band-pass: V1 = I * k5 (k5 = [1,4,6,4,1]/16 separable), V2 = V1 * k9 (k9 = k5 with holes,
            9 taps), W = V1 - V2; normalised convolution at the borders (as varunif with a unit variance map)
  thresh  = W >= 4 x MAD / 0.67, MAD = median |W - median W| per window, over the illuminated columns 55-305 (the paper:
            the whole image)
  peaks   = W equal to its 9 x 9 maximum and > 0 (fit_flag 4Pi: sz2 = 9)
  border  = the 15 x 15 patch must fit in all four channels and x in [55, 305) (paper: 13 x 13 sub-region inside the image)
Per window and candidate: 24 sub-images of 15 x 15 px (the network's patch cut: per channel centred on round(mapped
position), doff = sub-pixel rest), then the classical fits of real_cl.process (summed fit, pattern, SIMFLUX two-step,
4Pi moment phase, phase field, filters; the 24-image pixel MLE on the kept rows: NPC all, MT only in the dense boxes of
<work>/patches/regions_MT.json, for time).
Raw patches (uint16 ADU, README format of <work>/patches) are saved for NPC (all) and for the MT boxes 0 and 1.
Usage: CUDA_VISIBLE_DEVICES=<g> python analysis/fourpi_classical/full_cl.py --work <4Pi work folder> --data <raw 4Pi movie folder> --cells NPC MT
(work folder: <work>/classical/real tables, <work>/classical/patches, <work>/classical/tmp)
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p_ in (REPO, f'{REPO}/analysis/fourpi', os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path: sys.path.insert(0, p_)
from simcode.psf import fourpi_state
from simcode.setups.fourpi import FourPiSetup, acquisition, file_index
from simcode.setups.fourpi import DEFAULTS
from simcode.pipeline.configs.fourpi import GAIN_E_PER_ADU as GAIN
S4 = FourPiSetup()
P = DEFAULTS['patch']   # the same for MT and NPC
XLIM = tuple(DEFAULTS['xlim'])
# work = the 4Pi work folder, data = the folder of the raw 4Pi movies
movie_file = lambda cell, fi, work, data: S4.movie_file(acquisition(cell, work, data), fi)
movie_files = lambda cell, work, data: S4.movie_files(acquisition(cell, work, data))
from simcode.psf import fourpi_spline
from simcode.io.movie import Prefetch
from simcode.io.work import Units, atomic_save, atomic_savez, atomic_text, disk_guard
from cl4pi import Classic4Pi
import mov4pi_cl as M
import real_cl as RC

H_ = P // 2
DET_THRESH = 4.0


def _k(dev, holes):
    k = torch.tensor([1, 4, 6, 4, 1], device=dev, dtype=torch.float32) / 16
    if holes:
        k2 = torch.zeros(9, device=dev); k2[::2] = k; k = k2
    return k


def nconv(img, k):
    """separable normalised convolution with zero padding: conv(img) / conv(1) (varunif with unit variance)"""
    r = len(k) // 2
    one = torch.ones_like(img[:1])
    def c(x):
        x = F.conv2d(F.pad(x[:, None], (r, r, 0, 0)), k[None, None, None, :])
        return F.conv2d(F.pad(x, (0, 0, r, r)), k[None, None, :, None])[:, 0]
    return c(img) / c(one)


def detect_wavelet(fr, E, thresh=DET_THRESH):
    nW = E.shape[0] // 6
    S6 = E[:nW * 6].reshape(nW, 6, 4, 200, 360).sum(1)
    ssum = S6[:, 0].clone()
    for c in (1, 2, 3):
        ssum += F.grid_sample(S6[:, c:c + 1], fr.grid[c].expand(nW, -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
    V1 = nconv(ssum, _k(E.device, False)); V2 = nconv(V1, _k(E.device, True)); Wt = V1 - V2
    reg = Wt[:, :, XLIM[0]:XLIM[1]].reshape(nW, -1)
    med = reg.median(1).values; mad = (reg - med[:, None]).abs().median(1).values
    th = thresh * mad / 0.67
    A = torch.where(Wt >= th[:, None, None], Wt, torch.zeros_like(Wt))
    mx = F.max_pool2d(A[:, None], 9, 1, 4)[:, 0]
    pk = (A == mx) & (A > 0)
    pk[:, :, :XLIM[0]] = False; pk[:, :, XLIM[1]:] = False
    pk[:, :H_ + 1] = False; pk[:, -(H_ + 1):] = False
    w, cy, cx = torch.nonzero(pk, as_tuple=True)
    return w, cy, cx, Wt[w, cy, cx], th[w]


def cut(fr, E, raw_t, w, cy, cx):
    """e patches [n,6,4,P,P], raw ADU patches (uint16), doff, corner [n,4,2], ok"""
    img, doff, ok = fr.patches(E, w, cy, cx)
    cxs, cys = [cx.float()], [cy.float()]
    for c in (1, 2, 3):
        mx, my = fr.maps[c]; cxs.append(mx[cy, cx]); cys.append(my[cy, cx])
    ctr = torch.round(torch.stack([torch.stack(cxs, 1), torch.stack(cys, 1)], -1))
    ix = (ctr[..., 0].long() - H_).clamp(0, 360 - P); iy = (ctr[..., 1].long() - H_).clamp(0, 200 - P)
    o = torch.arange(P, device=E.device); t = 6 * w[:, None] + torch.arange(6, device=E.device)[None]; chn = torch.arange(4, device=E.device)
    Y = iy[:, :, None] + o[None, None]; X = ix[:, :, None] + o[None, None]
    adu = raw_t[t[:, :, None, None, None], chn[None, None, :, None, None], Y[:, None, :, :, None], X[:, None, :, None, :]]
    return img, adu, doff, torch.stack([ix, iy], -1), ok


class MemSource:
    """in-memory uint16 ADU patches -> e patches per chunk (stage B / pixel MLE)"""
    def __init__(self, adu, corner, doff, w, cx, cy, off, dev):
        self.adu, self.corner, self.doff, self.w, self.cx, self.cy, self.off, self.dev = adu, corner, doff, w, cx, cy, off, dev
        self.n = len(w)

    def info(self):
        return self.w, self.cx, self.cy, self.doff

    def chunks(self, size, rows=None):
        idx = np.arange(self.n) if rows is None else rows
        o = torch.arange(P, device=self.dev); c = torch.arange(4, device=self.dev)
        for i in range(0, len(idx), size):
            j = idx[i:i + size]
            a = torch.tensor(self.adu[j].astype(np.float32), device=self.dev)
            cor = torch.tensor(self.corner[j].astype(np.int64), device=self.dev)
            Y = cor[:, :, 1, None] + o[None, None]; X = cor[:, :, 0, None] + o[None, None]
            off = self.off[c[None, :, None, None], Y[:, :, :, None], X[:, :, None, :]]
            yield j, (a - off[:, None]) * GAIN, torch.tensor(self.doff[j], device=self.dev)


def read_units(cell, work):
    """reader-stage units: the temporary patch set of a file is complete when <tmp>/<cell>_<file>_meta.npz exists (the
    patch array is written before it; both atomically); <tmp> = <work>/classical/tmp"""
    return Units(f'{work}/classical/tmp', 'read', pattern='{cell}_{key:03d}_meta.npz', cell=cell, title=False)


def tmp_size_gb(work):
    import glob
    return sum(os.path.getsize(f) for f in glob.glob(f'{work}/classical/tmp/*.npy')) / 1e9


def read_file(cell, fi, fr, a, dev):
    """reader stage: read, detect, cut -> temporary uint16 patches + meta on the SSD (marker .ready when complete)"""
    TMP = f'{a.work}/classical/tmp'
    d = S4.reader(acquisition(cell, a.work, a.data), movie_file(cell, fi, a.work, a.data)); nfr = (a.nframes or d.n) // 6 * 6
    plan = [(a0, min(a0 + a.chunk_frames, nfr)) for a0 in range(0, nfr, a.chunk_frames)]
    pf = Prefetch(d, plan)
    acc = {k: [] for k in ('w', 'cx', 'cy', 'doff', 'corner', 'adu', 'score', 'thr')}
    t0 = time.time()
    for a0, a1 in plan:
        raw = pf.read(a0, a1)                                   # channel crops [n, 4, 200, 360], ADU
        with torch.no_grad():
            raw_t = torch.tensor(raw.astype(np.int32), device=dev)
            E = (raw_t.float() - fr.off) * GAIN
            w, cy, cx, sc, thr = detect_wavelet(fr, E)
            if len(w) == 0: continue
            img, adu, doff, corner, ok = cut(fr, E, raw_t, w, cy, cx)
            s = torch.nonzero(ok)[:, 0]
            for k, v in (('w', w[s] + a0 // 6), ('cx', cx[s]), ('cy', cy[s]), ('doff', doff[s]), ('corner', corner[s]), ('score', sc[s]), ('thr', thr[s])):
                acc[k].append(v.cpu().numpy())
            acc['adu'].append(adu[s].cpu().numpy().astype(np.uint16))
            del E, raw_t, img, adu
    pf.close()
    R = {k: np.concatenate(v) for k, v in acc.items()}
    base = f'{TMP}/{cell}_{fi:03d}'
    atomic_save(base + '.npy', R.pop('adu'))
    atomic_savez(base + '_meta.npz', **R, nframes=nfr)         # the unit's output: written last = the patch set is ready
    print(f'{cell} {fi}: read + detect {time.time() - t0:.0f}s, {len(R["w"])} candidates ({len(R["w"]) / max(nfr // 6, 1):.1f} per window)', flush=True)


def fit_file(cell, fi, c4, fr, boxes, a, dev):
    """fitter stage on the temporary patches; saves the table and the kept-ish raw patches, deletes the temp files"""
    TMP = f'{a.work}/classical/tmp'; PSAVE = f'{a.work}/classical/patches'
    base = f'{TMP}/{cell}_{fi:03d}'
    R = dict(np.load(base + '_meta.npz')); adu = np.load(base + '.npy', mmap_mode='r')
    n = len(R['w'])
    src = MemSource(adu, R['corner'], R['doff'], R['w'].astype(np.int64), R['cx'].astype(np.float64), R['cy'].astype(np.float64), fr.off, dev)
    t0 = time.time(); A = {}
    for j, img, df in src.chunks(a.chunk):
        r = M.stage_a(c4, img, df, refine=(10, 25))
        for k, v in r.items(): A.setdefault(k, []).append(v)
    A = {k: np.concatenate(v) for k, v in A.items()}
    tA = time.time() - t0
    inbox = np.full(n, -1)
    for b in boxes:
        m = (R['cx'] >= b['x0']) & (R['cx'] < b['x1']) & (R['cy'] >= b['y0']) & (R['cy'] < b['y1'])
        inbox[m] = b['id']
    pix_mask = np.ones(n, bool) if cell == 'NPC' else inbox >= 0
    a.order = a.order_npc if cell == 'NPC' else a.order_mt
    RC.process(c4, src, cell, fi, a, dev, A=A, pix_mask=pix_mask, extra=dict(score=R['score'], det_thr=R['thr'], box=inbox), tA=tA)
    # raw patches kept for reruns: rows with N >= 1000 and chi2/px <= 8 (NPC: whole field; MT: boxes 0, 1)
    save = (A['Nsum'] >= 1000) & (A['chi2px'] <= 8) & ((inbox >= 0) if cell == 'MT' else True)
    os.makedirs(f'{PSAVE}/{cell}', exist_ok=True)
    rows = np.nonzero(save)[0]
    atomic_save(f'{PSAVE}/{cell}/patches_{fi:03d}.npy', np.asarray(adu[rows]))
    atomic_savez(f'{PSAVE}/{cell}/meta_{fi:03d}.npz', row=rows, w=R['w'][rows], frame0=6 * R['w'][rows], cx=R['cx'][rows], cy=R['cy'][rows],
             corner=R['corner'][rows].astype(np.int16), doff=R['doff'][rows].astype(np.float32), box=inbox[rows], file_index=fi,
             score=R['score'][rows], detector='wavelet a-trous V1-V2, 4 x MAD/0.67, 9x9 max (published W4PiSMS_sumim_seg)')
    del adu, src
    for ext in ('_meta.npz', '.npy'):
        try: os.remove(base + ext)
        except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='read', choices=['read', 'fit'])
    ap.add_argument('--cells', nargs='+', default=['NPC', 'MT']); ap.add_argument('--files', type=int, nargs='*')
    ap.add_argument('--nframes', type=int, default=0); ap.add_argument('--chunk_frames', type=int, default=600)
    ap.add_argument('--chunk', type=int, default=2048); ap.add_argument('--psf', default='NPC_f_dxy15')
    ap.add_argument('--field', default='poly+tile'); ap.add_argument('--order_npc', type=int, default=3); ap.add_argument('--order_mt', type=int, default=3)
    ap.add_argument('--tile', type=float, default=12.0); ap.add_argument('--tblk', type=int, default=50)
    ap.add_argument('--pix', type=int, default=1); ap.add_argument('--tag', default='_wv')
    ap.add_argument('--tmp_max_gb', type=float, default=60.0)
    ap.add_argument('--min_free_gb', type=float, default=10.0, help='the reader does not start a file with less free space on the temporary directory')
    ap.add_argument('--poll_s', type=float, default=30.0)
    ap.add_argument('--work', required=True, help='4Pi work folder')
    ap.add_argument('--data', required=True, help='folder of the raw 4Pi movies')
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('4pi-classic')
    P4 = a.work; OUT = f'{P4}/classical/real'; TMP = f'{P4}/classical/tmp'
    dev = 'cuda'; os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    order = []
    for cell in a.cells:
        files = sorted(file_index(f) for f in movie_files(cell, P4, a.data))
        order += [(cell, f) for f in files if not a.files or f in a.files]
    UF = {c: RC.units(c, a.tag, P4, claim=True) for c in a.cells}       # fit-stage units = the per-file tables (claimed by a fitter)
    UR = {c: read_units(c, P4) for c in a.cells}                        # reader-stage units = the temporary patch sets
    done = lambda c, f: UF[c].done(f)
    if a.stage == 'read':
        frs = {}
        for cell, fi in order:
            if done(cell, fi) or UR[cell].done(fi) or UF[cell].claimed(fi) is not None:
                continue
            disk_guard(TMP, a.min_free_gb)
            while tmp_size_gb(P4) > a.tmp_max_gb:
                time.sleep(60)
            if cell not in frs: frs[cell] = RC.front(cell, dev, P4, a.data)
            try:
                read_file(cell, fi, frs[cell], a, dev)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f'{cell} {fi}: READ FAILED {e!r}', flush=True)
            torch.cuda.empty_cache()
        atomic_text(f'{TMP}/READER_DONE', time.strftime('%H:%M'))
        print('reader done', flush=True); return
    # fitter: claim ready files in movie order (claim marker of the table; a claim of a dead fitter is taken over)
    st = fourpi_state(a.psf, P4)
    lut = fourpi_spline(st, P, device=dev)
    reg = json.load(open(f'{P4}/patches/regions_MT.json'))
    c4s, frs = {}, {}
    while True:
        claimed = None
        for cell, fi in order:
            if UR[cell].done(fi) and not done(cell, fi) and not UF[cell].failed(fi) and UF[cell].claim(fi):
                if done(cell, fi) or not UR[cell].done(fi):      # finished by another fitter in the meantime
                    UF[cell].release(fi); continue
                claimed = (cell, fi); break
        if claimed is None:
            if os.path.exists(f'{TMP}/READER_DONE') and all(done(c, f) or UF[c].failed(f) or UF[c].claimed(f) is not None for c, f in order):
                print('fitter: nothing left', flush=True); return
            time.sleep(a.poll_s); continue
        cell, fi = claimed
        if cell not in c4s:
            c4s[cell] = Classic4Pi(lut, delta=(0.0, RC.delta(cell, P4, a.data)), dev=dev); frs[cell] = RC.front(cell, dev, P4, a.data)
        boxes = [dict(b, id=i) for i, b in enumerate(reg['boxes']) if b.get('save')] if cell == 'MT' else []
        try:
            fit_file(cell, fi, c4s[cell], frs[cell], boxes, a, dev)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f'{cell} {fi}: FIT FAILED {e!r}', flush=True)
            UF[cell].fail(fi, e)
        finally:
            UF[cell].release(fi)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
