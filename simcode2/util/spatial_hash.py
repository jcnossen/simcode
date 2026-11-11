"""
Sorted-grid spatial hash for same-key 2D neighbour queries, in torch (CUDA or CPU, same code) (2026-11-11, speed branch).
Replaces the scipy cKDTree queries of the inference path, which used a 3rd coordinate key * 1e4 only to restrict matches to
points with the same integer key (window start / FOV).

Hash construction adapted from the DME drift-estimation code (github.com/jcnossen/drift-estimation, master 374300f of
2026-11-10: dme/dme_gpu.py build_pairs, keyed variant dme/dme_grid.py _same_frame_pairs): cell = floor(u / h) shifted to >= 1 with a
margin (dims = max + 3) so the -1..1 offsets never wrap, mixed-radix int64 cell key with the integer key folded in
(full = key_index * ncell + cell key), one torch.sort, per neighbour offset a left / right torch.searchsorted for the cell
range, repeat_interleave expansion of the candidates, chunked over the queries. DME enumerates pairs (half the offsets);
here every query visits all 9 offsets and the candidates are reduced per query (scatter_reduce amin / sum).

Queries (all exact in float64: d^2 = dx^2 + dy^2 as cKDTree computes it for the key*1e4 embedding, where the key term is 0):
  nearest(...)          nearest point with the query's key and d < r (cKDTree query(distance_upper_bound=r) semantics:
                        strict <); returns (d, j) with d = inf, j = n_points when there is none; optional self exclusion
                        (nearest OTHER point). Ties at equal distance: the lowest point index (cKDTree's choice among
                        equidistant points depends on its tree; `stats['ties']` counts such queries).
  count(...)            number of points with the query's key and d <= r (query_ball_point(return_length=True) semantics).
Memory: the points are processed in key blocks (`block_points` points per block) and the candidates in pieces of at most
`cand_cap` (query, point) pairs, so the peak GPU memory is bounded (~1 GB at the defaults) for any number of points.
"""
import math
import numpy as np, torch

BLOCK_POINTS = 8_000_000      # points per key block on the device
CAND_CAP = 1 << 24             # (query, candidate) pairs per piece
Q_CHUNK = 1 << 20              # queries per range lookup ([Q_CHUNK, 9] lo / cnt)


class _Hash:
    """hash of one block of points (device tensors): x, y float64, k int64 key"""

    def __init__(self, x, y, k, h):
        dev = x.device
        self.h = float(h)
        u = torch.stack([x, y], 1)
        cell = torch.floor(u / h).long()
        self.cmin = cell.min(0).values
        cell = cell - self.cmin
        self.dims = cell.max(0).values + 3                     # +3 margin: offsets -1..1 never wrap (DME convention)
        cell = cell + 1
        self.mult = torch.stack([torch.ones((), dtype=torch.long, device=dev), self.dims[0]])
        self.ncell = int(self.dims[0]) * int(self.dims[1])
        self.kmin = int(k.min()); kr = int(k.max()) - self.kmin + 1
        assert kr * self.ncell < 2 ** 62, 'spatial hash: key x cell range overflows int64'
        key = (k - self.kmin) * self.ncell + (cell * self.mult).sum(1)
        self.skey, self.order = torch.sort(key)
        self.x, self.y, self.k = x, y, k
        offs = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        self.okey = torch.tensor([dx + dy * int(self.dims[0]) for dx, dy in offs], device=dev)

    def ranges(self, qx, qy, qk):
        """[m, 9] lo / cnt of the neighbour-cell ranges in the sorted order (cnt 0 for queries outside the block)"""
        qc = torch.floor(torch.stack([qx, qy], 1) / self.h).long() - self.cmin + 1
        ok = ((qc >= 0) & (qc <= self.dims - 1)).all(1) & (qk >= self.kmin) & (qk - self.kmin < (self.skey[-1] // self.ncell) + 1)
        qkey = (qk - self.kmin) * self.ncell + (qc.clamp(min=0) * self.mult).sum(1)
        nk = qkey[:, None] + self.okey[None]
        lo = torch.searchsorted(self.skey, nk, side='left')
        hi = torch.searchsorted(self.skey, nk, side='right')
        cnt = torch.where(ok[:, None], hi - lo, torch.zeros_like(lo))
        return lo, cnt

    def expand(self, lo, cnt):
        """candidates of a piece: (query row within the piece, point index within the block)"""
        m = lo.shape[0]; c = cnt.reshape(-1); tot = int(c.sum())
        dev = lo.device
        if tot == 0:
            return torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
        qi = torch.repeat_interleave(torch.arange(m, device=dev).repeat_interleave(9), c)
        base = torch.repeat_interleave(lo.reshape(-1) - (c.cumsum(0) - c), c)
        pj = self.order[base + torch.arange(tot, device=dev)]
        return qi, pj


def _pieces(cnt, cap):
    """split query rows into consecutive pieces with at most cap candidates each (a single query may exceed it)"""
    tq = cnt.sum(1)
    cs = torch.cumsum(tq, 0).cpu().numpy()
    out, s, base = [], 0, 0
    m = len(cs)
    while s < m:
        e = int(np.searchsorted(cs, base + cap, side='right'))
        e = max(e, s + 1)
        out.append((s, e)); base = int(cs[e - 1]); s = e
    return out


def _blocks(pk_sorted, qk_sorted, block_points, kpad):
    """key blocks: consecutive query ranges whose points (keys within [qk0 - kpad, qk1 + kpad]) number <= ~block_points"""
    nq = len(qk_sorted); out = []; s = 0
    while s < nq:
        k0 = qk_sorted[s]
        p0 = np.searchsorted(pk_sorted, k0 - kpad, side='left')
        # largest query key k1 such that the points up to k1 + kpad stay within the budget
        kmax_pts = pk_sorted[min(p0 + block_points, len(pk_sorted)) - 1] if len(pk_sorted) else k0
        k1 = max(kmax_pts - kpad, k0)
        e = int(np.searchsorted(qk_sorted, k1, side='right'))
        e = max(e, s + 1)
        k1 = qk_sorted[e - 1]
        p1 = np.searchsorted(pk_sorted, k1 + kpad, side='right')
        out.append((s, e, p0, p1)); s = e
    return out


def _run(px, py, pk, qx, qy, qk, r, mode, qself=None, device=None, block_points=BLOCK_POINTS, cand_cap=CAND_CAP, stats=None):
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    px = np.asarray(px, np.float64); py = np.asarray(py, np.float64); pk = np.asarray(pk, np.int64)
    qx = np.asarray(qx, np.float64); qy = np.asarray(qy, np.float64); qk = np.asarray(qk, np.int64)
    n, nq = len(px), len(qx)
    po = np.argsort(pk, kind='stable'); qo = np.argsort(qk, kind='stable')
    pks, qks = pk[po], qk[qo]
    if mode == 'nearest':
        D = np.full(nq, np.inf); J = np.full(nq, n, np.int64)
    else:
        C = np.zeros(nq, np.int64)
    r2 = float(r) * float(r)
    ties = 0
    for s, e, p0, p1 in (_blocks(pks, qks, int(block_points), 0) if nq and n else []):
        if p1 <= p0: continue
        pid = po[p0:p1]                                                   # original point indices of the block
        tt = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=dev)
        H = _Hash(tt(px[pid]), tt(py[pid]), tt(pk[pid], torch.long), max(float(r), 1e-9))
        qid = qo[s:e]
        Qx, Qy, Qk = tt(qx[qid]), tt(qy[qid]), tt(qk[qid], torch.long)
        Qs = None
        if qself is not None:                                             # self index -> block-local index (-1: not in block)
            loc = np.full(n, -1, np.int64); loc[pid] = np.arange(len(pid))
            Qs = tt(loc[np.asarray(qself)[qid]], torch.long)
        gpid = torch.as_tensor(pid, device=dev)
        for c0 in range(0, len(qid), Q_CHUNK):
            c1 = min(c0 + Q_CHUNK, len(qid))
            lo, cnt = H.ranges(Qx[c0:c1], Qy[c0:c1], Qk[c0:c1])
            for a, b in _pieces(cnt, cand_cap):
                qi, pj = H.expand(lo[a:b], cnt[a:b])
                m = b - a; A, B = c0 + a, c0 + b
                if not len(qi): continue
                ddx = Qx[A:B][qi] - H.x[pj]; ddy = Qy[A:B][qi] - H.y[pj]
                d2 = ddx * ddx + ddy * ddy                                # (not ** 2: CPU pow is not exactly x * x)
                if mode == 'count':
                    c = torch.zeros(m, dtype=torch.long, device=dev).index_add_(0, qi, (d2 <= r2).long())
                    C[qid[A:B]] = c.cpu().numpy(); continue
                keep = d2 < r2
                if Qs is not None: keep &= pj != Qs[A:B][qi]
                qi, pj, d2 = qi[keep], pj[keep], d2[keep]
                big = torch.full((m,), math.inf, dtype=torch.float64, device=dev)
                dmin = big.scatter_reduce(0, qi, d2, reduce='amin', include_self=True)
                at = d2 == dmin[qi]
                nmin = torch.zeros(m, dtype=torch.long, device=dev).index_add_(0, qi, at.long())
                ties += int((nmin > 1).sum())
                jmin = torch.full((m,), n, dtype=torch.long, device=dev).scatter_reduce(0, qi[at], gpid[pj[at]], reduce='amin', include_self=True)
                found = torch.isfinite(dmin)
                D[qid[A:B]] = np.sqrt(dmin.cpu().numpy())                # numpy sqrt: correctly rounded (torch's CPU sqrt is not)
                J[qid[A:B]] = torch.where(found, jmin, torch.full_like(jmin, n)).cpu().numpy()
            del lo, cnt
        del H
    if stats is not None:
        stats['ties'] = stats.get('ties', 0) + ties
        if dev.type == 'cuda': stats['peak_gb'] = max(stats.get('peak_gb', 0.0), torch.cuda.max_memory_allocated(dev) / 2 ** 30)
    return (D, J) if mode == 'nearest' else C


def nearest(px, py, pk, qx, qy, qk, r, qself=None, **kw):
    """nearest point (same key as the query, d < r); qself = the query's own point index to exclude (or None).
    Returns d [nq] float64 (inf if none), j [nq] int64 (n_points if none)."""
    return _run(px, py, pk, qx, qy, qk, r, 'nearest', qself=qself, **kw)


def count(px, py, pk, qx, qy, qk, r, **kw):
    """number of points with the query's key and d <= r (a query point that is itself in the set counts itself)"""
    return _run(px, py, pk, qx, qy, qk, r, 'count', **kw)
