"""Microtubule cell field: Python/numba port of webSMLM's 'microtubules (cells)' structure.

webSMLM (webSMLM.html, MODULE: simulation) does not implement this generator in JS: it embeds the inSiliScope
"CellField" world model (Koen J. A. Martens, github.com/kjamartens/inSiliScope, BSD-3-Clause), C++ compiled to WASM,
at source commit 57d959596206d027e7b032d9ee5db08697eb067e. This module is a line-by-line port of that C++ core
(core/src/{rng,cells,packing,cytomesh,microtubules,dyes,world}.cpp at the pinned commit) with the parameter set
webSMLM passes to it (CellField DEFAULTS in webSMLM.html, identical to the core's Params defaults) and webSMLM's own
wrapper (buildWindow / mtStartPosition / buildMicrotubuleStructure).

The model (see the README next to the figures for the parameter list):
  * an infinite, seed-addressed field of cells on a 26 um chunk grid (at most one jittered candidate per chunk,
    occupancy 0.33), blobby elliptical footprints (geometric-mean diameter 25-35 um), packed apart on fixed 8 x 8-chunk
    blocks (Jacobi relaxation + rotation, prune of stuck pairs); every cell has an ellipsoidal nucleus (long axis
    8-12 um) and a cytoplasm height field (rim 0.1-0.3 um, mid 1-2 um, dome over the nucleus, cell height 3-6 um);
  * per cell 0.9 microtubules per um^2 of footprint: each starts on the nucleus surface, runs as a persistent random
    walk in x-y to an end near the cell edge (min. turn radius 0.15 um), z rides a fraction of the local cytoplasm
    ceiling, crosses over/under the nucleus, is clamped into the cytoplasm and truncated at the cell edge;
  * every microtubule is a 25 nm cylinder carrying the 13_3 protofilament lattice (13 sites per 8 nm); each site is an
    attachment point on the 12.5 nm surface -> binder tip 12 nm radially further out -> dye 2-5 nm from the tip
    (uniform direction, radius uniform in volume). Each dye is one label position.
  * Everything is a pure function of (seed, address): the same seed gives the same cells, microtubules and dyes
    wherever the window sits.

Randomness: the core's address-based hash (pcg4d) is ported bit-exactly, and so is webSMLM's mulberry32 (start view),
so this port draws the SAME random numbers as webSMLM. The remaining difference is floating point: the core calls
fdlibm (V8 Math.*) for sin/cos/atan2/exp/log, this port the platform libm, which can differ in the last ulp; packing
and walk decisions are continuous in those values, so a rare decision can flip (see the README for the measured match).

Not part of webSMLM, added here: `label_efficiency` thins the lattice sites with the core's own per-site LABEL hash
(deterministic, nested: a lower efficiency keeps a subset of a higher one; webSMLM instead keeps each returned site
with probability 70 % using its movie RNG), and the binder length / linker range are parameters (webSMLM: fixed 12 nm
and 2-5 nm).

Usage:
    from mt_cells import simulate
    r = simulate(seed=3, fov_um=(10, 10), focus_nm=250, z_range_nm=750, label_efficiency=0.7)
    r['xyz']      # [N, 3] float64 nm: x, y in the FOV frame (0..W, 0..H; +y down as in an image), z above coverslip
"""
# Ported from inSiliScope (github.com/kjamartens/inSiliScope, commit 57d959596206d027e7b032d9ee5db08697eb067e;
# cite as Martens2026inSiliScope: K. J. A. Martens, "inSiliScope: a deterministic in-silico microscope", 2026, v0.1.0).
# Its licence (BSD-3-Clause), retained as it requires:
#
# Copyright (c) 2026, the demoCam_SMLM_MM contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#     * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from collections import OrderedDict

import numpy as np
from numba import njit, types
from numba.typed import List

PI = 3.141592653589793
LN2 = 0.6931471805599453
TWO_PI = PI * 2

# ---------------------------------------------------------------------------------------------------------------------
# parameters (names and defaults = the core's Params = webSMLM's CellField DEFAULTS; webSMLM passes labelEfficiency 1)
# ---------------------------------------------------------------------------------------------------------------------
DEFAULTS = OrderedDict([
    # field
    ('chunkSize', 26.0), ('jitter', 0.8), ('density', 0.33),
    # cell
    ('cellDiamMin', 25.0), ('cellDiamMax', 35.0), ('cellElongMin', 0.5), ('cellElongMax', 1.0), ('cellBlob', 1.75),
    ('cellHeightMin', 3.0), ('cellHeightMax', 6.0),
    # nucleus
    ('nucLongMin', 8.0), ('nucLongMax', 12.0), ('nucRatioMin', 0.6), ('nucRatioMax', 1.0),
    ('nucHeightMin', 0.3), ('nucHeightMax', 0.5), ('nucOffsetFrac', 0.1), ('nucMargin', 0.6),
    # cytoplasm
    ('cytoRimHeightMin', 0.1), ('cytoRimHeightMax', 0.3), ('cytoEdgeRiseMin', 0.1), ('cytoEdgeRiseMax', 0.5),
    ('cytoMidHeightMin', 1.0), ('cytoMidHeightMax', 2.0), ('cytoMidDistanceMin', 0.1), ('cytoMidDistanceMax', 0.3),
    ('cytoMaxSlope', 1.0), ('cytoDomeSlope', 3.0), ('cytoSmoothPasses', 12.0), ('cytoRings', 60.0), ('cytoTheta', 128.0),
    # packing
    ('enablePacking', 1.0), ('allowPackRotation', 1.0), ('packFrac', 1.0), ('relaxIters', 80.0), ('relaxDamping', 0.55),
    # microtubules
    ('mtDensity', 0.9), ('mtStartFracMin', 0.0), ('mtStartFracMax', 0.3), ('mtStartOffsetXY', 0.0),
    ('mtEndFracMin', 0.01), ('mtEndFracMax', 0.4), ('mtEndJitterDeg', 145.0), ('mtWobbleTurn', 0.8),
    ('mtWobbleFactor', 1.05), ('mtStepLen', 0.05), ('mtSmoothLen', 1.5), ('mtMinTurnRadius', 0.15),
    ('mtMinSeparation', 0.05), ('mtMaxZSlope', 5.0),
])
PARAM_NAMES = list(DEFAULTS)
for _i, _n in enumerate(PARAM_NAMES):
    globals()['P_' + _n] = _i          # numba reads these module ints as compile-time constants
NP_ = len(PARAM_NAMES)

# cell record fields (one float64 row per cell)
CELL_FIELDS = ['cx', 'cy', 'x', 'y', 'present', 'semiMajor', 'semiMinor', 'rot', 'height', 'ha0', 'ha1', 'ha2',
               'hp0', 'hp1', 'hp2', 'modFloor', 'nucLong', 'nucShort', 'nucHeight', 'nucOffX', 'nucOffY', 'nucRot',
               'nucZ', 'rim', 'edge', 'mid', 'midDist', 'rOuter', 'priority', 'packRot']
for _i, _n in enumerate(CELL_FIELDS):
    globals()['C_' + _n] = _i
NC_ = len(CELL_FIELDS)

# hash purpose channels (cells.h CH)
CH_X, CH_Y, CH_OCC, CH_SIZE, CH_ELONG, CH_ROT, CH_HEIGHT = 0, 1, 2, 3, 4, 5, 6
CH_HARM_AMP, CH_HARM_PH = 10, 20
CH_NUC_RATIO, CH_NUC_HRATIO, CH_NUC_OFFR, CH_NUC_OFFANG, CH_NUC_ROT, CH_NUC_ZFRAC = 31, 32, 33, 34, 35, 36
CH_PRIORITY, CH_CYTO_RIM, CH_CYTO_EDGE, CH_CYTO_MID, CH_CYTO_MIDDIST = 40, 41, 42, 43, 44
HARM_K = (2, 3, 5)
CELL_MOD_MIN, CELL_MOD_MAX = 0.35, 1.65
RADIUS_LUT_N, OUTLINE_SAMPLE_N = 128, 16
PACK_ROT_TRIAL = 12 * PI / 180
PRUNE_ROUNDS = 6
PACK_BLOCK_CHUNKS = 8
CYTO_RING_BIAS = 2.4
# microtubules.h
MT_CH_COUNT, MT_STREAM_BASE, MT_RESAMPLE_SPACING = 500, 1000, 100000
MT_N_DIR, MT_MARCH_STEPS, MT_BISECT_ITERS = 64, 24, 24
MT_NUDGE_ROUNDS, MT_RESAMPLE_ROUNDS, MT_POST_RESAMPLE_NUDGE_ROUNDS = 5, 2, 3
MT_COLLISION_OP_BUDGET, MT_COLLISION_MAX_TOTAL_POINTS = 300000, 8000
MT_MAX_STEPS_PER_MT, MT_MAX_PER_CELL = 1500, 5000
MT_CONTAIN_MARGIN, MT_NUCLEUS_CLEAR_BLEND, MT_MAX_END_TRIM = 0.98, 0.4, 50
# dyes.h
MT_RADIUS_NM, MT_N_PROTOFILAMENTS, MT_DIMER_NM, MT_LATTICE_START = 12.5, 13, 8.0, 3.0
MT_BINDER_NM, MT_LINKER_MIN_NM, MT_LINKER_MAX_NM = 12.0, 2.0, 5.0
MT_CH_SEAM_PHASE, DYE_SALT, DYE_BLOCK_UM = 8000000, 0x9E3779B9, 1.0
SITES_PER_UM = MT_N_PROTOFILAMENTS * 1000.0 / MT_DIMER_NM      # 1625 lattice sites per um of microtubule

_U = np.uint64
_ARR2 = types.float64[:, ::1]


# ---------------------------------------------------------------------------------------------------------------------
# math helpers (jsmath.cpp; hypot as V8's Math.hypot)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _lerp(a, b, t):
    return a + (b - a) * t


@njit(cache=True)
def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


@njit(cache=True)
def _hypot2(x, y):
    a = abs(x); b = abs(y)
    if a == np.inf or b == np.inf:
        return np.inf
    if a != a or b != b:
        return np.nan
    mx = a if a > b else b
    if mx == 0:
        return 0.0
    an = a / mx; bn = b / mx
    return math.sqrt(an * an + bn * bn) * mx


@njit(cache=True)
def _hypot3(x, y, z):
    v0 = abs(x); v1 = abs(y); v2 = abs(z)
    nan = False; mx = 0.0
    for a in (v0, v1, v2):
        if a != a:
            nan = True
        elif a > mx:
            mx = a
    if mx == np.inf:
        return np.inf
    if nan:
        return np.nan
    if mx == 0:
        return 0.0
    s = 0.0; comp = 0.0
    for a in (v0, v1, v2):
        n = a / mx
        summand = n * n - comp
        prelim = s + summand
        comp = (prelim - s) - summand
        s = prelim
    return math.sqrt(s) * mx


@njit(cache=True)
def _jsround(x):
    r = math.floor(x)
    return r + 1 if x - r >= 0.5 else r


@njit(cache=True)
def _normor1(n):
    return 1.0 if (n == 0 or n != n) else n


# ---------------------------------------------------------------------------------------------------------------------
# RNG: pcg4d + hashUnit (rng.h), bit-exact (uint32 arithmetic in uint64 with masking)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _u32(x):
    return _U(np.int64(x) & np.int64(0xFFFFFFFF))


@njit(cache=True)
def _pcg4d(a, b, c, d):
    M = _U(0xFFFFFFFF); A = _U(1664525); B = _U(1013904223); S = _U(16)
    a = (a * A + B) & M; b = (b * A + B) & M; c = (c * A + B) & M; d = (d * A + B) & M
    a = (a + b * d) & M; b = (b + c * a) & M; c = (c + a * b) & M; d = (d + b * c) & M
    a ^= a >> S; b ^= b >> S; c ^= c >> S; d ^= d >> S
    a = (a + b * d) & M; b = (b + c * a) & M; c = (c + a * b) & M; d = (d + b * c) & M
    return a, b, c, d


@njit(cache=True)
def _unit(a):
    return (float(a >> _U(9)) + 0.5) * 1.1920928955078125e-7


@njit(cache=True)
def _hash_unit(seed, cx, cy, k):
    """seed: uint64 (< 2^32), cx, cy: int (int32 semantics), k: int (uint32 semantics)"""
    return _unit(_pcg4d(seed, _u32(cx), _u32(cy), _u32(k))[0])


@njit(cache=True)
def _stream_next(seed, cx, cy, base4096, ctr):
    """HashStream::Next: draw ctr[0] of the stream with base*4096 = base4096 (uint64 < 2^32)"""
    k = (base4096 + _U(ctr[0])) & _U(0xFFFFFFFF)
    ctr[0] += 1
    return _unit(_pcg4d(seed, _u32(cx), _u32(cy), k)[0])


def pcg4d_np(a, b, c, d):
    """vectorised pcg4d for tests (uint32-like numpy arrays / ints)"""
    M = np.uint64(0xFFFFFFFF)
    a, b, c, d = (np.asarray(v, dtype=np.int64).astype(np.uint64) & M for v in (a, b, c, d))
    A, B, S = np.uint64(1664525), np.uint64(1013904223), np.uint64(16)
    a = (a * A + B) & M; b = (b * A + B) & M; c = (c * A + B) & M; d = (d * A + B) & M
    a = (a + b * d) & M; b = (b + c * a) & M; c = (c + a * b) & M; d = (d + b * c) & M
    a ^= a >> S; b ^= b >> S; c ^= c >> S; d ^= d >> S
    a = (a + b * d) & M; b = (b + c * a) & M; c = (c + a * b) & M; d = (d + b * c) & M
    return a, b, c, d


def mulberry32(seed):
    """webSMLM's mulberry32 (bit-exact): returns a callable giving uniforms in [0, 1)"""
    st = [seed & 0xFFFFFFFF]
    M = 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & M

    def nxt():
        a = (st[0] + 0x6D2B79F5) & M
        st[0] = a
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296.0
    return nxt


# ---------------------------------------------------------------------------------------------------------------------
# cells (cells.cpp)
# ---------------------------------------------------------------------------------------------------------------------
def normalize_params(p):
    """NormalizeParams: max >= min, floors of a few microtubule controls"""
    q = dict(p)
    for a, b in [('cellDiamMin', 'cellDiamMax'), ('cellElongMin', 'cellElongMax'), ('cellHeightMin', 'cellHeightMax'),
                 ('nucLongMin', 'nucLongMax'), ('nucRatioMin', 'nucRatioMax'), ('nucHeightMin', 'nucHeightMax'),
                 ('cytoRimHeightMin', 'cytoRimHeightMax'), ('cytoEdgeRiseMin', 'cytoEdgeRiseMax'),
                 ('cytoMidHeightMin', 'cytoMidHeightMax'), ('cytoMidDistanceMin', 'cytoMidDistanceMax'),
                 ('mtStartFracMin', 'mtStartFracMax'), ('mtEndFracMin', 'mtEndFracMax')]:
        q[b] = max(q[a], q[b])
    q['mtWobbleFactor'] = max(1.0, q['mtWobbleFactor'])
    q['mtStepLen'] = max(0.02, q['mtStepLen'])
    q['mtSmoothLen'] = max(0.0, q['mtSmoothLen'])
    q['mtMinTurnRadius'] = max(0.0, q['mtMinTurnRadius'])
    q['mtMaxZSlope'] = max(1.0, q['mtMaxZSlope'])
    q['enablePacking'] = 1.0 if q['enablePacking'] else 0.0
    q['allowPackRotation'] = 1.0 if q['allowPackRotation'] else 0.0
    return q


def params_array(overrides=None):
    p = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in p:
            raise KeyError(f'unknown cell-field parameter {k}')
        p[k] = float(v)
    p = normalize_params(p)
    return np.array([p[n] for n in PARAM_NAMES], dtype=np.float64)


@njit(cache=True)
def _cell_radius_at(c, theta):
    phi = theta - c[C_rot]
    a = c[C_semiMajor]; b = c[C_semiMinor]
    base = (a * b) / _hypot2(b * math.cos(phi), a * math.sin(phi))
    mod = 1.0
    mod += c[C_ha0] * math.cos(2 * theta + c[C_hp0])
    mod += c[C_ha1] * math.cos(3 * theta + c[C_hp1])
    mod += c[C_ha2] * math.cos(5 * theta + c[C_hp2])
    mod = max(c[C_modFloor], min(CELL_MOD_MAX, mod))
    return base * mod


@njit(cache=True)
def _cyto_dome_reach(c, P):
    margin = max(0.1, P[P_nucMargin])
    if not (P[P_cytoDomeSlope] > 0):
        return margin
    H = max(max(c[C_height], c[C_rim]), c[C_mid])
    return max(margin, 1.5 * (H - c[C_mid]) / P[P_cytoDomeSlope])


@njit(cache=True)
def _cyto_slope_runout(c, P):
    margin = max(0.1, P[P_nucMargin])
    ceil_run = 2 * c[C_mid] * LN2 / P[P_cytoMaxSlope] if P[P_cytoMaxSlope] > 0 else 0.0
    return max(0.0, _cyto_dome_reach(c, P) + ceil_run - margin)


@njit(cache=True)
def _envelop_nucleus(c, margin_um, P):
    margin = max(0.1, margin_um)
    rz = c[C_nucHeight] / 2
    needed = c[C_nucHeight] + 2 * margin
    if c[C_height] < needed:
        c[C_height] = needed
    c[C_nucZ] = min(max(c[C_nucZ], rz + margin), c[C_height] - rz - margin)
    extra = _cyto_slope_runout(c, P)
    N = 32
    needed_floor = CELL_MOD_MIN
    sa = c[C_semiMajor]; sb = c[C_semiMinor]
    for i in range(N):
        th = (i / N) * PI * 2
        lx = math.cos(th) * (c[C_nucLong] / 2); ly = math.sin(th) * (c[C_nucShort] / 2)
        wx = c[C_nucOffX] + lx * math.cos(c[C_nucRot]) - ly * math.sin(c[C_nucRot])
        wy = c[C_nucOffY] + lx * math.sin(c[C_nucRot]) + ly * math.cos(c[C_nucRot])
        dist = _hypot2(wx, wy)
        angle = math.atan2(wy, wx)
        phi = angle - c[C_rot]
        base = (sa * sb) / _hypot2(sb * math.cos(phi), sa * math.sin(phi))
        if base <= 1e-6:
            continue
        required = (dist * 1.05 + margin + extra) / base
        nat = 1.0
        nat += c[C_ha0] * math.cos(2 * angle + c[C_hp0])
        nat += c[C_ha1] * math.cos(3 * angle + c[C_hp1])
        nat += c[C_ha2] * math.cos(5 * angle + c[C_hp2])
        capped = min(CELL_MOD_MAX, nat)
        if capped < required:
            needed_floor = max(needed_floor, required)
    c[C_modFloor] = min(needed_floor, CELL_MOD_MAX)
    if needed_floor > CELL_MOD_MAX:
        k = needed_floor / CELL_MOD_MAX
        c[C_semiMajor] *= k
        c[C_semiMinor] *= k


@njit(cache=True)
def _raw_candidate(seed, cx, cy, P):
    c = np.zeros(NC_)
    c[C_cx] = cx; c[C_cy] = cy
    c[C_modFloor] = CELL_MOD_MIN
    S = P[P_chunkSize]; jit = P[P_jitter]
    margin = (1 - jit) / 2
    c[C_x] = (cx + margin + jit * _hash_unit(seed, cx, cy, CH_X)) * S
    c[C_y] = (cy + margin + jit * _hash_unit(seed, cx, cy, CH_Y)) * S
    present = _hash_unit(seed, cx, cy, CH_OCC) < P[P_density]
    c[C_present] = 1.0 if present else 0.0
    if not present:
        return c
    sizeT = _hash_unit(seed, cx, cy, CH_SIZE)
    baseR = _lerp(P[P_cellDiamMin], P[P_cellDiamMax], sizeT) / 2
    elong = _lerp(P[P_cellElongMin], P[P_cellElongMax], _hash_unit(seed, cx, cy, CH_ELONG))
    sa = baseR / math.sqrt(elong)
    sb = sa * elong
    c[C_rot] = _hash_unit(seed, cx, cy, CH_ROT) * PI * 2
    c[C_height] = _lerp(P[P_cellHeightMin], P[P_cellHeightMax], _hash_unit(seed, cx, cy, CH_HEIGHT))
    blob = P[P_cellBlob]
    c[C_ha0] = _hash_unit(seed, cx, cy, CH_HARM_AMP + 0) * blob / 2
    c[C_ha1] = _hash_unit(seed, cx, cy, CH_HARM_AMP + 1) * blob / 3
    c[C_ha2] = _hash_unit(seed, cx, cy, CH_HARM_AMP + 2) * blob / 5
    c[C_hp0] = _hash_unit(seed, cx, cy, CH_HARM_PH + 0) * PI * 2
    c[C_hp1] = _hash_unit(seed, cx, cy, CH_HARM_PH + 1) * PI * 2
    c[C_hp2] = _hash_unit(seed, cx, cy, CH_HARM_PH + 2) * PI * 2
    # area correction on the clamped outline (48 samples)
    N = 48
    sb2 = 0.0; sr2 = 0.0
    for i in range(N):
        th = (i / N) * PI * 2
        phi = th - c[C_rot]
        base = (sa * sb) / _hypot2(sb * math.cos(phi), sa * math.sin(phi))
        mod = 1.0
        mod += c[C_ha0] * math.cos(2 * th + c[C_hp0])
        mod += c[C_ha1] * math.cos(3 * th + c[C_hp1])
        mod += c[C_ha2] * math.cos(5 * th + c[C_hp2])
        mod = max(CELL_MOD_MIN, min(CELL_MOD_MAX, mod))
        sb2 += base * base
        sr2 += base * base * mod * mod
    k = math.sqrt(sb2 / sr2)
    c[C_semiMajor] = sa * k
    c[C_semiMinor] = sb * k
    nucLong = _lerp(P[P_nucLongMin], P[P_nucLongMax], sizeT)
    nucRatio = _lerp(P[P_nucRatioMin], P[P_nucRatioMax], _hash_unit(seed, cx, cy, CH_NUC_RATIO))
    offR = _hash_unit(seed, cx, cy, CH_NUC_OFFR) * P[P_nucOffsetFrac] * baseR
    offAng = _hash_unit(seed, cx, cy, CH_NUC_OFFANG) * PI * 2
    c[C_nucLong] = nucLong
    c[C_nucShort] = nucLong * nucRatio
    c[C_nucHeight] = nucLong * _lerp(P[P_nucHeightMin], P[P_nucHeightMax], _hash_unit(seed, cx, cy, CH_NUC_HRATIO))
    c[C_nucOffX] = math.cos(offAng) * offR
    c[C_nucOffY] = math.sin(offAng) * offR
    c[C_nucRot] = _hash_unit(seed, cx, cy, CH_NUC_ROT) * PI * 2
    c[C_nucZ] = c[C_height] * _lerp(0.4, 0.6, _hash_unit(seed, cx, cy, CH_NUC_ZFRAC))
    c[C_rim] = _lerp(P[P_cytoRimHeightMin], P[P_cytoRimHeightMax], _hash_unit(seed, cx, cy, CH_CYTO_RIM))
    c[C_edge] = _lerp(P[P_cytoEdgeRiseMin], P[P_cytoEdgeRiseMax], _hash_unit(seed, cx, cy, CH_CYTO_EDGE))
    c[C_mid] = _lerp(P[P_cytoMidHeightMin], P[P_cytoMidHeightMax], _hash_unit(seed, cx, cy, CH_CYTO_MID))
    c[C_midDist] = _lerp(P[P_cytoMidDistanceMin], P[P_cytoMidDistanceMax], _hash_unit(seed, cx, cy, CH_CYTO_MIDDIST))
    _envelop_nucleus(c, P[P_nucMargin], P)
    c[C_height] = max(max(c[C_height], c[C_rim]), c[C_mid])
    c[C_rOuter] = c[C_semiMajor] * CELL_MOD_MAX
    c[C_priority] = _hash_unit(seed, cx, cy, CH_PRIORITY)
    c[C_packRot] = 0.0
    return c


# ---------------------------------------------------------------------------------------------------------------------
# packing (packing.cpp)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _radius_from_lut(lut, theta_local):
    t = np.fmod(theta_local, TWO_PI)
    if t < 0:
        t += TWO_PI
    f = (t / TWO_PI) * RADIUS_LUT_N
    i0 = int(f)
    i1 = (i0 + 1) % RADIUS_LUT_N
    frac = f - i0
    if i0 >= RADIUS_LUT_N:
        return np.nan
    return float(lut[i0]) * (1 - frac) + float(lut[i1]) * frac


@njit(cache=True)
def _ensure_luts(cells, lut, col):
    for q in range(cells.shape[0]):
        c = cells[q]
        for i in range(RADIUS_LUT_N):
            lut[q, i] = np.float32(_cell_radius_at(c, (i / RADIUS_LUT_N) * PI * 2))
        for i in range(OUTLINE_SAMPLE_N):
            th = (i / OUTLINE_SAMPLE_N) * TWO_PI
            r = _radius_from_lut(lut[q], th)
            col[q, i * 2] = np.float32(r * math.cos(th))
            col[q, i * 2 + 1] = np.float32(r * math.sin(th))


@njit(cache=True)
def _boundary_clearance(cells, lut, col, ci, crot, ni, nrot):
    mn = np.inf
    cC = math.cos(crot); cS = math.sin(crot)
    nC = math.cos(nrot); nS = math.sin(nrot)
    cxw = cells[ci, C_x]; cyw = cells[ci, C_y]; nxw = cells[ni, C_x]; nyw = cells[ni, C_y]
    for i in range(OUTLINE_SAMPLE_N):
        lx = float(col[ci, i * 2]); ly = float(col[ci, i * 2 + 1])
        wx = cxw + lx * cC - ly * cS; wy = cyw + lx * cS + ly * cC
        bdx = wx - nxw; bdy = wy - nyw
        bd = _hypot2(bdx, bdy)
        gap = bd - _radius_from_lut(lut[ni], math.atan2(bdy, bdx) - nrot)
        if gap < mn:
            mn = gap
    for i in range(OUTLINE_SAMPLE_N):
        lx = float(col[ni, i * 2]); ly = float(col[ni, i * 2 + 1])
        wx = nxw + lx * nC - ly * nS; wy = nyw + lx * nS + ly * nC
        adx = wx - cxw; ady = wy - cyw
        ad = _hypot2(adx, ady)
        gap = ad - _radius_from_lut(lut[ci], math.atan2(ady, adx) - crot)
        if gap < mn:
            mn = gap
    return mn


@njit(cache=True)
def _directional_overlap(cells, lut, col, ci, crot, ni, min_gap_frac):
    dx = cells[ni, C_x] - cells[ci, C_x]; dy = cells[ni, C_y] - cells[ci, C_y]
    d = _hypot2(dx, dy)
    if d == 0 or d != d:
        d = 1e-6
    thAB = math.atan2(dy, dx)
    ra = _radius_from_lut(lut[ci], thAB - crot)
    rb = _radius_from_lut(lut[ni], thAB + PI - cells[ni, C_packRot])
    allowed = (ra + rb) * (1 - min_gap_frac)
    clear = _boundary_clearance(cells, lut, col, ci, crot, ni, cells[ni, C_packRot])
    return -allowed - clear, dx / d, dy / d


@njit(cache=True)
def _interaction_chunks(P):
    worst_sa = (P[P_cellDiamMax] / 2) / math.sqrt(max(0.05, P[P_cellElongMin]))
    worst_ro = worst_sa * CELL_MOD_MAX
    return int(min(6.0, max(1.0, math.ceil((2 * worst_ro) / P[P_chunkSize]))))


@njit(cache=True)
def _find(grid, alive, cx0, cy0, cx1, cy1, Ht, cx, cy):
    if cx < cx0 or cx > cx1 or cy < cy0 or cy > cy1:
        return -1
    i = grid[(cx - cx0) * Ht + (cy - cy0)]
    if i >= 0 and alive[i]:
        return i
    return -1


@njit(cache=True)
def _relax(cells, grid, alive, cx0, cy0, cx1, cy1, lut, col, P, iters):
    NR = _interaction_chunks(P)
    damping = P[P_relaxDamping]; mgf = P[P_packFrac]; allow_rot = P[P_allowPackRotation] != 0
    N = cells.shape[0]; Ht = cy1 - cy0 + 1
    K = (2 * NR + 1) * (2 * NR + 1)
    nb = np.full((N, K), -1, np.int64); nbn = np.zeros(N, np.int64)
    for i in range(N):
        if not alive[i]:
            continue
        ccx = int(cells[i, C_cx]); ccy = int(cells[i, C_cy])
        for dx in range(-NR, NR + 1):
            for dy in range(-NR, NR + 1):
                if dx == 0 and dy == 0:
                    continue
                j = _find(grid, alive, cx0, cy0, cx1, cy1, Ht, ccx + dx, ccy + dy)
                if j >= 0:
                    nb[i, nbn[i]] = j; nbn[i] += 1
    disp = np.zeros((N, 3))
    neigh = np.empty(K, np.int64); ovl = np.empty(K, np.int64)
    for it in range(iters):
        for i in range(N):
            if not alive[i]:
                continue
            nn = 0
            for q in range(nbn[i]):
                j = nb[i, q]
                dxp = cells[j, C_x] - cells[i, C_x]; dyp = cells[j, C_y] - cells[i, C_y]
                maxD = cells[i, C_rOuter] + cells[j, C_rOuter]
                if dxp * dxp + dyp * dyp <= maxD * maxD:
                    neigh[nn] = j; nn += 1
            ax = 0.0; ay = 0.0; cur = 0.0; no = 0
            for q in range(nn):
                j = neigh[q]
                o, ux, uy = _directional_overlap(cells, lut, col, i, cells[i, C_packRot], j, mgf)
                if o > 0:
                    cur += o
                    ovl[no] = j; no += 1
                    ax += -ux * o * 0.5
                    ay += -uy * o * 0.5
            rd = 0.0
            if allow_rot and no > 0:
                plus = 0.0; minus = 0.0
                for q in range(no):
                    o = _directional_overlap(cells, lut, col, i, cells[i, C_packRot] + PACK_ROT_TRIAL, ovl[q], mgf)[0]
                    if o > 0:
                        plus += o
                for q in range(no):
                    o = _directional_overlap(cells, lut, col, i, cells[i, C_packRot] - PACK_ROT_TRIAL, ovl[q], mgf)[0]
                    if o > 0:
                        minus += o
                if plus < cur and plus <= minus:
                    rd = PACK_ROT_TRIAL
                elif minus < cur:
                    rd = -PACK_ROT_TRIAL
            disp[i, 0] = ax * damping; disp[i, 1] = ay * damping; disp[i, 2] = rd * damping
        for i in range(N):
            if not alive[i]:
                continue
            cells[i, C_x] += disp[i, 0]; cells[i, C_y] += disp[i, 1]
            if disp[i, 2] != 0:
                cells[i, C_packRot] = cells[i, C_packRot] + disp[i, 2]


@njit(cache=True)
def _prune(cells, grid, alive, cx0, cy0, cx1, cy1, lut, col, P):
    NR = _interaction_chunks(P)
    mgf = P[P_packFrac]
    N = cells.shape[0]; Ht = cy1 - cy0 + 1
    rem = np.zeros(N, np.bool_)
    for i in range(N):
        if not alive[i]:
            continue
        ccx = int(cells[i, C_cx]); ccy = int(cells[i, C_cy])
        for dx in range(-NR, NR + 1):
            for dy in range(-NR, NR + 1):
                if dx == 0 and dy == 0:
                    continue
                j = _find(grid, alive, cx0, cy0, cx1, cy1, Ht, ccx + dx, ccy + dy)
                if j < 0:
                    continue
                dxp = cells[j, C_x] - cells[i, C_x]; dyp = cells[j, C_y] - cells[i, C_y]
                maxD = cells[i, C_rOuter] + cells[j, C_rOuter]
                if dxp * dxp + dyp * dyp > maxD * maxD:
                    continue
                if _directional_overlap(cells, lut, col, i, cells[i, C_packRot], j, mgf)[0] > 0:
                    if cells[i, C_priority] >= cells[j, C_priority]:
                        rem[j] = True
                    else:
                        rem[i] = True
    removed = 0
    for i in range(N):
        if rem[i]:
            alive[i] = False; removed += 1
    return removed


@njit(cache=True)
def _loop_count(iters):
    return int(math.ceil(iters)) if iters > 0 else 0


@njit(cache=True)
def _pack_block(seed, bx, by, P):
    """World::PackBlock: the packed cells whose home chunk lies in block (bx, by)"""
    cx0 = bx * PACK_BLOCK_CHUNKS; cy0 = by * PACK_BLOCK_CHUNKS
    cx1 = cx0 + PACK_BLOCK_CHUNKS - 1; cy1 = cy0 + PACK_BLOCK_CHUNKS - 1
    if P[P_enablePacking] == 0:
        out = np.zeros((PACK_BLOCK_CHUNKS * PACK_BLOCK_CHUNKS, NC_)); n = 0
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                c = _raw_candidate(seed, cx, cy, P)
                if c[C_present] != 0:
                    out[n] = c; n += 1
        return out[:n].copy()
    M = _interaction_chunks(P) + 2
    mx0 = cx0 - M; my0 = cy0 - M; mx1 = cx1 + M; my1 = cy1 + M
    W = mx1 - mx0 + 1; Ht = my1 - my0 + 1
    grid = np.full(W * Ht, -1, np.int64)
    tmp = np.zeros((W * Ht, NC_)); n = 0
    for cx in range(mx0, mx1 + 1):
        for cy in range(my0, my1 + 1):
            c = _raw_candidate(seed, cx, cy, P)
            if c[C_present] == 0:
                continue
            grid[(cx - mx0) * Ht + (cy - my0)] = n
            tmp[n] = c; n += 1
    cells = tmp[:n].copy()
    alive = np.ones(n, np.bool_)
    lut = np.zeros((n, RADIUS_LUT_N), np.float32); col = np.zeros((n, OUTLINE_SAMPLE_N * 2), np.float32)
    _ensure_luts(cells, lut, col)
    _relax(cells, grid, alive, mx0, my0, mx1, my1, lut, col, P, _loop_count(P[P_relaxIters]))
    for rnd in range(PRUNE_ROUNDS):
        r = _prune(cells, grid, alive, mx0, my0, mx1, my1, lut, col, P)
        if r == 0:
            break
        _relax(cells, grid, alive, mx0, my0, mx1, my1, lut, col, P, _loop_count(min(8.0, P[P_relaxIters])))
    keep = np.zeros(n, np.bool_)
    for i in range(n):
        cx = cells[i, C_cx]; cy = cells[i, C_cy]
        keep[i] = alive[i] and cx >= cx0 and cx <= cx1 and cy >= cy0 and cy <= cy1
    return cells[keep].copy()


# ---------------------------------------------------------------------------------------------------------------------
# cytoplasm height field (cytomesh.cpp)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _cell_outline_local(c, n):
    out = np.empty((n, 2))
    for i in range(n):
        th = (i / n) * PI * 2
        r = _cell_radius_at(c, th)
        out[i, 0] = math.cos(th) * r; out[i, 1] = math.sin(th) * r
    return out


@njit(cache=True)
def _nearest_dist_to_outline(pts, x, y):
    best = np.inf
    n = pts.shape[0]
    j = n - 1
    for i in range(n):
        ax = pts[j, 0]; ay = pts[j, 1]; bx = pts[i, 0]; by = pts[i, 1]
        dx = bx - ax; dy = by - ay
        l2 = dx * dx + dy * dy
        t = ((x - ax) * dx + (y - ay) * dy) / l2 if l2 > 1e-12 else 0.0
        t = max(0.0, min(1.0, t))
        px = ax + dx * t; py = ay + dy * t
        d = _hypot2(x - px, y - py)
        if d < best:
            best = d
        j = i
    return best


@njit(cache=True)
def _nearest_point_on_ellipse(a, b, x, y):
    swapped = False
    if a < b:
        a, b = b, a; x, y = y, x; swapped = True
    sx = -1.0 if x < 0 else 1.0; sy = -1.0 if y < 0 else 1.0
    x0 = abs(x); y0 = abs(y)
    if y0 > 1e-9:
        if x0 > 1e-9:
            lo = -b * b; hi = a * x0 + b * y0
            t = (lo + hi) / 2
            for it in range(80):
                ta = t + a * a; tb = t + b * b
                rx = (a * x0) / ta; ry = (b * y0) / tb
                f = rx * rx + ry * ry - 1
                if f > 0:
                    lo = t
                else:
                    hi = t
                if abs(f) < 1e-12:
                    break
                df = -2 * ((a * a * x0 * x0) / (ta * ta * ta) + (b * b * y0 * y0) / (tb * tb * tb))
                tn = t - f / df if df != 0 else (lo + hi) / 2
                if not (tn > lo and tn < hi):
                    tn = (lo + hi) / 2
                t = tn
            ex = (a * a * x0) / (t + a * a); ey = (b * b * y0) / (t + b * b)
        else:
            ex = 0.0; ey = b
    else:
        numer = a * x0; denom = a * a - b * b
        if denom > 1e-9 and numer < denom:
            xde = numer / denom
            ex = a * xde; ey = b * math.sqrt(max(0.0, 1 - xde * xde))
        else:
            ex = a; ey = 0.0
    ex *= sx; ey *= sy
    if swapped:
        ex, ey = ey, ex
    return ex, ey


@njit(cache=True)
def _nucleus_signed_dist_local(c, lx0, ly0):
    dx = lx0 - c[C_nucOffX]; dy = ly0 - c[C_nucOffY]
    cr = math.cos(-c[C_nucRot]); sr = math.sin(-c[C_nucRot])
    lx = dx * cr - dy * sr; ly = dx * sr + dy * cr
    a = c[C_nucLong] / 2; b = c[C_nucShort] / 2
    ex, ey = _nearest_point_on_ellipse(a, b, lx, ly)
    dist = _hypot2(lx - ex, ly - ey)
    inside = (lx * lx) / (a * a) + (ly * ly) / (b * b) < 1
    return -dist if inside else dist


@njit(cache=True)
def _cyto_height_at(c, P, dEdge, dNuc):
    rise = max(1e-6, c[C_edge])
    slope_on = P[P_cytoMaxSlope] > 0
    inner = _cyto_dome_reach(c, P)
    mid_dist = max(0.0, c[C_midDist]) * c[C_rOuter]
    if slope_on:
        mid_dist = max(mid_dist, 1.5 * max(0.0, c[C_mid] - c[C_rim]) / P[P_cytoMaxSlope])
    h_edge = c[C_rim] * (1 - math.exp(-dEdge / rise))
    h_env = 0.0
    if dNuc <= 0:
        h_env = c[C_height]
    elif dNuc < inner:
        h_env = _lerp(c[C_mid], c[C_height], _smoothstep(1 - dNuc / inner))
    if dNuc < inner:
        h_cyto = c[C_mid]
    elif dNuc < inner + mid_dist:
        h_cyto = _lerp(c[C_rim], c[C_mid], _smoothstep(1 - (dNuc - inner) / max(1e-6, mid_dist)))
    else:
        h_cyto = c[C_rim]
    h_cyto = max(h_edge, h_cyto)
    if slope_on:
        Hc = 2 * c[C_mid]
        h_cyto = min(h_cyto, Hc * (1 - math.exp(-P[P_cytoMaxSlope] * dEdge / Hc)))
    h = min(c[C_height], max(h_cyto, h_env))
    edge_ceiling = c[C_height] * (1 - math.exp(-dEdge / rise))
    return min(h, edge_ceiling)


@njit(cache=True)
def _build_cyto_mesh(c, P):
    n = int(max(8.0, _jsround(P[P_cytoTheta])))
    outline = _cell_outline_local(c, n)
    rings = int(max(2.0, _jsround(P[P_cytoRings])))
    h = np.empty((rings + 1, n))
    for k in range(rings + 1):
        frac = 1.0 if k == rings else 1 - math.pow(1 - k / rings, CYTO_RING_BIAS)
        for i in range(n):
            lx = outline[i, 0] * frac; ly = outline[i, 1] * frac
            dEdge = 0.0 if frac >= 1 else _nearest_dist_to_outline(outline, lx, ly)
            dNuc = _nucleus_signed_dist_local(c, lx, ly)
            h[k, i] = _cyto_height_at(c, P, dEdge, dNuc)
    passes = P[P_cytoSmoothPasses]
    nxt = np.empty_like(h)
    ps = 0
    while ps < passes:
        for k in range(1, rings):
            km = k - 1; kp = k + 1
            for i in range(n):
                im = (i - 1 + n) % n; ip = (i + 1) % n
                s = (h[k, i] * 4 + (h[k, im] + h[k, ip] + h[km, i] + h[kp, i]) * 2
                     + h[km, im] + h[km, ip] + h[kp, im] + h[kp, ip])
                nxt[k, i] = s / 16
        for k in range(1, rings):
            for i in range(n):
                h[k, i] = nxt[k, i]
        ps += 1
    return h


@njit(cache=True)
def _sample_cyto_mesh_height(c, mesh, x, y):
    rings = mesh.shape[0] - 1; n = mesh.shape[1]
    theta = math.atan2(y, x)
    tn = np.fmod(np.fmod(theta / (PI * 2), 1.0) + 1, 1.0)
    iF = tn * n
    i0 = int(math.floor(iF)) % n; i1 = (i0 + 1) % n
    ti = iF - math.floor(iF)
    rc = _cell_radius_at(c, theta)
    f = min(1.0, max(0.0, _hypot2(x, y) / rc if rc > 1e-9 else 0.0))
    kF = min(float(rings), rings * (1 - math.pow(1 - f, 1 / CYTO_RING_BIAS)))
    k0 = min(rings - 1, int(math.floor(kF))); k1 = k0 + 1
    tk = kF - k0
    return ((mesh[k0, i0] * (1 - ti) + mesh[k0, i1] * ti) * (1 - tk)
            + (mesh[k1, i0] * (1 - ti) + mesh[k1, i1] * ti) * tk)


# ---------------------------------------------------------------------------------------------------------------------
# microtubules (microtubules.cpp)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _mt_nucleus_footprint_blend(c, x, y):
    dxN = x - c[C_nucOffX]; dyN = y - c[C_nucOffY]
    cr = math.cos(-c[C_nucRot]); sr = math.sin(-c[C_nucRot])
    lx = dxN * cr - dyN * sr; ly = dxN * sr + dyN * cr
    a = c[C_nucLong] / 2; b = c[C_nucShort] / 2
    norm = _hypot2(lx / max(1e-6, a), ly / max(1e-6, b))
    t = min(1.0, max(0.0, (norm - 1) / MT_NUCLEUS_CLEAR_BLEND))
    return _smoothstep(1 - t)


@njit(cache=True)
def _mt_enforce_min_turn_radius(px, py, min_r):
    n = px.shape[0]
    if not (min_r > 0) or n < 3:
        return
    ex = px[n - 1]; ey = py[n - 1]
    sx = np.empty(n - 1); sy = np.empty(n - 1); sl = np.empty(n - 1)
    for rnd in range(8):
        for i in range(1, n):
            sx[i - 1] = px[i] - px[i - 1]; sy[i - 1] = py[i] - py[i - 1]
            sl[i - 1] = _hypot2(sx[i - 1], sy[i - 1])
        changed = False
        for i in range(1, n - 1):
            pl = sl[i - 1]; cl = sl[i]
            if pl < 1e-9 or cl < 1e-9:
                continue
            pa = math.atan2(sy[i - 1], sx[i - 1]); ca = math.atan2(sy[i], sx[i])
            da = ca - pa
            while da > PI:
                da -= 2 * PI
            while da < -PI:
                da += 2 * PI
            avg = (pl + cl) / 2
            cap = 2 * math.asin(min(1.0, avg / (2 * min_r)))
            if abs(da) <= cap:
                continue
            cla = pa + (1.0 if da > 0 else -1.0) * cap
            sx[i] = math.cos(cla) * cl; sy[i] = math.sin(cla) * cl
            changed = True
        for i in range(1, n):
            px[i] = px[i - 1] + sx[i - 1]; py[i] = py[i - 1] + sy[i - 1]
        dxe = ex - px[n - 1]; dye = ey - py[n - 1]
        if abs(dxe) > 1e-9 or abs(dye) > 1e-9:
            for i in range(1, n - 1):
                t = i / (n - 1)
                s = t * t * (3 - 2 * t)
                px[i] += dxe * s; py[i] += dye * s
            px[n - 1] = ex; py[n - 1] = ey
        if not changed:
            break


@njit(cache=True)
def _mt_trim_steep_ends(pts, max_slope):
    if not (max_slope > 0):
        return pts
    n = pts.shape[0]
    front = 0
    for k in range(MT_MAX_END_TRIM):
        if not (n - front > 2):
            break
        lat = _hypot2(pts[front + 1, 0] - pts[front, 0], pts[front + 1, 1] - pts[front, 1])
        if abs(pts[front + 1, 2] - pts[front, 2]) <= max_slope * max(lat, 1e-6):
            break
        front += 1
    end = n
    for k in range(MT_MAX_END_TRIM):
        if not (end - front > 2):
            break
        last = end - 1
        lat = _hypot2(pts[last, 0] - pts[last - 1, 0], pts[last, 1] - pts[last - 1, 1])
        if abs(pts[last, 2] - pts[last - 1, 2]) <= max_slope * max(lat, 1e-6):
            break
        end -= 1
    return pts[front:end].copy()


@njit(cache=True)
def _mt_limit_z_slope_realized(pts, max_slope):
    n = pts.shape[0]
    if not (max_slope > 0) or n < 3:
        return
    for ps in range(20):
        changed = False
        fwd = ps % 2 == 0
        for k in range(1, n - 1):
            i = k if fwd else n - 1 - k
            dp = _hypot2(pts[i, 0] - pts[i - 1, 0], pts[i, 1] - pts[i - 1, 1]) * max_slope
            dn = _hypot2(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]) * max_slope
            lo = max(pts[i - 1, 2] - dp, pts[i + 1, 2] - dn)
            hi = min(pts[i - 1, 2] + dp, pts[i + 1, 2] + dn)
            if lo > hi:
                continue
            if pts[i, 2] < lo:
                pts[i, 2] = lo; changed = True
            elif pts[i, 2] > hi:
                pts[i, 2] = hi; changed = True
        if not changed:
            break


@njit(cache=True)
def _mt_nucleus_radius_at(c, theta):
    a = c[C_nucLong] / 2; b = c[C_nucShort] / 2
    phi = theta - c[C_nucRot]
    return (a * b) / _hypot2(b * math.cos(phi), a * math.sin(phi))


@njit(cache=True)
def _mt_ray_cell_boundary(c, theta, r0):
    ux = math.cos(theta); uy = math.sin(theta)
    maxT = (c[C_rOuter] + _hypot2(c[C_nucOffX], c[C_nucOffY])) * 1.3 + 1
    dt = max(1e-3, (maxT - r0) / MT_MARCH_STEPS)
    prevT = r0; tLo = 0.0; tHi = 0.0; found = False
    for s in range(MT_MARCH_STEPS + 1):
        t = r0 + s * dt
        px = c[C_nucOffX] + ux * t; py = c[C_nucOffY] + uy * t
        rc = _cell_radius_at(c, math.atan2(py, px))
        if _hypot2(px, py) > rc:
            tHi = t; tLo = t if s == 0 else prevT; found = True
            break
        prevT = t
    if not found:
        return prevT
    for it in range(MT_BISECT_ITERS):
        mid = (tLo + tHi) / 2
        px = c[C_nucOffX] + ux * mid; py = c[C_nucOffY] + uy * mid
        rc = _cell_radius_at(c, math.atan2(py, px))
        if _hypot2(px, py) > rc:
            tHi = mid
        else:
            tLo = mid
    return (tLo + tHi) / 2


@njit(cache=True)
def _mt_sample_direction(dir_theta, dir_cum, dir_total, u1, u2):
    if dir_total <= 1e-9:
        return u1 * PI * 2
    target = u1 * dir_total
    lo = 0; hi = MT_N_DIR
    while lo < hi:
        mid = (lo + hi) >> 1
        if dir_cum[mid + 1] < target:
            lo = mid + 1
        else:
            hi = mid
    i = min(MT_N_DIR - 1, lo)
    return dir_theta[i] + (u2 - 0.5) * ((PI * 2) / MT_N_DIR)


@njit(cache=True)
def _mt_clamp_into_cytoplasm(c, mesh, pts, j):
    x = pts[j, 0]; y = pts[j, 1]; z = pts[j, 2]
    dxN = x - c[C_nucOffX]; dyN = y - c[C_nucOffY]
    cr = math.cos(-c[C_nucRot]); sr = math.sin(-c[C_nucRot])
    lx = dxN * cr - dyN * sr; ly = dxN * sr + dyN * cr
    a = c[C_nucLong] / 2; b = c[C_nucShort] / 2; rz = c[C_nucHeight] / 2
    nz = (z - c[C_nucZ]) / max(1e-6, rz)
    nx = lx / max(1e-6, a); ny = ly / max(1e-6, b)
    en = math.sqrt(nx * nx + ny * ny + nz * nz)
    if en < 1:
        scale = (1 / en if en > 1e-9 else 1.0) / MT_CONTAIN_MARGIN
        lx2 = lx * scale; ly2 = ly * scale
        z = c[C_nucZ] + nz * rz * scale
        cr2 = math.cos(c[C_nucRot]); sr2 = math.sin(c[C_nucRot])
        x = c[C_nucOffX] + lx2 * cr2 - ly2 * sr2
        y = c[C_nucOffY] + lx2 * sr2 + ly2 * cr2
    rc = _cell_radius_at(c, math.atan2(y, x))
    dist = _hypot2(x, y)
    if dist > rc:
        scale = (rc * MT_CONTAIN_MARGIN) / max(1e-9, dist)
        x *= scale; y *= scale
    top = max(0.0, _sample_cyto_mesh_height(c, mesh, x, y))
    if z < 0:
        z = 0.0
    elif z > top:
        z = top * MT_CONTAIN_MARGIN
    pts[j, 0] = x; pts[j, 1] = y; pts[j, 2] = z


@njit(cache=True)
def _mt_cell_geom(c, P):
    dir_theta = np.empty(MT_N_DIR); dir_cum = np.zeros(MT_N_DIR + 1)
    for i in range(MT_N_DIR):
        th = (i / MT_N_DIR) * PI * 2
        rN = _mt_nucleus_radius_at(c, th)
        rC = _mt_ray_cell_boundary(c, th, rN)
        dir_theta[i] = th
        dir_cum[i + 1] = dir_cum[i] + max(0.0, rC - rN)
    ol = _cell_outline_local(c, 48)
    area = 0.0
    n = ol.shape[0]
    for i in range(n):
        j = (i + 1) % n
        area += ol[i, 0] * ol[j, 1] - ol[j, 0] * ol[i, 1]
    area = abs(area) / 2
    mesh = _build_cyto_mesh(c, P)
    return dir_theta, dir_cum, area, mesh


@njit(cache=True)
def _mt_generate_one(seed, mt_index, resample_round, c, P, dir_theta, dir_cum, mesh):
    cx = int(c[C_cx]); cy = int(c[C_cy])
    base = (_U(MT_STREAM_BASE) + _U(mt_index) + _U(resample_round) * _U(MT_RESAMPLE_SPACING)) & _U(0xFFFFFFFF)
    b4096 = (base * _U(4096)) & _U(0xFFFFFFFF)
    ctr = np.zeros(1, np.int64)
    dir_total = dir_cum[MT_N_DIR]
    u1 = _stream_next(seed, cx, cy, b4096, ctr)
    u2 = _stream_next(seed, cx, cy, b4096, ctr)
    theta0 = _mt_sample_direction(dir_theta, dir_cum, dir_total, u1, u2)
    phi0 = theta0 - c[C_nucRot]
    cosPsi = 1 - 2 * _stream_next(seed, cx, cy, b4096, ctr)
    sinPsi = math.sqrt(max(0.0, 1 - cosPsi * cosPsi))
    aN = c[C_nucLong] / 2; bN = c[C_nucShort] / 2; rzN = c[C_nucHeight] / 2
    lx0 = aN * sinPsi * math.cos(phi0); ly0 = bN * sinPsi * math.sin(phi0); lz0 = rzN * cosPsi
    crN = math.cos(c[C_nucRot]); srN = math.sin(c[C_nucRot])
    surfX = c[C_nucOffX] + lx0 * crN - ly0 * srN
    surfY = c[C_nucOffY] + lx0 * srN + ly0 * crN
    surfZ = c[C_nucZ] + lz0
    ox = surfX - c[C_nucOffX]; oy = surfY - c[C_nucOffY]; oz = lz0
    nrm = _normor1(_hypot3(ox, oy, oz))
    ox = ox / nrm; oy = oy / nrm; oz = oz / nrm
    rNuc0 = _mt_nucleus_radius_at(c, theta0)
    rCell0 = _mt_ray_cell_boundary(c, theta0, rNuc0)
    startFrac = _lerp(P[P_mtStartFracMin], P[P_mtStartFracMax], _stream_next(seed, cx, cy, b4096, ctr))
    startDist = startFrac * rCell0
    startX = surfX + ox * startDist; startY = surfY + oy * startDist
    startZ = surfZ + oz * startDist
    offR = _stream_next(seed, cx, cy, b4096, ctr) * max(0.0, P[P_mtStartOffsetXY])
    offAng = _stream_next(seed, cx, cy, b4096, ctr) * PI * 2
    startX += math.cos(offAng) * offR; startY += math.sin(offAng) * offR
    jitterRad = (_stream_next(seed, cx, cy, b4096, ctr) * 2 - 1) * P[P_mtEndJitterDeg] * PI / 180
    thetaEnd = theta0 + jitterRad
    rNucEnd = _mt_nucleus_radius_at(c, thetaEnd)
    rCell1 = _mt_ray_cell_boundary(c, thetaEnd, rNucEnd)
    endFrac = _lerp(P[P_mtEndFracMin], P[P_mtEndFracMax], _stream_next(seed, cx, cy, b4096, ctr))
    endR = max(0.0, rCell1 - endFrac * rCell1)
    endX = c[C_nucOffX] + math.cos(thetaEnd) * endR
    endY = c[C_nucOffY] + math.sin(thetaEnd) * endR
    startTopH = max(0.0, _sample_cyto_mesh_height(c, mesh, startX, startY))
    fracStart = min(1.0, max(0.0, startZ / startTopH)) if startTopH > 1e-9 else 0.0
    fracEnd = _stream_next(seed, cx, cy, b4096, ctr)

    dx = endX - startX; dy = endY - startY
    straightLen = _hypot2(dx, dy)
    turnMag = max(0.0, P[P_mtWobbleTurn])
    pathBudget = max(straightLen, straightLen * max(1.0, P[P_mtWobbleFactor])) if turnMag > 0 else straightLen
    steps = int(min(float(MT_MAX_STEPS_PER_MT), max(4.0, _jsround(pathBudget / max(0.02, P[P_mtStepLen])))))
    stepLen = pathBudget / steps
    smoothLp = max(stepLen, P[P_mtSmoothLen])
    kick = math.sqrt(stepLen / smoothLp)
    minTR = P[P_mtMinTurnRadius]
    capT = 2 * math.asin(min(1.0, stepLen / (2 * minTR))) if minTR > 0 else PI
    hx = 1.0; hy = 0.0
    if straightLen > 1e-9:
        hx = dx / straightLen; hy = dy / straightLen
    rx = np.zeros(steps + 1); ry = np.zeros(steps + 1)
    for i in range(1, steps + 1):
        if turnMag > 0:
            n1 = _stream_next(seed, cx, cy, b4096, ctr) * 2 - 1
            ta = math.atan2(turnMag * kick * n1, 1)
            if ta > capT:
                ta = capT
            elif ta < -capT:
                ta = -capT
            cT = math.cos(ta); sT = math.sin(ta)
            nhx = hx * cT - hy * sT; nhy = hx * sT + hy * cT
            hx = nhx; hy = nhy
        rx[i] = rx[i - 1] + hx * stepLen; ry[i] = ry[i - 1] + hy * stepLen
    driftX = dx - rx[steps]; driftY = dy - ry[steps]
    px = np.empty(steps + 1); py = np.empty(steps + 1)
    for i in range(steps + 1):
        t = i / steps
        s = t * t * (3 - 2 * t)
        px[i] = startX + rx[i] + driftX * s; py[i] = startY + ry[i] + driftY * s
    _mt_enforce_min_turn_radius(px, py, minTR)

    fnAmp = 0.15 * turnMag
    fnDecay = math.exp(-stepLen / smoothLp)
    fnInnov = math.sqrt(max(0.0, 1 - fnDecay * fnDecay))
    fn = 0.0
    nucTop = c[C_nucZ] + c[C_nucHeight] / 2
    nucBot = c[C_nucZ] - c[C_nucHeight] / 2
    clr = max(0.05, 0.5 * P[P_nucMargin])
    canUnder = nucBot - clr > 0
    goOver = (not canUnder) or cosPsi >= 0
    pts = np.empty((steps + 1, 3))
    for i in range(steps + 1):
        t = i / steps
        s = t * t * (3 - 2 * t)
        x = px[i]; y = py[i]
        ceilH = max(0.0, _sample_cyto_mesh_height(c, mesh, x, y))
        if i == 0:
            z = fracStart * ceilH
        else:
            if i == steps:
                zN = fracEnd * ceilH
            else:
                u = _stream_next(seed, cx, cy, b4096, ctr)
                fn = fnDecay * fn + fnAmp * fnInnov * (u * 2 - 1)
                fr = min(1.0, max(0.0, _lerp(fracStart, fracEnd, s) + fn * math.sin(PI * t)))
                zN = fr * ceilH
            blend = _mt_nucleus_footprint_blend(c, x, y)
            if blend <= 0:
                z = zN
            else:
                if goOver:
                    target = max(zN, min(ceilH, nucTop + clr))
                else:
                    target = min(zN, max(0.0, nucBot - clr))
                z = _lerp(zN, target, blend)
        pts[i, 0] = x; pts[i, 1] = y; pts[i, 2] = z

    win = max(0.0, _jsround(P[P_mtSmoothLen] / stepLen))
    if win > 0:
        zo = pts[:, 2].copy()
        half = int(max(1.0, _jsround(win / 2)))
        last = pts.shape[0] - 1
        for i in range(1, last):
            lo = max(0, i - half); hi = min(last, i + half)
            sm = 0.0; nn = 0
            for j in range(lo, hi + 1):
                sm += zo[j]; nn += 1
            pts[i, 2] = sm / nn

    maxDz = max(1.0, P[P_mtMaxZSlope]) * stepLen
    n = pts.shape[0]
    for ps in range(20):
        changed = False
        fwd = ps % 2 == 0
        for k in range(1, n - 1):
            i = k if fwd else n - 1 - k
            lo = max(pts[i - 1, 2] - maxDz, pts[i + 1, 2] - maxDz)
            hi = min(pts[i - 1, 2] + maxDz, pts[i + 1, 2] + maxDz)
            if lo > hi:
                continue
            if pts[i, 2] < lo:
                pts[i, 2] = lo; changed = True
            elif pts[i, 2] > hi:
                pts[i, 2] = hi; changed = True
        if not changed:
            break

    cut = n
    for i in range(n):
        if _hypot2(pts[i, 0], pts[i, 1]) > _cell_radius_at(c, math.atan2(pts[i, 1], pts[i, 0])):
            cut = i
            break
    pts = pts[:cut].copy()
    for j in range(pts.shape[0]):
        _mt_clamp_into_cytoplasm(c, mesh, pts, j)
    prio = _stream_next(seed, cx, cy, b4096, ctr)
    return pts, prio


# ---- collision resolution: the JS Map<key, [i, j, ...]> as stably sorted keys (same buckets, same order) ----
@njit(cache=True)
def _bucket_key(ix, iy, iz):
    M = _U(0xFFFFFFFF)
    return ((_u32(ix) * _U(73856093)) & M) ^ ((_u32(iy) * _U(19349663)) & M) ^ ((_u32(iz) * _U(83492791)) & M)


@njit(cache=True)
def _build_index(mts, cs):
    tot = 0
    for m in mts:
        tot += m.shape[0]
    keys = np.empty(tot, np.uint64); ii = np.empty(tot, np.int64); jj = np.empty(tot, np.int64)
    q = 0
    for i in range(len(mts)):
        m = mts[i]
        for j in range(m.shape[0]):
            keys[q] = _bucket_key(int(math.floor(m[j, 0] / cs)), int(math.floor(m[j, 1] / cs)),
                                  int(math.floor(m[j, 2] / cs)))
            ii[q] = i; jj[q] = j; q += 1
    order = np.argsort(keys, kind='mergesort')
    return keys[order], ii[order], jj[order]


@njit(cache=True)
def _mt_nudge_round(c, mesh, mts, min_sep):
    cs = max(min_sep, 1e-6)
    sk, si, sj = _build_index(mts, cs)
    pushes = List()
    for m in mts:
        pushes.append(np.zeros((m.shape[0], 3)))
    anyv = False; ops = 0; stop = False
    for i in range(len(mts)):
        if stop:
            break
        pts = mts[i]
        for j in range(1, pts.shape[0] - 1):
            if stop:
                break
            ix = math.floor(pts[j, 0] / cs); iy = math.floor(pts[j, 1] / cs); iz = math.floor(pts[j, 2] / cs)
            for dx in range(-1, 2):
                if stop:
                    break
                for dy in range(-1, 2):
                    if stop:
                        break
                    for dz in range(-1, 2):
                        if stop:
                            break
                        key = _bucket_key(int(ix + dx), int(iy + dy), int(iz + dz))
                        lo = np.searchsorted(sk, key, side='left'); hi = np.searchsorted(sk, key, side='right')
                        for q in range(lo, hi):
                            i2 = si[q]; j2 = sj[q]
                            if i2 == i:
                                continue
                            ops += 1
                            if ops > MT_COLLISION_OP_BUDGET:
                                stop = True
                                break
                            p2 = mts[i2]
                            ddx = pts[j, 0] - p2[j2, 0]; ddy = pts[j, 1] - p2[j2, 1]; ddz = pts[j, 2] - p2[j2, 2]
                            d = _hypot3(ddx, ddy, ddz)
                            if d >= min_sep:
                                continue
                            anyv = True
                            if d < 1e-6:
                                nx = 1.0; ny = 0.0; nz = 0.0
                            else:
                                nx = ddx / d; ny = ddy / d; nz = ddz / d
                            mag = (min_sep - d) / 2 + 1e-4
                            pu = pushes[i]
                            pu[j, 0] += nx * mag; pu[j, 1] += ny * mag; pu[j, 2] += nz * mag
    if not anyv:
        return False
    for i in range(len(mts)):
        pts = mts[i]; pu = pushes[i]
        for j in range(1, pts.shape[0] - 1):
            if pu[j, 0] == 0 and pu[j, 1] == 0 and pu[j, 2] == 0:
                continue
            pts[j, 0] += pu[j, 0]; pts[j, 1] += pu[j, 1]; pts[j, 2] += pu[j, 2]
            _mt_clamp_into_cytoplasm(c, mesh, pts, j)
    return True


@njit(cache=True)
def _nudge_pass(c, mesh, mts, min_sep, rounds):
    for r in range(rounds):
        if not _mt_nudge_round(c, mesh, mts, min_sep):
            return True
    return False


@njit(cache=True)
def _mt_resolve_collisions(seed, c, P, mts, prios, dir_theta, dir_cum, mesh):
    min_sep = max(0.0, P[P_mtMinSeparation])
    if min_sep <= 0 or len(mts) < 2:
        return
    tot = 0
    for m in mts:
        tot += m.shape[0]
    if tot > MT_COLLISION_MAX_TOTAL_POINTS:
        return
    if _nudge_pass(c, mesh, mts, min_sep, MT_NUDGE_ROUNDS):
        return
    for rr in range(MT_RESAMPLE_ROUNDS):
        cs = max(min_sep, 1e-6)
        sk, si, sj = _build_index(mts, cs)
        viol = np.empty(len(mts), np.int64); nv = 0
        for i in range(len(mts)):
            pts = mts[i]
            found = False
            for j in range(pts.shape[0]):
                if found:
                    break
                ix = math.floor(pts[j, 0] / cs); iy = math.floor(pts[j, 1] / cs); iz = math.floor(pts[j, 2] / cs)
                for dx in range(-1, 2):
                    if found:
                        break
                    for dy in range(-1, 2):
                        if found:
                            break
                        for dz in range(-1, 2):
                            if found:
                                break
                            key = _bucket_key(int(ix + dx), int(iy + dy), int(iz + dz))
                            lo = np.searchsorted(sk, key, side='left'); hi = np.searchsorted(sk, key, side='right')
                            for q in range(lo, hi):
                                i2 = si[q]; j2 = sj[q]
                                if i2 == i:
                                    continue
                                p2 = mts[i2]
                                if _hypot3(pts[j, 0] - p2[j2, 0], pts[j, 1] - p2[j2, 1], pts[j, 2] - p2[j2, 2]) < min_sep:
                                    found = True
                                    break
            if found:
                viol[nv] = i; nv += 1
        if nv == 0:
            return
        v = viol[:nv]
        pr = np.empty(nv)
        for q in range(nv):
            pr[q] = prios[v[q]]
        v = v[np.argsort(pr, kind='mergesort')]
        losers = max(1, (nv + 1) // 2)
        for q in range(losers):
            idx = v[q]
            pts, pri = _mt_generate_one(seed, idx, rr + 1, c, P, dir_theta, dir_cum, mesh)
            mts[idx] = pts; prios[idx] = pri
        if _nudge_pass(c, mesh, mts, min_sep, MT_POST_RESAMPLE_NUDGE_ROUNDS):
            return


@njit(cache=True)
def _build_mts_for_cell(seed, c, P):
    dir_theta, dir_cum, area, mesh = _mt_cell_geom(c, P)
    mts = List.empty_list(_ARR2)
    expected = max(0.0, area * P[P_mtDensity])
    base = math.floor(expected)
    roll = _hash_unit(seed, int(c[C_cx]), int(c[C_cy]), MT_CH_COUNT)
    count = min(MT_MAX_PER_CELL, int(base) + (1 if roll < expected - base else 0))
    prios = np.zeros(max(count, 0))
    for i in range(count):
        pts, pri = _mt_generate_one(seed, i, 0, c, P, dir_theta, dir_cum, mesh)
        mts.append(pts); prios[i] = pri
    if count > 0:
        _mt_resolve_collisions(seed, c, P, mts, prios, dir_theta, dir_cum, mesh)
    for i in range(len(mts)):
        m = _mt_trim_steep_ends(mts[i], P[P_mtMaxZSlope])
        _mt_limit_z_slope_realized(m, P[P_mtMaxZSlope])
        for j in range(m.shape[0]):
            _mt_clamp_into_cytoplasm(c, mesh, m, j)
        mts[i] = m
    return mts, area, mesh


# ---------------------------------------------------------------------------------------------------------------------
# lattice + dyes (microtubules.cpp BuildMtFrames, dyes.cpp DyesInBlock)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _mt_frames(pts):
    n = pts.shape[0]
    cum = np.zeros(n)
    T = np.zeros((max(n - 1, 0), 3)); Uf = np.zeros((max(n - 1, 0), 3)); V = np.zeros((max(n - 1, 0), 3))
    if n < 2:
        return cum, T, Uf, V
    pu0 = 0.0; pu1 = 0.0; pu2 = 0.0
    for i in range(n - 1):
        dx = pts[i + 1, 0] - pts[i, 0]; dy = pts[i + 1, 1] - pts[i, 1]; dz = pts[i + 1, 2] - pts[i, 2]
        ln = _hypot3(dx, dy, dz)
        cum[i + 1] = cum[i] + ln
        if ln > 1e-12:
            t0 = dx / ln; t1 = dy / ln; t2 = dz / ln
        elif i > 0:
            t0 = T[i - 1, 0]; t1 = T[i - 1, 1]; t2 = T[i - 1, 2]
        else:
            t0 = 1.0; t1 = 0.0; t2 = 0.0
        if i == 0:
            if abs(t2) < 0.9:
                u0 = 0.0; u1 = 0.0; u2 = 1.0
            else:
                u0 = 1.0; u1 = 0.0; u2 = 0.0
        else:
            u0 = pu0; u1 = pu1; u2 = pu2
        d = u0 * t0 + u1 * t1 + u2 * t2
        u0 = u0 - d * t0; u1 = u1 - d * t1; u2 = u2 - d * t2
        ul = _normor1(_hypot3(u0, u1, u2))
        u0 = u0 / ul; u1 = u1 / ul; u2 = u2 / ul
        T[i, 0] = t0; T[i, 1] = t1; T[i, 2] = t2
        Uf[i, 0] = u0; Uf[i, 1] = u1; Uf[i, 2] = u2
        V[i, 0] = t1 * u2 - t2 * u1; V[i, 1] = t2 * u0 - t0 * u2; V[i, 2] = t0 * u1 - t1 * u0
        pu0 = u0; pu1 = u1; pu2 = u2
    return cum, T, Uf, V


@njit(cache=True)
def _segment_at(cum, nseg, S):
    lo = 0; hi = nseg - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid + 1] < S:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _dyes_in_blocks(seed, cx, cy, mt_index, pts, cum, T, Uf, V, blocks, eff, binder_nm, link_min_nm, link_max_nm):
    """DyesInBlock for every block in `blocks` (ascending). Returns [n, 7]: x, y, z (cell-local um), k, n, label u,
    hash id. eff = fraction of lattice sites labelled (the LABEL channel, u < eff)."""
    NM = 1e-3
    if pts.shape[0] < 2:
        return np.zeros((0, 7))
    total = cum[cum.shape[0] - 1]
    nseg = T.shape[0]
    eff = min(1.0, max(0.0, eff))
    sd = _U(seed) & _U(0xFFFFFFFF)
    h1 = _pcg4d(sd ^ _U(DYE_SALT), _u32(cx), _u32(cy), _u32(mt_index))[0]
    phase = 2 * PI * _hash_unit(sd, cx, cy, MT_CH_SEAM_PHASE + mt_index)
    lmin = link_min_nm * NM; lmax = link_max_nm * NM
    minL3 = math.pow(lmin, 3); maxL3 = math.pow(lmax, 3)
    R = MT_RADIUS_NM * NM; B = (MT_RADIUS_NM + binder_nm) * NM
    cap = 0
    for bi in range(blocks.shape[0]):
        cap += 13 * (int(DYE_BLOCK_UM * 1000 / MT_DIMER_NM) + 3)
    out = np.empty((cap, 7)); q = 0
    if not (eff > 0):
        return out[:0]
    for bi in range(blocks.shape[0]):
        block = blocks[bi]
        blockNm0 = block * DYE_BLOCK_UM * 1000
        if blockNm0 * NM >= total:
            continue
        for k in range(MT_N_PROTOFILAMENTS):
            off = np.fmod(k * MT_LATTICE_START * MT_DIMER_NM / MT_N_PROTOFILAMENTS, MT_DIMER_NM)
            theta = phase + k * 2 * PI / MT_N_PROTOFILAMENTS
            ct = math.cos(theta); st = math.sin(theta)
            n = max(0, int(math.floor((blockNm0 - off) / MT_DIMER_NM)) - 1)
            while True:
                sNm = off + n * MT_DIMER_NM
                blk = math.floor(sNm / (DYE_BLOCK_UM * 1000))
                if blk < block:
                    n += 1
                    continue
                if blk > block:
                    break
                S = sNm * NM
                if S >= total:
                    break
                lab = _pcg4d(h1, _U(k), _U(n), _U(0))[0]
                u = _unit(lab)
                if u >= eff:
                    n += 1
                    continue
                r1 = _unit(_pcg4d(h1, _U(k), _U(n), _U(1))[0])
                r2 = _unit(_pcg4d(h1, _U(k), _U(n), _U(2))[0])
                r3 = _unit(_pcg4d(h1, _U(k), _U(n), _U(3))[0])
                seg = _segment_at(cum, nseg, S)
                f = S - cum[seg]
                ccx = pts[seg, 0] + T[seg, 0] * f; ccy = pts[seg, 1] + T[seg, 1] * f; ccz = pts[seg, 2] + T[seg, 2] * f
                rx = ct * Uf[seg, 0] + st * V[seg, 0]; ry = ct * Uf[seg, 1] + st * V[seg, 1]
                rz = ct * Uf[seg, 2] + st * V[seg, 2]
                tx = ccx + rx * B; ty = ccy + ry * B; tz = ccz + rz * B
                lu = r1 * 2 - 1; phi = r2 * 2 * PI; sn = math.sqrt(1 - lu * lu)
                r = np.cbrt(minL3 + (maxL3 - minL3) * r3)
                out[q, 0] = tx + r * sn * math.cos(phi)
                out[q, 1] = ty + r * sn * math.sin(phi)
                out[q, 2] = tz + r * lu
                out[q, 3] = k; out[q, 4] = n; out[q, 5] = u; out[q, 6] = float(lab)
                q += 1
                n += 1
    return out[:q]


@njit(cache=True)
def _point_at_arc(pts, cum, T, S):
    seg = _segment_at(cum, T.shape[0], S)
    f = S - cum[seg]
    return pts[seg, 0] + T[seg, 0] * f, pts[seg, 1] + T[seg, 1] * f, pts[seg, 2] + T[seg, 2] * f


# ---------------------------------------------------------------------------------------------------------------------
# world (world.cpp): blocks of packed cells, per-cell assets, dyes in a window
# ---------------------------------------------------------------------------------------------------------------------
def _floordiv(a, b):
    return a // b                       # Python floor division = FloorDiv for b > 0


def _rect_dist(px, py, x0, y0, x1, y1):
    dx = max(max(x0 - px, 0.0), px - x1)
    dy = max(max(y0 - py, 0.0), py - y1)
    return math.hypot(dx, dy)


def local_to_world(c, lx, ly):
    rot = c[C_packRot]
    if rot == 0 or rot != rot:
        return c[C_x] + lx, c[C_y] + ly
    cr, sr = math.cos(rot), math.sin(rot)
    return c[C_x] + lx * cr - ly * sr, c[C_y] + lx * sr + ly * cr


class CellField:
    """One seed-addressed world (World in world.cpp). Units: um, z = height above the coverslip."""

    def __init__(self, seed, params=None, asset_cache=64):
        self.seed = int(seed) & 0xFFFFFFFF
        self.params = dict(DEFAULTS); self.params.update(params or {})
        self.P = params_array(params)
        self._blocks = {}
        self._assets = OrderedDict()
        self._asset_cap = asset_cache

    # -- cells --
    def cell_reach_um(self):
        P = self.P
        worst = (P[P_cellDiamMax] / 2) / math.sqrt(max(0.05, P[P_cellElongMin]))
        return 2 * worst * CELL_MOD_MAX + P[P_chunkSize]

    def packed_block(self, bx, by):
        key = (bx, by)
        if key not in self._blocks:
            self._blocks[key] = _pack_block(_U(self.seed), bx, by, self.P)
        return self._blocks[key]

    def cells_in_rect(self, x0, y0, x1, y1):
        """cells (packed pose) whose footprint circle rOuter intersects the rect; [k, NC] rows (CELL_FIELDS)"""
        S = self.P[P_chunkSize]; reach = self.cell_reach_um()
        cxlo, cxhi = int(math.floor((x0 - reach) / S)), int(math.floor((x1 + reach) / S))
        cylo, cyhi = int(math.floor((y0 - reach) / S)), int(math.floor((y1 + reach) / S))
        out = []
        for bx in range(_floordiv(cxlo, PACK_BLOCK_CHUNKS), _floordiv(cxhi, PACK_BLOCK_CHUNKS) + 1):
            for by in range(_floordiv(cylo, PACK_BLOCK_CHUNKS), _floordiv(cyhi, PACK_BLOCK_CHUNKS) + 1):
                for c in self.packed_block(bx, by):
                    if _rect_dist(c[C_x], c[C_y], x0, y0, x1, y1) <= c[C_rOuter]:
                        out.append(c)
        return np.array(out).reshape(-1, NC_)

    def find_cell(self, cx, cy):
        for c in self.packed_block(_floordiv(cx, PACK_BLOCK_CHUNKS), _floordiv(cy, PACK_BLOCK_CHUNKS)):
            if int(c[C_cx]) == cx and int(c[C_cy]) == cy:
                return c
        return None

    def assets(self, c):
        """microtubule centrelines of a cell (cell-local um, before packRot), footprint area, cytoplasm mesh"""
        key = (int(c[C_cx]), int(c[C_cy]))
        if key in self._assets:
            self._assets.move_to_end(key)
            return self._assets[key]
        mts, area, mesh = _build_mts_for_cell(_U(self.seed), c, self.P)
        mts = [np.asarray(m) for m in mts]
        a = dict(cell=c, mts=mts, area_um2=area, mesh=mesh,
                 reach=np.array([np.hypot(m[:, 0], m[:, 1]).max() if len(m) else 0.0 for m in mts]),
                 frames=[None] * len(mts))
        self._assets[key] = a
        while len(self._assets) > self._asset_cap:
            self._assets.popitem(last=False)
        return a

    def frames(self, a, i):
        if a['frames'][i] is None:
            a['frames'][i] = _mt_frames(a['mts'][i])
        return a['frames'][i]

    # -- dyes --
    def sites_in_window(self, x0, y0, x1, y1, zmin=-np.inf, zmax=np.inf, label_efficiency=1.0,
                        binder_nm=MT_BINDER_NM, linker_nm=(MT_LINKER_MIN_NM, MT_LINKER_MAX_NM)):
        """labelled dyes in [x0,x1) x [y0,y1) x [zmin,zmax) (world um), ordered like World::SitesInWindow.
        Returns dict: xyz [N,3] world um, cell [N,2] (cx, cy), mt [N], k [N], n [N], u [N] (label draw), id [N]."""
        dye_reach = (MT_RADIUS_NM + binder_nm + linker_nm[1]) * 1e-3
        block_reach = DYE_BLOCK_UM / 2 + dye_reach
        parts = []
        for c in self.cells_in_rect(x0, y0, x1, y1):
            A = self.assets(c)
            cdist = _rect_dist(c[C_x], c[C_y], x0, y0, x1, y1)
            for i, pts in enumerate(A['mts']):
                if len(pts) < 2 or cdist > A['reach'][i] + dye_reach:
                    continue
                cum, T, Uf, V = self.frames(A, i)
                L = cum[-1]
                blocks = []
                for b in range(int(math.ceil(L / DYE_BLOCK_UM))):
                    mx, my, mz = _point_at_arc(pts, cum, T, min((b + 0.5) * DYE_BLOCK_UM, L))
                    if mz + block_reach < zmin or mz - block_reach >= zmax:
                        continue
                    wx, wy = local_to_world(c, mx, my)
                    if _rect_dist(wx, wy, x0, y0, x1, y1) > block_reach:
                        continue
                    blocks.append(b)
                if not blocks:
                    continue
                d = _dyes_in_blocks(self.seed, int(c[C_cx]), int(c[C_cy]), i, pts, cum, T, Uf, V,
                                    np.array(blocks, np.int64), label_efficiency, binder_nm, linker_nm[0], linker_nm[1])
                if not len(d):
                    continue
                rot = c[C_packRot]
                if rot == 0 or rot != rot:
                    wx, wy = c[C_x] + d[:, 0], c[C_y] + d[:, 1]
                else:
                    cr, sr = math.cos(rot), math.sin(rot)
                    wx = c[C_x] + d[:, 0] * cr - d[:, 1] * sr
                    wy = c[C_y] + d[:, 0] * sr + d[:, 1] * cr
                z = d[:, 2]
                m = (z >= zmin) & (z < zmax) & (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
                if m.any():
                    parts.append(dict(xyz=np.c_[wx[m], wy[m], z[m]],
                                      cell=np.tile([int(c[C_cx]), int(c[C_cy])], (m.sum(), 1)),
                                      mt=np.full(m.sum(), i), k=d[m, 3].astype(np.int32), n=d[m, 4].astype(np.int64),
                                      u=d[m, 5], id=d[m, 6].astype(np.uint32)))
        if not parts:
            return dict(xyz=np.zeros((0, 3)), cell=np.zeros((0, 2), int), mt=np.zeros(0, int),
                        k=np.zeros(0, np.int32), n=np.zeros(0, np.int64), u=np.zeros(0), id=np.zeros(0, np.uint32))
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    def mts_world(self, c):
        """centrelines of a cell's microtubules in world um"""
        A = self.assets(c)
        out = []
        for m in A['mts']:
            if len(m) == 0:
                out.append(m.copy()); continue
            rot = c[C_packRot]
            if rot == 0 or rot != rot:
                out.append(np.c_[c[C_x] + m[:, 0], c[C_y] + m[:, 1], m[:, 2]])
            else:
                cr, sr = math.cos(rot), math.sin(rot)
                out.append(np.c_[c[C_x] + m[:, 0] * cr - m[:, 1] * sr, c[C_y] + m[:, 0] * sr + m[:, 1] * cr, m[:, 2]])
        return out

    def cell_outline_world(self, c, n=256):
        ol = _cell_outline_local(c, n)
        rot = c[C_packRot]; cr, sr = math.cos(rot), math.sin(rot)
        return np.c_[c[C_x] + ol[:, 0] * cr - ol[:, 1] * sr, c[C_y] + ol[:, 0] * sr + ol[:, 1] * cr]

    def nucleus_outline_world(self, c, n=128):
        th = np.linspace(0, 2 * np.pi, n)
        lx, ly = np.cos(th) * c[C_nucLong] / 2, np.sin(th) * c[C_nucShort] / 2
        nr = c[C_nucRot]
        x = c[C_nucOffX] + lx * np.cos(nr) - ly * np.sin(nr)
        y = c[C_nucOffY] + lx * np.sin(nr) + ly * np.cos(nr)
        rot = c[C_packRot]; cr, sr = math.cos(rot), math.sin(rot)
        return np.c_[c[C_x] + x * cr - y * sr, c[C_y] + x * sr + y * cr]


# ---------------------------------------------------------------------------------------------------------------------
# webSMLM wrapper (CellField.buildWindow, countMts, mtStartPosition, buildMicrotubuleStructure)
# ---------------------------------------------------------------------------------------------------------------------
PAD_UM = (12.5 + 12 + 5) / 1000
MT_START_MIN_COVER, MT_START_TRIES, MT_START_RANGE_UM = 5 / 9, 16, 500
_FIELDS = {}


def get_field(seed, params=None):
    key = (int(seed) & 0xFFFFFFFF, tuple(sorted((params or {}).items())))
    if key not in _FIELDS:
        if len(_FIELDS) > 8:
            _FIELDS.clear()
        _FIELDS[key] = CellField(seed, params)
    return _FIELDS[key]


def count_mts(field, cells, win):
    n = 0
    for c in cells:
        for m in field.mts_world(c):
            if len(m) < 2:
                continue
            X, Y, Z = m[:, 0], m[:, 1], m[:, 2]
            hit = ((np.maximum(X[:-1], X[1:]) + PAD_UM >= win[0]) & (np.minimum(X[:-1], X[1:]) - PAD_UM <= win[2]) &
                   (np.maximum(Y[:-1], Y[1:]) + PAD_UM >= win[1]) & (np.minimum(Y[:-1], Y[1:]) - PAD_UM <= win[3]) &
                   (np.maximum(Z[:-1], Z[1:]) + PAD_UM >= win[4]) & (np.minimum(Z[:-1], Z[1:]) - PAD_UM <= win[5]))
            n += bool(hit.any())
    return n


def build_window(w, h, seed, x_um, y_um, pxnm, focus_um, slab_nm, mt_density=0.9, cell_density=0.33, params=None,
                 count=True):
    """CellField.buildWindow: sites as webSMLM returns them ([x px, y px, z nm from focus]) + nCells, nMt"""
    p = dict(params or {}); p['mtDensity'] = mt_density; p['density'] = cell_density
    F = get_field(seed, p)
    pxum = pxnm / 1000
    hw, hh, slab = w * pxum / 2, h * pxum / 2, slab_nm / 1000
    win = (x_um - hw, y_um - hh, x_um + hw, y_um + hh, focus_um - slab, focus_um + slab)
    s = F.sites_in_window(win[0], win[1], win[2], win[3], win[4], win[5], 1.0)
    X = s['xyz']
    sites = np.c_[w / 2 + (X[:, 0] - x_um) / pxum, h / 2 + (X[:, 1] - y_um) / pxum, (X[:, 2] - focus_um) * 1000]
    res = dict(sites=sites, raw=s)
    if count:
        cells = F.cells_in_rect(win[0] - 0.5, win[1] - 0.5, win[2] + 0.5, win[3] + 0.5)
        res.update(nCells=len(cells), nMt=count_mts(F, cells, win))
    return res


def _jsround_py(x):
    return math.floor(x + 0.5)


def mt_start_position(seed, w, h, pxnm, focus_um, slab_nm, mt_density=0.9, cell_density=0.33, params=None):
    """webSMLM mtStartPosition: first of a seed-fixed candidate list (origin, then mulberry32 draws within +-500 um)
    whose window is >= 5/9 covered, probed on a 3 x 3 grid of ~2 um sub-windows"""
    Wum, Hum, g = w * pxnm / 1000, h * pxnm / 1000, 3
    pw = max(1, _jsround_py(2000 / pxnm))
    s32 = int(seed) & 0xFFFFFFFF
    rng = mulberry32((s32 ^ 0x5EEDCE11) & 0xFFFFFFFF)
    best = None
    for t in range(MT_START_TRIES):
        cx = _jsround_py((rng() * 2 - 1) * MT_START_RANGE_UM * 10) / 10 if t else 0.0
        cy = _jsround_py((rng() * 2 - 1) * MT_START_RANGE_UM * 10) / 10 if t else 0.0
        hit = 0
        for j in range(g):
            for i in range(g):
                r = build_window(pw, pw, seed, cx + ((i + 0.5) / g - 0.5) * Wum, cy + ((j + 0.5) / g - 0.5) * Hum, pxnm,
                                 focus_um, slab_nm, mt_density, cell_density, params, count=False)
                if len(r['sites']):
                    hit += 1
        cand = dict(x=cx, y=cy, cover=hit / (g * g), tries=t + 1)
        if best is None or cand['cover'] > best['cover']:
            best = cand
        if cand['cover'] >= MT_START_MIN_COVER:
            break
    return best


# ---------------------------------------------------------------------------------------------------------------------
# user-facing API
# ---------------------------------------------------------------------------------------------------------------------
def simulate(seed=1, fov_um=(14.1, 14.1), offset_um=(0.0, 0.0), start_view=True, focus_nm=250.0, z_range_nm=750.0,
             label_efficiency=0.7, binder_nm=MT_BINDER_NM, linker_nm=(MT_LINKER_MIN_NM, MT_LINKER_MAX_NM),
             mt_density=0.9, cell_density=0.33, params=None, probe_px_nm=100.0, centrelines=True, stats=True):
    """Label positions of the microtubule cell field in one field of view.

    seed           cell-field seed (webSMLM 'Cell field seed'; webSMLM's 0 = random is not supported: pass a seed)
    fov_um         (W, H) field of view in um (webSMLM: structure FOV = round(1.1 * camera px) * pixel size,
                   default 141 px x 100 nm = 14.1 um)
    offset_um      view offset (webSMLM simulation_mt_x / _y, +y down) from the seed's start view (start_view=True,
                   webSMLM's well-covered start position) or from the world origin (start_view=False)
    focus_nm       centre of the optical section, nm above the coverslip (webSMLM 'Focus height', 250)
    z_range_nm     half-height of the optical section (webSMLM 'Structure Z range', +-750); None = every height
    label_efficiency  fraction of the 1625 lattice sites per um that carry a dye (webSMLM 'Labeling efficiency' 70 %)
    binder_nm, linker_nm  label displacement: binder stalk radially out from the 12.5 nm tube surface (12 nm) and the
                   dye's linker shell around the binder tip (2-5 nm, uniform in volume); webSMLM's values by default
    mt_density, cell_density  microtubules per um^2 of footprint (0.9) and chunk occupancy (0.33), as webSMLM
    params         overrides of any other core parameter (DEFAULTS names)
    probe_px_nm    pixel size used only by the start-view search (webSMLM probes 3 x 3 windows of round(2000/px) px)

    Returns dict:
      xyz [N,3] nm: x, y in the FOV frame (origin at the FOV corner, x right, y down), z height above the coverslip
      mt [N] int: microtubule id (index into 'centrelines'), k/n: protofilament / dimer index of the lattice site
      centrelines: list of [n,3] nm arrays (same frame), microtubules reaching the FOV (+ 1 um margin)
      center_um, start, cells (summary per cell reaching the FOV), stats (densities)
    """
    W, H = (float(fov_um), float(fov_um)) if np.isscalar(fov_um) else map(float, fov_um)
    focus_um = focus_nm / 1000
    slab_nm = np.inf if z_range_nm is None else float(z_range_nm)
    p = dict(params or {}); p['mtDensity'] = mt_density; p['density'] = cell_density
    F = get_field(seed, p)
    st = dict(x=0.0, y=0.0, cover=None, tries=0)
    if start_view:
        wpx, hpx = W * 1000 / probe_px_nm, H * 1000 / probe_px_nm
        st = mt_start_position(seed, wpx, hpx, probe_px_nm, focus_um, 750.0 if z_range_nm is None else z_range_nm,
                               mt_density, cell_density, params)
    cxu = round((st['x'] + offset_um[0]) * 1000) / 1000
    cyu = round((st['y'] + offset_um[1]) * 1000) / 1000
    x0, y0, x1, y1 = cxu - W / 2, cyu - H / 2, cxu + W / 2, cyu + H / 2
    zlo, zhi = focus_um - slab_nm / 1000, focus_um + slab_nm / 1000
    s = F.sites_in_window(x0, y0, x1, y1, zlo, zhi, label_efficiency, binder_nm, tuple(linker_nm))
    out = dict(seed=seed, center_um=(cxu, cyu), fov_um=(W, H), focus_nm=focus_nm, z_range_nm=z_range_nm, start=st,
               label_efficiency=label_efficiency, binder_nm=binder_nm, linker_nm=tuple(linker_nm))
    xyz = s['xyz'].copy()
    xyz[:, 0] = (xyz[:, 0] - x0) * 1000; xyz[:, 1] = (xyz[:, 1] - y0) * 1000; xyz[:, 2] *= 1000
    # microtubule ids: (cx, cy, mt) -> running index
    cells = F.cells_in_rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
    lines, ids, cinfo = [], {}, []
    for c in cells:
        cl = F.mts_world(c)
        A = F.assets(c)
        L = [np.sum(np.hypot(np.hypot(np.diff(m[:, 0]), np.diff(m[:, 1])), np.diff(m[:, 2]))) if len(m) > 1 else 0.0
             for m in cl]
        cinfo.append(dict(cx=int(c[C_cx]), cy=int(c[C_cy]), x_um=c[C_x], y_um=c[C_y], n_mt=len(cl),
                          area_um2=A['area_um2'], mt_length_um=float(np.sum(L)), height_um=c[C_height],
                          nuc_center_um=local_to_world(c, c[C_nucOffX], c[C_nucOffY]), nuc_z_um=c[C_nucZ],
                          nuc_height_um=c[C_nucHeight],
                          outline_um=F.cell_outline_world(c), nucleus_um=F.nucleus_outline_world(c)))
        for i, m in enumerate(cl):
            if len(m) < 2:
                continue
            inside = ((m[:, 0] > x0 - 1) & (m[:, 0] < x1 + 1) & (m[:, 1] > y0 - 1) & (m[:, 1] < y1 + 1))
            if inside.any():
                ids[(int(c[C_cx]), int(c[C_cy]), i)] = len(lines)
                lines.append(np.c_[(m[:, 0] - x0) * 1000, (m[:, 1] - y0) * 1000, m[:, 2] * 1000])
    mt = np.array([ids.get((int(a), int(b), int(i)), -1) for (a, b), i in zip(s['cell'], s['mt'])], dtype=np.int64)
    out.update(xyz=xyz, mt=mt, k=s['k'], n=s['n'], cells=cinfo)
    if centrelines:
        out['centrelines'] = lines
    if stats:
        out['stats'] = window_stats(out)
    return out


def window_stats(r):
    """densities in the FOV: labels, centreline length inside the FOV and slab, z distribution"""
    W, H = r['fov_um']
    xyz = r['xyz']
    area = W * H
    st = dict(n_labels=int(len(xyz)), labels_per_um2=len(xyz) / area)
    if 'centrelines' in r:
        f = r['focus_nm']; zr = r['z_range_nm']
        Ltot = 0.0
        for m in r['centrelines']:
            mid = 0.5 * (m[1:] + m[:-1])
            seg = np.linalg.norm(np.diff(m, axis=0), axis=1)
            ok = (mid[:, 0] >= 0) & (mid[:, 0] < W * 1000) & (mid[:, 1] >= 0) & (mid[:, 1] < H * 1000)
            if zr is not None:
                ok &= np.abs(mid[:, 2] - f) < zr
            Ltot += seg[ok].sum() / 1000
        st.update(mt_length_um=Ltot, mt_length_per_um2=Ltot / area,
                  n_mt_in_fov=int(len(np.unique(r['mt'][r['mt'] >= 0]))))
    if len(xyz):
        st['z_nm_percentiles_5_25_50_75_95'] = [float(v) for v in np.percentile(xyz[:, 2], [5, 25, 50, 75, 95])]
    return st


def to_showcase_sites(r, px_nm=128.0, z0_nm=None):
    """labels in the format of analysis/fourpi/showcase4pi_movie.py make_sites: [N, 4] x px, y px, z um relative to
    z0_nm (default: the focus height), microtubule id"""
    z0 = r['focus_nm'] if z0_nm is None else z0_nm
    xyz = r['xyz']
    return np.c_[xyz[:, 0] / px_nm, xyz[:, 1] / px_nm, (xyz[:, 2] - z0) / 1000, r['mt']]
