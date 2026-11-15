"""Icon-based SIMCODE workflow sheet (inspiration material for the user's own Fig. 1 redesign).

Writes hand-built SVG (no matplotlib): every element is a <g id=...> of plain shapes, text stays <text>,
no clipping masks (stripes are analytically clipped polygons, mini-images are embedded PNGs).

    python analysis/paper/fig1_inspiration_icons.py [--cache out/paper_figs_v2/cache_fig1/fig1_data.npz]
        -> out/paper_figs_v2/fig1_inspiration/workflow_icons.{svg,pdf,png} + elements/<name>.svg

CPU only; reads only the cached mini-image data of fig_pipeline_v2_prep.py.
"""
import argparse
import base64
import io
import math
import os

import numpy as np

ROOT = os.path.expanduser('~/simcode2')
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"

# palette (consistent with fig_pipeline_v2 / si_4pi_pipeline)
C = dict(
    ink='#333333', grey='#8a8a8a', grey_l='#d4d4d4', grey_ll='#f1f1f1',
    blue='#3d7fc4', blue_m='#9ec5ef', blue_l='#dbe9f8', blue_d='#1f4e8c',
    orange='#d9822b', orange_m='#f2a65a', orange_l='#fbe6cf', orange_d='#a85f17',
    purple='#7a4ea3', purple_m='#c3a9df', purple_l='#efe7f7',
    green='#2e8b57', green_m='#7cc49a', green_l='#e1f2e7', green_d='#1b5e38',
    a1='#3b75af', a2='#c44e52', red='#c8102e',
)
SW, SW2 = 2.0, 1.4          # main / detail stroke widths


# ----------------------------------------------------------------------------------------------- primitives
def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def f(v):
    return f'{v:.2f}'.rstrip('0').rstrip('.')


def rect(x, y, w, h, fill='none', stroke='none', sw=SW, rx=0, extra=''):
    r = f' rx="{f(rx)}"' if rx else ''
    return (f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"{r} fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{f(sw)}" stroke-linejoin="round"{extra}/>')


def circle(x, y, r, fill='none', stroke='none', sw=SW, extra=''):
    return f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}"{extra}/>'


def ellipse(x, y, rx, ry, fill='none', stroke='none', sw=SW, rot=0):
    t = f' transform="rotate({f(rot)} {f(x)} {f(y)})"' if rot else ''
    return (f'<ellipse cx="{f(x)}" cy="{f(y)}" rx="{f(rx)}" ry="{f(ry)}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{f(sw)}"{t}/>')


def line(x0, y0, x1, y1, stroke=C['ink'], sw=SW, dash=None, cap='round'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{f(x0)}" y1="{f(y0)}" x2="{f(x1)}" y2="{f(y1)}" stroke="{stroke}" stroke-width="{f(sw)}" '
            f'stroke-linecap="{cap}"{d}/>')


def pts(p):
    return ' '.join(f'{f(x)},{f(y)}' for x, y in p)


def polyline(p, stroke=C['ink'], sw=SW, dash=None, fill='none'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<polyline points="{pts(p)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d}/>')


def polygon(p, fill='none', stroke='none', sw=SW):
    return (f'<polygon points="{pts(p)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" '
            f'stroke-linejoin="round"/>')


def path(d, fill='none', stroke=C['ink'], sw=SW, dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" stroke-linejoin="round" '
            f'stroke-linecap="round"{ds}/>')


def text(x, y, s, size=17, anchor='middle', color=C['ink'], weight='normal', italic=False):
    st = ' font-style="italic"' if italic else ''
    return (f'<text x="{f(x)}" y="{f(y)}" font-family="{FONT}" font-size="{f(size)}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}"{st}>{esc(s)}</text>')


def g(content, gid=None, tr=None):
    i = f' id="{gid}"' if gid else ''
    t = f' transform="{tr}"' if tr else ''
    body = ''.join(content) if isinstance(content, (list, tuple)) else content
    return f'<g{i}{t}>{body}</g>'


def head(x, y, ang, color, size=10):
    """filled arrow head with its tip at (x, y), pointing along ang (rad)."""
    w = 0.42 * size
    bx, by = x - size * math.cos(ang), y - size * math.sin(ang)
    nx, ny = -math.sin(ang), math.cos(ang)
    return polygon([(x, y), (bx + w * nx, by + w * ny), (bx - w * nx, by - w * ny)], fill=color)


def arrow(x0, y0, x1, y1, color=C['ink'], sw=SW, size=10, dash=None):
    ang = math.atan2(y1 - y0, x1 - x0)
    xe, ye = x1 - 0.8 * size * math.cos(ang), y1 - 0.8 * size * math.sin(ang)
    return g([line(x0, y0, xe, ye, color, sw, dash), head(x1, y1, ang, color, size)])


def carrow(x0, y0, cx, cy, x1, y1, color=C['ink'], sw=SW, size=10, dash=None):
    """quadratic-bezier arrow (control point cx, cy)."""
    ang = math.atan2(y1 - cy, x1 - cx)
    xe, ye = x1 - 0.8 * size * math.cos(ang), y1 - 0.8 * size * math.sin(ang)
    return g([path(f'M{f(x0)},{f(y0)} Q{f(cx)},{f(cy)} {f(xe)},{f(ye)}', stroke=color, sw=sw, dash=dash),
              head(x1, y1, ang, color, size)])


def png_b64(rgb, up=1):
    from PIL import Image
    a = np.clip(np.asarray(rgb) * 255 if np.asarray(rgb).dtype.kind == 'f' else rgb, 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    if up > 1:
        im = im.resize((im.width * up, im.height * up), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, format='PNG', optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def image(x, y, w, h, rgb, up=1, stroke=C['grey'], sw=SW2):
    s = (f'<image x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" preserveAspectRatio="none" '
         f'href="data:image/png;base64,{png_b64(rgb, up)}"/>')
    return s + (rect(x, y, w, h, stroke=stroke, sw=sw) if stroke else '')


def cmap(a, name='viridis', lo=None, hi=None):
    import matplotlib
    lo = np.min(a) if lo is None else lo
    hi = np.max(a) if hi is None else hi
    return matplotlib.colormaps[name](np.clip((a - lo) / (hi - lo + 1e-12), 0, 1))[..., :3]


# ----------------------------------------------------------------------------------------------- glyph builders
def _clip(poly, a, b, c):
    """Sutherland-Hodgman: keep a*x + b*y <= c."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if dp <= 0:
            out.append(p)
        if dp * dq < 0:
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def stripes(x, y, w, h, theta, pitch, phase=0.0, band=C['purple_m'], bg='white', stroke=C['ink'], sw=SW,
            duty=0.5, dots=None, dot_col=C['ink'], poly=None):
    """flat stripe tile: illuminated bands (analytically clipped polygons) + optional emitter dots whose size
    follows the local illumination. theta: direction of the pattern wave vector (deg)."""
    poly = poly or [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    cx, cy = x + w / 2, y + h / 2
    ct, st = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    off = phase / (2 * math.pi) * pitch
    R = math.hypot(w, h) / 2 + pitch
    out = [polygon(poly, fill=bg)]
    for n in range(-int(R / pitch) - 2, int(R / pitch) + 3):
        lo, hi = n * pitch + off, n * pitch + off + duty * pitch
        q = _clip(poly, -ct, -st, -(lo + ct * cx + st * cy))
        q = _clip(q, ct, st, hi + ct * cx + st * cy) if q else q
        if len(q) >= 3:
            out.append(polygon(q, fill=band))
    if dots:
        for (u, v) in dots:
            px, py = x + u * w, y + v * h
            uu = (px - cx) * ct + (py - cy) * st
            inten = 0.5 * (1 + math.cos(2 * math.pi * (uu - off - duty * pitch / 2) / pitch))
            rs = (min(w, h) / 64) ** 0.5
            out.append(circle(px, py, rs * (1.2 + 2.8 * inten), fill=dot_col, extra=f' fill-opacity="{f(0.3 + 0.7 * inten)}"'))
    if stroke:
        out.append(polygon(poly, stroke=stroke, sw=sw))
    return g(out)


DOTS = [(0.22, 0.28), (0.7, 0.2), (0.45, 0.55), (0.18, 0.78), (0.8, 0.7), (0.58, 0.86)]


def shade(hexcol, k):
    r, gg, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    if k >= 0:
        r, gg, b = (int(v + (255 - v) * k) for v in (r, gg, b))
    else:
        r, gg, b = (int(v * (1 + k)) for v in (r, gg, b))
    return f'#{r:02x}{gg:02x}{b:02x}'


def cube(x, y, s, col, face_content=None, d=None, stroke=None, sw=SW):
    """isometric cube, bounding box (x, y, s + d, s + d). col: front-face colour."""
    d = 0.38 * s if d is None else d
    stroke = stroke or shade(col, -0.45)
    fy = y + d
    out = [polygon([(x, fy), (x + d, y), (x + s + d, y), (x + s, fy)], fill=shade(col, 0.4), stroke=stroke, sw=sw),
           polygon([(x + s, fy), (x + s + d, y), (x + s + d, y + s), (x + s, fy + s)], fill=shade(col, -0.18),
                   stroke=stroke, sw=sw),
           rect(x, fy, s, s, fill=col, stroke=stroke, sw=sw)]
    if face_content:
        out.append(face_content(x, fy, s))
        out.append(rect(x, fy, s, s, stroke=stroke, sw=sw))
    return g(out)


def sine_pts(x0, x1, yc, amp, period, phase, n=80):
    return [(x0 + (x1 - x0) * t, yc - amp * math.sin(2 * math.pi * (x1 - x0) * t / period - phase))
            for t in np.linspace(0, 1, n)]


def bell_pts(x0, x1, xc, yb, height, sig, n=60):
    return [(x0 + (x1 - x0) * t, yb - height * math.exp(-0.5 * ((x0 + (x1 - x0) * t - xc) / sig) ** 2))
            for t in np.linspace(0, 1, n)]


def crosshair(x, y, r, col):
    return g([circle(x, y, r, stroke=col, sw=SW), line(x - 1.5 * r, y, x - 0.45 * r, y, col, SW),
              line(x + 0.45 * r, y, x + 1.5 * r, y, col, SW), line(x, y - 1.5 * r, x, y - 0.45 * r, col, SW),
              line(x, y + 0.45 * r, x, y + 1.5 * r, col, SW), circle(x, y, 2.6, fill=col)])


# ----------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, path):
        D = np.load(path)
        self.D = D
        k = int(D['pick'])
        ex, ey = int(round(float(D['c_x'][k]))), int(round(float(D['c_y'][k])))
        win = D['win']
        c0x, c0y = np.clip(ex - 16, 0, 120 - 32), np.clip(ey - 16, 0, 120 - 32)
        self.win = win[:, c0y:c0y + 32, c0x:c0x + 32]
        lo, hi = np.median(self.win), 0.75 * self.win.max()
        self.win_rgb = [cmap(w, 'viridis', lo, hi) for w in self.win]
        pm = D['pmap'][c0y:c0y + 32, c0x:c0x + 32]
        self.pmap_rgb = cmap(pm, 'viridis', 0, 1)
        sf = D['sim_frames'][1]
        self.sim_rgb = [cmap(w, 'viridis', np.median(sf), np.percentile(sf, 99.7)) for w in sf]
        si = D['sim_illum'][1]
        self.illum_rgb = [cmap(w, 'viridis', 0, 2) for w in si]
        r = D['render_I']
        n = r.shape[0]
        a = r[n // 2 - 260:n // 2 + 260, n // 2 - 260:n // 2 + 260]
        # 2x2 box downsample to keep the file light
        a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]
        self.render = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, 3).mean((1, 3))
        self.dphi = D['pat_dphi']            # (2, nknots) rad
        self.I = D['c_I'][k]
        self.sI = D['c_sI'][k]
        self.bg = D['c_bg'][k]


# ----------------------------------------------------------------------------------------------- elements
# each returns (w, h, svg-body) drawn in [0, w] x [0, h]

def el_movie_stack(Dt):
    out = []
    for k in range(7, 0, -1):
        out.append(rect(6 + 5 * k, 40 - 5 * k, 64, 64, fill='white', stroke=C['grey'], sw=SW2))
    out.append(stripes(6, 40, 64, 64, 0, 16, 0, dots=DOTS))
    out.append(arrow(80, 110, 108, 82, C['grey'], SW2, 9))
    out.append(text(104, 110, 't', 15, 'middle', C['grey'], italic=True))
    return 112, 114, g(out)


def el_movie_film(Dt):
    out = [rect(0, 22, 110, 66, fill=C['ink'], rx=3)]
    for i in range(9):
        out.append(rect(4 + i * 12, 26, 6, 5, fill='white', rx=1))
        out.append(rect(4 + i * 12, 79, 6, 5, fill='white', rx=1))
    for i, (th, ph) in enumerate([(0, 0), (0, 2.1), (90, 0)]):
        out.append(stripes(5 + i * 35, 36, 30, 38, th, 12, ph, stroke=None, dots=DOTS[:4]))
    return 110, 110, g(out)


def el_movie_real(Dt):
    out = []
    for k in range(5, -1, -1):
        out.append(image(8 + 6 * k, 38 - 6 * k, 62, 62, Dt.win_rgb[k], up=4))
    out.append(arrow(78, 108, 106, 80, C['grey'], SW2, 9))
    out.append(text(102, 108, 't', 15, 'middle', C['grey'], italic=True))
    return 110, 110, g(out)


def _win_tile(x, y, s, t, sw=SW):
    th = 0 if t < 3 else 90
    return stripes(x, y, s, s, th, 12, 2 * math.pi * (t % 3) / 3, stroke=C['a1'] if t < 3 else C['a2'], sw=sw,
                   dots=DOTS[:5])


def el_window_grid(Dt):
    out = []
    for t in range(6):
        out.append(_win_tile(4 + (t % 3) * 38, 14 + (t // 3) * 44, 34, t, SW + 0.6))
    out.append(arrow(6, 104, 112, 104, C['grey'], SW2, 8))      # step axis
    return 120, 110, g(out)


def el_window_row(Dt):
    out = []
    for t in range(6):
        out.append(_win_tile(1 + t * 18.2, 40, 16, t, SW2))
        out.append(rect(1 + t * 18.2, 62, 16, 5, fill=C['a1'] if t < 3 else C['a2']))
    out.append(path('M3,34 Q3,26 12,26 L46,26 Q55,26 55,18 Q55,26 64,26 L98,26 Q107,26 107,34', stroke=C['ink'],
                    sw=SW2))
    out.append(text(55, 13, '6', 16, 'middle', C['ink'], 'bold'))
    return 110, 110, g(out)


def el_window_slide(Dt):
    out = []
    for t in range(15):
        a1 = (t % 6) < 3
        out.append(rect(1 + t * 7.2, 50, 6, 22, fill=C['a1'] if a1 else C['a2']))
    out.append(rect(-1, 44, 6 * 7.2 + 1.5, 34, stroke=C['ink'], sw=SW, rx=3))
    out.append(rect(3 * 7.2 - 1, 38, 6 * 7.2 + 1.5, 46, stroke=C['grey'], sw=SW2, rx=3, extra=' stroke-dasharray="4 3"'))
    out.append(arrow(2, 94, 108, 94, C['grey'], SW2, 8))
    out.append(text(104, 110, 't', 14, 'middle', C['grey'], italic=True))
    return 110, 110, g(out)


def dice(x, y, s, col=C['ink'], fill='white', pips=5):
    out = [rect(x, y, s, s, fill=fill, stroke=col, sw=SW, rx=s * 0.18)]
    q = s / 4
    pos = {5: [(1, 1), (3, 1), (2, 2), (1, 3), (3, 3)], 3: [(1, 1), (2, 2), (3, 3)], 4: [(1, 1), (3, 1), (1, 3), (3, 3)]}
    for a, b in pos[pips]:
        out.append(circle(x + a * q, y + b * q, s * 0.085, fill=col))
    return g(out)


def el_sim_dice(Dt):
    out = [stripes(40, 6, 52, 52, 25, 13, 0.5, band=C['purple_m']),
           stripes(52, 26, 52, 52, -50, 9, 1.0, band=C['purple_m']),
           dice(4, 52, 46, C['ink'], 'white', 5)]
    return 110, 110, g(out)


def el_sim_fan(Dt):
    out = []
    for i, (th, p, rot) in enumerate([(70, 15, -14), (-35, 10, 0), (15, 7, 14)]):
        cx, cy = 55, 64
        out.append(g(stripes(30, 34, 50, 50, th, p, i, band=C['purple_m'], dots=DOTS[:4]),
                     tr=f'rotate({rot} {cx} {cy + 30})'))
    return 110, 110, g(out)


def el_sim_real(Dt):
    out = []
    for k in range(5, -1, -1):
        out.append(image(6 + 6 * k, 40 - 6 * k, 62, 62, Dt.sim_rgb[k], up=3))
    out.append(dice(72, 70, 32, C['ink'], 'white', 3))
    return 110, 110, g(out)


def el_cube_det(Dt):
    return 92, 92, cube(4, 4, 62, C['blue_m'])


def el_cube_em(Dt):
    return 92, 92, cube(4, 4, 62, C['orange_m'])


def el_unet(Dt):
    hs = [70, 50, 32, 50, 70]
    out = []
    for i, h in enumerate(hs):
        x = 6 + i * 21
        out.append(rect(x, 55 - h / 2, 13, h, fill=C['blue_m'], stroke=C['blue'], sw=SW2, rx=2))
    out.append(path('M12.5,18 Q55,-6 97.5,18', stroke=C['grey'], sw=SW2, dash='4 3'))
    out.append(path('M33.5,28 Q55,14 76.5,28', stroke=C['grey'], sw=SW2, dash='4 3'))
    return 110, 110, g(out)


def el_nodes(Dt):
    layers = [3, 4, 4, 2]
    xs = [12, 42, 72, 100]
    P = [[(xs[i], 55 + (j - (n - 1) / 2) * 22) for j in range(n)] for i, n in enumerate(layers)]
    out = []
    for i in range(len(P) - 1):
        for a in P[i]:
            for b in P[i + 1]:
                out.append(line(a[0], a[1], b[0], b[1], C['grey_l'], 1.2))
    for i, L in enumerate(P):
        for (x, y) in L:
            out.append(circle(x, y, 7, fill=C['blue_m'] if i < 3 else C['orange_m'], stroke=C['blue'] if i < 3 else
                              C['orange'], sw=SW2))
    return 110, 110, g(out)


def el_train_loop(Dt):
    out = [cube(30, 10, 36, C['blue_m']), cube(30, 60, 36, C['orange_m'])]
    out.append(carrow(98, 92, 122, 56, 98, 18, C['red'], SW, 10))
    out.append(carrow(14, 18, -10, 56, 14, 92, C['red'], SW, 10))
    return 112, 110, g(out, tr='translate(0,0)')


def el_train_loss(Dt):
    out = [polyline([(14, 8), (14, 96), (104, 96)], C['ink'], SW2)]
    p = [(16 + t * 86, 92 - 74 * math.exp(-4 * t) - 3 * math.sin(30 * t) * math.exp(-2 * t)) for t in
         np.linspace(0, 1, 70)]
    out.append(polyline(p, C['red'], SW))
    out.append(circle(p[8][0] + 2, p[8][1] - 6, 6, fill=C['blue_m'], stroke=C['blue'], sw=SW2))
    return 110, 110, g(out)


def el_train_target(Dt):
    out = []
    for r, c in [(40, 'white'), (29, C['red']), (19, 'white'), (9, C['red'])]:
        out.append(circle(50, 58, r, fill=c, stroke=C['red'], sw=SW2))
    out.append(arrow(100, 10, 54, 55, C['ink'], SW, 11))
    out.append(polygon([(100, 10), (108, 4), (106, 14)], fill=C['blue_m'], stroke=C['blue'], sw=1.0))
    return 110, 110, g(out)


def el_pmap_real(Dt):
    return 92, 92, image(4, 4, 84, 84, Dt.pmap_rgb, up=4, stroke=C['grey'])


def el_pmap_flat(Dt):
    out = [rect(4, 4, 102, 102, fill='#2a1a4a', rx=2)]
    for (u, v, s) in [(0.25, 0.3, 1.0), (0.7, 0.25, 0.8), (0.55, 0.65, 1.1), (0.22, 0.75, 0.7)]:
        x, y = 4 + u * 102, 4 + v * 102
        for r, c in [(13, '#3b528b'), (9, '#21918c'), (5.5, '#5ec962'), (2.6, '#fde725')]:
            out.append(circle(x, y, r * s, fill=c))
        out.append(circle(x, y, 15 * s + 2, stroke=C['orange_m'], sw=SW2))
    return 110, 110, g(out)


def el_pmap_grid(Dt):
    rng = np.random.default_rng(3)
    n, s = 6, 16
    P = np.exp(-(((np.arange(n)[:, None] - 2.3) ** 2 + (np.arange(n)[None] - 3.6) ** 2) / 1.6))
    P = np.clip(P + 0.06 * rng.random((n, n)), 0, 1)
    import matplotlib
    cm = matplotlib.colormaps['viridis']
    out = []
    for i in range(n):
        for j in range(n):
            r, gg, b, _ = cm(P[i, j])
            out.append(rect(7 + j * s, 7 + i * s, s, s, fill=f'#{int(r * 255):02x}{int(gg * 255):02x}{int(b * 255):02x}',
                            stroke='white', sw=1.0))
    ii, jj = np.unravel_index(np.argmax(P), P.shape)
    out.append(rect(7 + jj * s, 7 + ii * s, s, s, stroke=C['red'], sw=SW + 0.5))
    return 110, 110, g(out)


def _bars(x0, yb, w, hmax, vals, errs=None, cols=None, gap=0.28, sw=SW2):
    n = len(vals)
    bw = w / n
    vm = max(v + (e if errs is not None else 0) for v, e in zip(vals, errs if errs is not None else [0] * n))
    out = []
    for i, v in enumerate(vals):
        h = hmax * v / vm
        bx = x0 + i * bw + gap * bw / 2
        out.append(rect(bx, yb - h, bw * (1 - gap), h, fill=cols[i] if cols else C['grey']))
        if errs is not None:
            e = hmax * errs[i] / vm
            xc = bx + bw * (1 - gap) / 2
            out.append(line(xc, yb - h - e, xc, yb - h + e, C['ink'], sw * 0.9))
            out.append(line(xc - 3, yb - h - e, xc + 3, yb - h - e, C['ink'], sw * 0.9))
            out.append(line(xc - 3, yb - h + e, xc + 3, yb - h + e, C['ink'], sw * 0.9))
    out.append(line(x0 - 2, yb, x0 + w + 2, yb, C['ink'], sw))
    return g(out)


ANG = [C['a1']] * 3 + [C['a2']] * 3


def el_int_bars(Dt):
    return 110, 110, _bars(8, 96, 94, 80, list(Dt.I), list(Dt.sI), ANG)


def el_int_sine(Dt):
    out = [polyline(sine_pts(6, 104, 58, 30, 98, 0.0), C['grey_l'], SW)]
    for t in range(6):
        ph = 2 * math.pi * ((t % 3) / 3 + (0.15 if t >= 3 else 0.0))
        x = 6 + 98 * ((t % 3) / 3 + (0.15 if t >= 3 else 0.0) + 0.08)
        y = 58 - 30 * math.sin(2 * math.pi * (x - 6) / 98) + (4 if t % 2 else -3)
        out.append(line(x, y - 11, x, y + 11, C['ink'], SW2))
        out.append(circle(x, y, 5, fill=ANG[t]))
    return 110, 110, g(out)


def el_int_bgbars(Dt):
    return 110, 110, g([_bars(8, 60, 94, 52, list(Dt.I), list(Dt.sI), ANG),
                        _bars(8, 102, 94, 24, list(Dt.bg), None, [C['grey']] * 6)])


def el_bg_bars(Dt):
    return 110, 70, _bars(8, 62, 94, 40, list(Dt.bg), None, ['#9a9a9a'] * 6)


def el_pos_ellipse(Dt):
    out = [ellipse(42, 55, 32, 24, fill=C['blue_l'], stroke=C['blue'], sw=SW), circle(42, 55, 4, fill=C['blue_d']),
           line(96, 28, 96, 82, C['blue'], SW), line(90, 28, 102, 28, C['blue'], SW),
           line(90, 82, 102, 82, C['blue'], SW), circle(96, 55, 4, fill=C['blue_d'])]
    return 110, 110, g(out)


def el_pos_axes(Dt):
    o = (40, 72)
    out = [arrow(*o, 104, 72, C['ink'], SW2, 8), arrow(*o, 40, 6, C['ink'], SW2, 8),
           arrow(*o, 10, 102, C['ink'], SW2, 8),
           text(104, 92, 'x', 14, 'middle', C['grey'], italic=True), text(52, 14, 'z', 14, 'middle', C['grey'], italic=True),
           text(4, 92, 'y', 14, 'middle', C['grey'], italic=True),
           ellipse(70, 44, 16, 10, fill=C['blue_l'], stroke=C['blue'], sw=SW2, rot=-15),
           line(70, 44, 70, 72, C['grey'], 1.2, '3 3'), circle(70, 44, 3.5, fill=C['blue_d'])]
    return 110, 110, g(out)


def el_pos_blob(Dt):
    out = []
    for r, c in [(36, '#eef4fb'), (27, '#d2e3f5'), (18, '#a9c9ec'), (9, '#6fa4dc')]:
        out.append(circle(55, 55, r, fill=c))
    out.append(crosshair(55, 55, 14, C['blue_d']))
    return 110, 110, g(out)


def el_pattern_est(Dt):
    rng = np.random.default_rng(7)
    out = [stripes(2, 6, 44, 44, 0, 11, 0, band=C['purple_m'])]
    # intensity vs phase
    x0, x1, yc = 54, 136, 30
    xs = rng.uniform(x0, x1, 46)
    ys = yc - 15 * np.sin(2 * math.pi * (xs - x0) / (x1 - x0)) + rng.normal(0, 5, xs.size)
    for x, y in zip(xs, ys):
        out.append(circle(x, y, 1.9, fill=C['a1'], extra=' fill-opacity="0.55"'))
    out.append(polyline(sine_pts(x0, x1, yc, 15, x1 - x0, 0), C['ink'], SW))
    out.append(polyline([(x0, 4), (x0, 52), (x1, 52)], C['ink'], SW2))
    # drift trace (real eps_i(t))
    dp = Dt.dphi
    t = np.linspace(2, 136, dp.shape[1])
    for i, (col, yb) in enumerate([(C['a1'], 70), (C['a2'], 88)]):
        yv = yb - np.degrees(dp[i] - dp[i].mean()) * 0.55
        sel = slice(None, None, 4)
        out.append(polyline(list(zip(t[sel], yv[sel])), col, 1.4))
    return 140, 100, g(out)


def el_pattern_dial(Dt):
    out = [circle(55, 55, 42, fill=C['purple_l'], stroke=C['purple'], sw=SW)]
    for j, a in enumerate([90, -30, 210]):
        r = math.radians(a)
        out.append(arrow(55, 55, 55 + 36 * math.cos(r), 55 - 36 * math.sin(r), C['purple'], SW, 10))
    out.append(circle(55, 55, 4, fill=C['purple']))
    return 110, 110, g(out)


def el_pattern_drift(Dt):
    dp = Dt.dphi
    t = np.linspace(8, 104, dp.shape[1])
    out = [polyline([(6, 6), (6, 100), (106, 100)], C['ink'], SW2)]
    for i, (col, yb) in enumerate([(C['a1'], 38), (C['a2'], 72)]):
        yv = yb - np.degrees(dp[i] - dp[i].mean()) * 0.9
        sel = slice(None, None, 4)
        out.append(polyline(list(zip(t[sel], yv[sel])), col, 1.6))
    return 110, 110, g(out)


def _stripe_face(theta):
    return lambda x, y, s: stripes(x, y, s, s, theta, s / 3.2, 0, band='white', bg='none', stroke=None)


def el_pass2_cubes(Dt):
    out = [cube(4, 2, 46, C['blue'], _stripe_face(0)), cube(4, 70, 46, C['orange'], _stripe_face(90))]
    out.append(arrow(31, 66, 31, 84, C['ink'], SW2, 8))
    return 72, 136, g(out)


def el_pass2_plug(Dt):
    out = [stripes(2, 34, 34, 34, 0, 9, 0, band=C['purple_m']), arrow(40, 51, 52, 51, C['ink'], SW, 9),
           cube(54, 22, 40, C['blue_m'])]
    return 110, 110, g(out)


def el_pass2_badge(Dt):
    out = [cube(8, 16, 62, C['blue'], _stripe_face(0)), circle(88, 22, 15, fill=C['red']),
           text(88, 28.5, '2', 19, 'middle', 'white', 'bold')]
    return 110, 110, g(out)


GREENS = ['#1b5e38', '#2e8b57', '#7cc49a']


def el_fit(Dt):
    out = []
    x0, x1, yc, amp, per = 4, 108, 34, 22, 72
    for j in range(3):
        out.append(polyline(sine_pts(x0, x1, yc, amp, per, 2 * math.pi * j / 3), GREENS[j], SW))
    xs = 62
    out.append(line(xs, 4, xs, 100, C['ink'], SW2, '4 3'))
    for j in range(3):
        y = yc - amp * math.sin(2 * math.pi * (xs - x0) / per - 2 * math.pi * j / 3)
        out.append(circle(xs, y, 4.2, fill=C['ink']))
    bp = bell_pts(x0, x1, xs + 2, 100, 30, 13)
    out.append(polygon(bp + [(x1, 100), (x0, 100)], fill=C['blue_l']))
    out.append(polyline(bp, C['blue'], SW))
    out.append(arrow(114, 56, 132, 56, C['ink'], SW, 9))
    out.append(crosshair(152, 56, 13, C['green']))
    return 176, 104, g(out)


def el_fit_cost(Dt):
    xs = np.linspace(6, 104, 90)
    pr = 0.012 * (xs - 58) ** 2
    cost = 30 + 18 * np.cos(2 * math.pi * (xs - 58) / 30) + pr
    yb = 92
    out = [polyline(list(zip(xs, yb - (90 - (cost + 0) * 0.9) + 2)), C['green'], SW)]
    out.append(polyline(list(zip(xs, yb - (90 - (pr + 12) * 0.9) + 2)), C['blue'], SW2, '5 4'))
    yy = lambda x: yb - (90 - (30 + 18 * math.cos(2 * math.pi * (x - 58) / 30) + 0.012 * (x - 58) ** 2) * 0.9) + 2
    for x, c in [(28, C['grey']), (58, C['ink']), (88, C['grey'])]:
        out.append(polygon([(x, 98), (x - 5, 106), (x + 5, 106)], fill=c))
    out.append(circle(43, yy(43), 4.5, fill=C['green_d']))
    return 110, 110, g(out)


def el_fit_target(Dt):
    out = [ellipse(55, 55, 34, 24, fill=C['green_l'], stroke=C['green_m'], sw=SW2, rot=-20),
           ellipse(55, 55, 18, 12, fill=C['green_m'], stroke='none', rot=-20),
           crosshair(55, 55, 14, C['green_d'])]
    return 110, 110, g(out)


def el_funnel(Dt):
    rng = np.random.default_rng(1)
    out = [polygon([(10, 34), (100, 34), (64, 74), (64, 98), (46, 104), (46, 74)], fill=C['grey_ll'],
                   stroke=C['ink'], sw=SW)]
    for _ in range(11):
        x, y = rng.uniform(16, 94), rng.uniform(6, 28)
        out.append(circle(x, y, 3.2, fill=C['grey'] if rng.random() < 0.5 else C['green']))
    out.append(circle(55, 116, 3.6, fill=C['green']))
    out.append(circle(55, 128, 3.6, fill=C['green']))
    return 110, 134, g(out)


def el_sieve(Dt):
    out = [path('M8,40 Q55,104 102,40 Z', fill=C['grey_ll'], stroke=C['ink'], sw=SW),
           line(102, 40, 110, 32, C['ink'], SW)]
    for (x, y) in [(36, 60), (55, 66), (74, 60), (46, 74), (64, 74)]:
        out.append(circle(x, y, 2.6, fill='white', stroke=C['ink'], sw=1.0))
    for (x, y) in [(30, 32), (48, 26), (66, 30), (80, 34), (40, 42), (58, 40), (72, 44)]:
        out.append(circle(x, y, 4.2, fill=C['grey']))
    for y in [88, 102]:
        out.append(circle(55, y, 3.2, fill=C['green']))
    return 110, 110, g(out)


def el_select_scatter(Dt):
    rng = np.random.default_rng(5)
    out = [polyline([(8, 6), (8, 102), (106, 102)], C['ink'], SW2),
           line(46, 6, 46, 102, C['red'], SW2, '5 4'), line(8, 56, 106, 56, C['red'], SW2, '5 4')]
    for _ in range(28):
        x, y = rng.uniform(14, 102), rng.uniform(10, 98)
        if x > 46 and y > 56:
            out.append(circle(x, y, 3.4, fill=C['green']))
        else:
            out.append(line(x - 3, y - 3, x + 3, y + 3, C['grey'], 1.6))
            out.append(line(x - 3, y + 3, x + 3, y - 3, C['grey'], 1.6))
    return 110, 110, g(out)


def el_render_real(Dt):
    return 110, 110, image(2, 2, 106, 106, Dt.render, stroke=C['ink'], sw=SW2)


def el_render_vector(Dt):
    rng = np.random.default_rng(11)
    out = [rect(2, 2, 106, 106, fill='#0b0b0b')]
    curves = [((4, 30), (50, 0), (106, 70)), ((10, 104), (40, 40), (104, 20)), ((30, 4), (70, 60), (60, 106))]
    for (p0, p1, p2) in curves:
        for t in np.linspace(0, 1, 36):
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0] + rng.normal(0, 1.2)
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1] + rng.normal(0, 1.2)
            if 4 < x < 106 and 4 < y < 106:
                out.append(circle(x, y, rng.uniform(1.0, 2.2), fill=rng.choice(['#fca50a', '#f1ef75', '#e35933'])))
    return 110, 110, g(out)


def el_render_lens(Dt):
    out = [image(2, 2, 84, 84, Dt.render, stroke=C['ink'], sw=SW2),
           circle(62, 62, 26, fill='white', stroke=C['ink'], sw=SW + 0.5, extra=' fill-opacity="0.15"'),
           line(81, 81, 104, 104, C['ink'], SW + 3)]
    return 110, 110, g(out)


def el_arrow_straight(Dt):
    return 110, 40, arrow(4, 20, 106, 20, C['ink'], SW, 11)


def el_arrow_curved(Dt):
    return 110, 60, carrow(4, 50, 55, -10, 106, 50, C['ink'], SW, 11)


def el_arrow_dashed(Dt):
    return 110, 40, arrow(4, 20, 106, 20, C['grey'], SW2, 9, dash='5 4')


def el_palette(Dt):
    cols = [C['blue_m'], C['blue'], C['orange_m'], C['purple_m'], C['purple'], C['green'], C['a1'], C['a2'], C['red'],
            C['grey']]
    out = []
    for i, c in enumerate(cols):
        out.append(rect(4 + (i % 5) * 21, 30 + (i // 5) * 26, 18, 20, fill=c, rx=3))
    return 110, 110, g(out)


# name -> (builder, label, short description used only as the gallery variant tag)
GALLERY = [
    ('movie', [el_movie_stack, el_movie_film, el_movie_real]),
    ('6 frames', [el_window_grid, el_window_row, el_window_slide]),
    ('simulate', [el_sim_dice, el_sim_fan, el_sim_real]),
    ('network', [el_cube_det, el_unet, el_nodes]),
    ('train', [el_train_loop, el_train_loss, el_train_target]),
    ('detect', [el_pmap_real, el_pmap_flat, el_pmap_grid]),
    ('intensities', [el_int_bars, el_int_sine, el_int_bgbars]),
    ('position', [el_pos_ellipse, el_pos_axes, el_pos_blob]),
    ('pattern', [el_pattern_est, el_pattern_dial, el_pattern_drift]),
    ('pass 2', [el_pass2_cubes, el_pass2_plug, el_pass2_badge]),
    ('fit', [el_fit, el_fit_cost, el_fit_target]),
    ('select', [el_funnel, el_sieve, el_select_scatter]),
    ('render', [el_render_real, el_render_vector, el_render_lens]),
    ('arrows', [el_arrow_straight, el_arrow_curved, el_arrow_dashed]),
    ('palette', [el_palette]),
]


def fname(fn):
    return fn.__name__[3:]


def svg_doc(w, h, body, bg=True):
    b = rect(0, 0, w, h, fill='white') if bg else ''
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{f(w)}" height="{f(h)}" viewBox="0 0 {f(w)} {f(h)}">'
            f'{b}{body}</svg>\n')


# ----------------------------------------------------------------------------------------------- sheet
def place(fn, Dt, x, y, gid=None):
    w, h, body = fn(Dt)
    return w, h, g(body, gid or fname(fn), f'translate({f(x)},{f(y)})')


def backplate(x, y, w, h, col, label=None, lcol=None):
    out = [rect(x, y, w, h, fill=col, rx=14)]
    if label:
        out.append(text(x + 12, y + 22, label, 16, 'start', lcol or C['ink'], 'bold'))
    return g(out)


def build_sheet(Dt):
    W = 2080
    out = [text(30, 46, 'SIMCODE', 30, 'start', C['ink'], 'bold'),
           text(200, 46, 'workflow icons', 22, 'start', C['grey'])]

    # ---------------- workflow row (inference), icons centred on yc
    yc = 420
    seq = [('movie', el_movie_stack, 'movie'), ('window', el_window_grid, '6 frames'),
           ('det1', el_cube_det, 'detect'), ('pmap', el_pmap_real, 'p'), ('em1', el_cube_em, 'emitter'),
           ('outputs', None, 'outputs'), ('pattern', el_pattern_est, 'pattern'),
           ('pass2', el_pass2_cubes, 'pass 2'), ('fit', el_fit, 'fit'), ('select', el_funnel, 'select'),
           ('render', el_render_real, 'SIMCODE')]
    gap = 46
    x = 40
    boxes = {}
    flow = []
    for key, fn, lab in seq:
        if key == 'outputs':
            w, h = 250, 116
            sub = [rect(0, 0, w, h, fill='white', stroke=C['grey_l'], sw=SW2, rx=10)]
            _, _, b1 = el_int_bars(Dt)
            sub.append(g(b1, 'intensities', 'translate(6,4) scale(0.95)'))
            _, _, b2 = el_bg_bars(Dt)
            sub.append(g(b2, 'backgrounds', 'translate(112,34) scale(0.62)'))
            _, _, b3 = el_pos_ellipse(Dt)
            sub.append(g(b3, 'position', 'translate(176,24) scale(0.66)'))
            sub.append(text(118, 96, 'bg', 14, 'start', C['grey']))
            sub.append(text(206, 100, 'x y z', 14, 'middle', C['grey'], italic=True))
            flow.append(g(sub, 'outputs', f'translate({f(x)},{f(yc - h / 2)})'))
        else:
            w, h, body = fn(Dt)
            flow.append(g(body, key, f'translate({f(x)},{f(yc - h / 2)})'))
        boxes[key] = (x, yc - h / 2, w, h)
        flow.append(text(x + w / 2, yc + 94, lab, 20, 'middle', C['ink'] if key != 'render' else C['ink'],
                         'bold' if key == 'render' else 'normal'))
        x += w + gap
    total = x - gap

    # backplates (drawn first)
    def span(a, b, pad=16):
        x0 = boxes[a][0] - pad
        x1 = boxes[b][0] + boxes[b][2] + pad
        return x0, x1
    plates = []
    for a, b, col, lab, lcol in [('det1', 'outputs', C['blue_l'], 'pass 1', C['blue']),
                                 ('pattern', 'pattern', C['purple_l'], None, None),
                                 ('pass2', 'pass2', C['blue_l'], None, None),
                                 ('fit', 'fit', C['green_l'], None, None)]:
        x0, x1 = span(a, b)
        plates.append(backplate(x0, yc - 110, x1 - x0, 230, col, lab, lcol))
    out.append(g(plates, 'stage_backplates'))
    out.append(g(flow, 'workflow'))

    # flow arrows
    arr = []
    order = [k for k, _, _ in seq]
    for a, b in zip(order[:-1], order[1:]):
        xa = boxes[a][0] + boxes[a][2] + 6
        xb = boxes[b][0] - 6
        arr.append(arrow(xa, yc, xb, yc, C['ink'], SW, 11))
    # window -> pass 2 (dashed, below the row)
    wx = boxes['window'][0] + boxes['window'][2] / 2
    px = boxes['pass2'][0] + boxes['pass2'][2] / 2
    yl = yc + 140
    arr.append(polyline([(wx, yc + 108), (wx, yl), (px, yl)], C['grey'], SW2, '6 4'))
    arr.append(arrow(px, yl, px, yc + 122, C['grey'], SW2, 9))
    # z from pass 1 to select
    ox = boxes['outputs'][0] + 206
    sx = boxes['select'][0] + boxes['select'][2] / 2
    yz = yc + 166
    arr.append(polyline([(ox, yc + 70), (ox, yz), (sx, yz)], C['grey'], SW2, '6 4'))
    arr.append(arrow(sx, yz, sx, yc + 122, C['grey'], SW2, 9))
    arr.append(text((ox + sx) / 2, yz + 20, 'z', 15, 'middle', C['grey'], italic=True))
    out.append(g(arr, 'workflow_arrows'))

    # ---------------- training lane (top): simulate -> sim frames -> train; weights down into both passes
    ty = 170
    tr = []
    tx = boxes['window'][0] + 30
    _, _, b = el_sim_dice(Dt)
    tr.append(g(b, 'simulate', f'translate({f(tx)},{f(ty - 55)})'))
    tr.append(text(tx + 55, ty + 80, 'simulate', 20))
    tr.append(arrow(tx + 120, ty, tx + 166, ty, C['ink'], SW, 11))
    _, _, b = el_sim_real(Dt)
    tr.append(g(b, 'sim_frames', f'translate({f(tx + 176)},{f(ty - 55)})'))
    tr.append(text(tx + 231, ty + 80, 'sim data', 20))
    tr.append(arrow(tx + 296, ty, tx + 342, ty, C['ink'], SW, 11))
    _, _, b = el_train_loop(Dt)
    tlx = tx + 352
    tr.append(g(b, 'train', f'translate({f(tlx)},{f(ty - 55)})'))
    tr.append(text(tlx + 56, ty + 80, 'train', 20))
    tplate = backplate(tx - 20, ty - 92, tlx + 112 + 20 - (tx - 20), 196, C['grey_ll'], 'training', C['grey'])
    # weights: dashed down to pass-1 networks and pass 2
    wsrc = tlx + 132
    tr.append(polyline([(wsrc, ty), (boxes['pass2'][0] + 36, ty), ], C['grey'], SW2, '6 4'))
    tr.append(arrow(boxes['pass2'][0] + 36, ty, boxes['pass2'][0] + 36, yc - 122, C['grey'], SW2, 9))
    # and into pass 1 (between the two pass-1 networks)
    p1x = (boxes['det1'][0] + boxes['em1'][0] + boxes['em1'][2]) / 2
    tr.append(polyline([(tlx + 56, ty + 104), (tlx + 56, ty + 120), (p1x, ty + 120)], C['grey'], SW2, '6 4'))
    tr.append(arrow(p1x, ty + 120, p1x, yc - 112, C['grey'], SW2, 9))
    out.insert(0, g(tplate, 'training_backplate'))
    out.append(g(tr, 'training'))

    # ---------------- gallery
    gy0 = yc + 240
    out.append(line(30, gy0, W - 30, gy0, C['grey_l'], SW2))
    out.append(text(30, gy0 + 42, 'element gallery', 22, 'start', C['grey']))
    cw, ch, vs = 380, 200, 120
    gal = []
    for i, (name, fns) in enumerate(GALLERY):
        r, c = divmod(i, 5)
        cx0 = 40 + c * (cw + 26)
        cy0 = gy0 + 70 + r * (ch + 20)
        cell = [rect(cx0, cy0, cw, ch, fill='white', stroke=C['grey_l'], sw=1.2, rx=12),
                text(cx0 + 16, cy0 + 26, name, 17, 'start', C['ink'], 'bold')]
        for k, fn in enumerate(fns):
            w, h, body = fn(Dt)
            s = min(1.0, 110 / max(w, h)) if name != 'arrows' else 1.0
            vx = cx0 + 14 + k * vs + (110 - w * s) / 2
            vy = cy0 + 40 + (140 - h * s) / 2
            tf = f'translate({f(vx)},{f(vy)})' + (f' scale({f(s)})' if s != 1 else '')
            cell.append(g(body, f'gallery_{fname(fn)}', tf))
            cell.append(text(cx0 + 14 + k * vs + 55, cy0 + ch - 8, 'abc'[k], 13, 'middle', C['grey']))
        gal.append(g(cell, f'gallery_{name.replace(" ", "")}'))
    out.append(g(gal, 'gallery'))
    H = gy0 + 70 + 3 * (ch + 20) + 20
    return W, H, out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(ROOT, 'out/paper_figs_v2/cache_fig1/fig1_data.npz'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'out/paper_figs_v2/fig1_inspiration'))
    a = ap.parse_args()
    Dt = Data(a.cache)
    os.makedirs(os.path.join(a.out, 'elements'), exist_ok=True)

    W, H, body, _ = build_sheet(Dt)
    svg = svg_doc(W, H, ''.join(body))
    base = os.path.join(a.out, 'workflow_icons')
    with open(base + '.svg', 'w') as fh:
        fh.write(svg)
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=base + '.pdf')
    cairosvg.svg2png(bytestring=svg.encode(), write_to=base + '.png', output_width=W * 1.5)

    # one file per element (icon group + an editable label group)
    n = 0
    for name, fns in GALLERY:
        for k, fn in enumerate(fns):
            w, h, body = fn(Dt)
            lab = g(text(w / 2, h + 22, name, 17), 'label')
            doc = svg_doc(w, h + 30, g(body, fname(fn)) + lab, bg=False)
            with open(os.path.join(a.out, 'elements', f'{fname(fn)}.svg'), 'w') as fh:
                fh.write(doc)
            n += 1
    # extra elements used only inside the workflow row
    for fn, lab in [(el_cube_em, 'emitter'), (el_bg_bars, 'bg')]:
        w, h, body = fn(Dt)
        doc = svg_doc(w, h + 30, g(body, fname(fn)) + g(text(w / 2, h + 22, lab, 17), 'label'), bg=False)
        with open(os.path.join(a.out, 'elements', f'{fname(fn)}.svg'), 'w') as fh:
            fh.write(doc)
        n += 1
    print(f'{base}.{{svg,pdf,png}}  ({W} x {H}); {n} element files in {a.out}/elements')


if __name__ == '__main__':
    main()
