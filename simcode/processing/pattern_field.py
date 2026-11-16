"""
Pattern FIELD maps (2026-11-10, opt-in; the default pipeline does not use them): evaluation of a per-cell illumination phase
map and a smooth (Legendre-2) depth field per angle on top of a global pattern, at given positions (field_at; used by the
real-data-like simulation analysis/simreal/sim_movie_r.py, SIM_PFIELD=<field .pt>). The estimator of these maps (estimate_field,
2026-11-10) was removed in the 2026-11-15 cleanup (no user; see git history).

Model: illumination of angle a at position r = rel (1 + m_a f_a(r) cos(k_a.r - psi_s - eps_a(t) - phi_a(r))), with
  phi_a(r): bilinear interpolation of a cell map (cell size `cell` px, movie pixel frame), and
  f_a(r): depth factor = D_a(r) / <D_a>, D_a a Legendre-2 polynomial in u = (x - W/2)/(W/2), v = (y - H/2)/(H/2).
"""
import numpy as np


def leg2(u, v):
    return np.stack([np.ones_like(u), u, v, 0.5 * (3 * u ** 2 - 1), u * v, 0.5 * (3 * v ** 2 - 1)], -1)


def field_at(F, x, y):
    """dphi [n,2] (rad, added to the pattern phase) and depth factor [n,2] at movie-pixel positions x, y.
    F['map_offset'] = (dx, dy) (optional): the phase map lives in another frame (e.g. the 0.5 nM sensor frame): map coords =
    (x + dx, y + dy). The depth field is in the movie's own frame (centre W/2, H/2) unless F['depth_offset'] = (dx, dy) is
    set (2026-11-10, optional; a depth field transferred from another sample): it is then evaluated at (x + dx, y + dy) in the
    frame of the sample it was fitted on (F['shape']). F['phase_on'] = False (depth-only field): dphi = 0."""
    cell = F['cell']; Hh, Ww = F['shape']
    mo = F.get('map_offset', (0.0, 0.0)); do = F.get('depth_offset', (0.0, 0.0))
    out_p = np.zeros((len(x), 2)); out_f = np.ones((len(x), 2))
    for a in (0, 1):
        if len(F.get('depth_coef', [])) == 2:
            xd, yd = x + do[0], y + do[1]
            Dv = leg2((xd - Ww / 2) / (Ww / 2), (yd - Hh / 2) / (Hh / 2)) @ np.asarray(F['depth_coef'][a])
            out_f[:, a] = np.clip(Dv / F['depth_norm'][a], 0.3, 1.5)
        if not F.get('phase_on', True): continue
        M = np.asarray(F['phase'][a]); ncy, ncx = M.shape
        gx = np.clip((x + mo[0]) / cell - 0.5, 0, ncx - 1); gy = np.clip((y + mo[1]) / cell - 0.5, 0, ncy - 1)
        x0 = np.clip(np.floor(gx).astype(int), 0, max(ncx - 2, 0)); y0 = np.clip(np.floor(gy).astype(int), 0, max(ncy - 2, 0))
        tx = np.clip(gx - x0, 0, 1); ty = np.clip(gy - y0, 0, 1)
        x1 = np.minimum(x0 + 1, ncx - 1); y1 = np.minimum(y0 + 1, ncy - 1)
        out_p[:, a] = (M[y0, x0] * (1 - tx) * (1 - ty) + M[y0, x1] * tx * (1 - ty) + M[y1, x0] * (1 - tx) * ty + M[y1, x1] * tx * ty)
    return out_p, out_f


