"""webSMLM microtubule ground truth for the paper sims (the *_wsm movie sets, 2026-11-15).

Structure: webSMLM's 'microtubules (cells)' sample (webSMLM embeds the inSiliScope CellField world model; Python port
analysis/fourpi_sim/mt_cells.py, checked against webSMLM by analysis/fourpi_sim/compare_websmlm.py; cite webSMLM /
inSiliScope, Martens2026inSiliScope). webSMLM defaults throughout (cell field, 0.9 microtubules / um^2 of cell footprint,
optical section focus 250 nm +- 750 nm, the 13_3 protofilament lattice with 12 nm binder + 2-5 nm linker, i.e. a label
shell ~25-30 nm from the tube axis) except:
  - field of view = the paper sims' 40 x 40 px of 108.33 nm (4.333 um), placed at the seed's webSMLM start view + an offset;
  - label efficiency LABEL_EFF = 500 / 1625: 0.5 labels per nm of microtubule, the line density of the preprint GT
    (common.preprint_tubules: linedensity 0.5 / nm; webSMLM's default is 70 % of the 1625 lattice sites per um);
  - heights: the label's own z is NOT used for rendering. The movie generators draw one height per label with the
    experiment-matched v2 calibration rule (common.heights_v2), as for the preprint GT (2D sims); the webSMLM z is kept in
    gt.npz as wsm_z_nm for reference.
The zoom box of the figures (16 px, as the preprint's crop) is picked at the densest filament crossing
(analysis/figures/zoompick.py pick_crossings) and stored in gen.json 'zoom_crop' = [x0, y0, w] px.

usage: python analysis/papersim/wsm_gt.py --candidates 1 2 3 ... [--offsets 0,0 ...]   -> out/papersim/websmlm_gt/candidates*.png/.json
       (the generators call wsm_gt.load(spec) with spec = 'seed:dx:dy')"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, 'analysis', 'fourpi_sim')); sys.path.insert(0, os.path.join(REPO, 'analysis', 'figures'))
import common as C

W = 40
FOV_UM = W * C.PX_NM / 1000
LATTICE_PER_UM = 1625.0                 # 13 sites per 8 nm (mt_cells: 13_3 lattice)
LINE_DENSITY_NM = 0.5                   # labels per nm of filament (= the preprint GT, common.preprint_tubules)
LABEL_EFF = LINE_DENSITY_NM * 1000 / LATTICE_PER_UM
FOCUS_NM, ZR_NM = 250.0, 750.0          # webSMLM defaults (optical section)
ZOOM_W = 16
OUTD = os.path.join(C.OUT, 'websmlm_gt')


def parse(spec):
    s = str(spec).split(':'); seed = int(s[0]); dx = float(s[1]) if len(s) > 1 else 0.0; dy = float(s[2]) if len(s) > 2 else 0.0
    return seed, dx, dy


def make(spec):
    """-> dict(pts [N,3] px (z = 0), mt [N], wsm_z_nm [N], centre (list of [n,2] px centrelines), stats, zoom_crop)"""
    import mt_cells as MC
    seed, dx, dy = parse(spec)
    r = MC.simulate(seed=seed, fov_um=(FOV_UM, FOV_UM), offset_um=(dx, dy), focus_nm=FOCUS_NM, z_range_nm=ZR_NM, label_efficiency=LABEL_EFF)
    xyz = r['xyz']
    # px, pixel centre = integer: x_px = x_nm / px - 0.5 (field [0, 4333) nm -> [-0.5, 39.5) px, as the preprint GT)
    pts = np.c_[xyz[:, 0] / C.PX_NM - 0.5, xyz[:, 1] / C.PX_NM - 0.5, np.zeros(len(xyz))]
    st = r['stats']
    st = dict(spec=f'{seed}:{dx:g}:{dy:g}', seed=seed, offset_um=[dx, dy], center_um=list(r['center_um']), n_labels=int(len(pts)),
              mt_length_um=float(st['mt_length_um']), labels_per_nm=float(len(pts) / (st['mt_length_um'] * 1000)),
              n_mt_in_fov=int(st['n_mt_in_fov']), label_eff=LABEL_EFF, focus_nm=FOCUS_NM, z_range_nm=ZR_NM, fov_um=FOV_UM)
    cl = [np.c_[m[:, 0] / C.PX_NM - 0.5, m[:, 1] / C.PX_NM - 0.5] for m in r['centrelines']]
    return dict(pts=pts, mt=r['mt'], wsm_z_nm=xyz[:, 2], centre=cl, stats=st, zoom_crop=zoom_box(pts))


def zoom_box(pts, n=4):
    """16 px zoom boxes at filament crossings (densest first), corner = integer px as fig_density.CROP"""
    from zoompick import pick_crossings
    lim = (-0.5, W - 0.5, -0.5, W - 0.5)
    b = pick_crossings(pts[:, 0], pts[:, 1], lim=lim, n=n, size=ZOOM_W, px=0.2, margin=ZOOM_W / 2 + 0.5)
    out = [[int(round(x0)), int(round(y0)), ZOOM_W] for x0, x1, y0, y1 in b]
    return out


def load(spec):
    """cached GT of a spec (out/papersim/websmlm_gt/gt_<seed>_<dx>_<dy>.npz)"""
    seed, dx, dy = parse(spec)
    f = os.path.join(OUTD, f'gt_{seed}_{dx:g}_{dy:g}.npz')
    if not os.path.exists(f):
        g = make(spec); os.makedirs(OUTD, exist_ok=True)
        cl = np.empty(len(g['centre']), object); cl[:] = g['centre']
        np.savez_compressed(f, pts=g['pts'], mt=g['mt'], wsm_z_nm=g['wsm_z_nm'], stats=json.dumps(g['stats']), zoom_crop=np.array(g['zoom_crop']), centre=cl)
    z = np.load(f, allow_pickle=True)
    return dict(pts=z['pts'], mt=z['mt'], wsm_z_nm=z['wsm_z_nm'], stats=json.loads(str(z['stats'])), zoom_crop=z['zoom_crop'].tolist(), centre=list(z['centre']))


def render_full(pts, px=0.1):
    from scipy.ndimage import gaussian_filter
    n = int(W / px)
    img = np.histogram2d(pts[:, 1], pts[:, 0], bins=n, range=[[-0.5, W - 0.5], [-0.5, W - 0.5]])[0]
    img = gaussian_filter(img, 1.0)
    return np.clip(img / (np.percentile(img, 99.5) + 1e-9), 0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidates', nargs='+', required=True, help="specs 'seed[:dx:dy]' (offset um from webSMLM's start view)")
    ap.add_argument('--out', default=os.path.join(OUTD, 'candidates.png')); a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from fig_density import render
    ref = C.preprint_tubules()[0]
    G = [load(s) for s in a.candidates]
    n = len(G) + 1
    fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 7.0))
    items = [('current GT (preprint, seed 1)', ref, dict(n_labels=len(ref)), [[9, 10, 16]])] + \
            [(f"webSMLM seed {g['stats']['spec']}", g['pts'], g['stats'], g['zoom_crop']) for g in G]
    for i, (t, p, st, zc) in enumerate(items):
        A = ax[0, i]; A.imshow(render_full(p), cmap='inferno', extent=[-0.5, W - 0.5, W - 0.5, -0.5]); A.set_xticks([]); A.set_yticks([])
        for j, (x0, y0, w) in enumerate(zc[:3]):
            A.add_patch(Rectangle((x0, y0), w, w, fill=False, ec='w' if j == 0 else '0.6', lw=1.5 if j == 0 else 0.8, ls='-' if j == 0 else '--'))
        s = f"{st['n_labels']} labels" + (f", {st['mt_length_um']:.1f} µm MT, {st['labels_per_nm']:.2f}/nm, {st['n_mt_in_fov']} MTs" if 'mt_length_um' in st else '')
        A.set_title(f'{t}\n{s}', fontsize=7)
        (x0, y0, w) = zc[0] if zc else (12, 12, ZOOM_W); A = ax[1, i]; A.imshow(render(p[:, 0], p[:, 1], (x0, y0), w), cmap='inferno'); A.set_xticks([]); A.set_yticks([])
        A.set_title(f'zoom corner ({x0}, {y0}) px, {w} px', fontsize=7)
    plt.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); plt.savefig(a.out, dpi=130); print(a.out)
    json.dump([g['stats'] | dict(zoom_crop=g['zoom_crop']) for g in G], open(a.out[:-4] + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
