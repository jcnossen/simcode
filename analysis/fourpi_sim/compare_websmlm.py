"""Compare mt_cells.py against webSMLM's own generator, run headless in Playwright Chromium.

Loads ~/websmlm_src/webSMLM/webSMLM.html (file://), and for each seed runs, in the page:
  * mtStartPosition() + CellField.buildWindow() exactly as buildMicrotubuleStructure() does (webSMLM defaults:
    structure window 141 x 141 px of 100 nm, focus 250 nm, +-750 nm, mtDensity 0.9, cell occupancy 0.33);
  * through a second instance of the embedded CellField block (its workerSource() with the WASM module handle
    exposed), the core's C API: isc_cells_in_window and isc_cell_microtubules for the cells reaching the window.
Then the same with mt_cells.py, and compares: start view, nCells, nMt, number of dye sites (100 % lattice), the dye
positions themselves, per-cell microtubule count / total length / footprint density, z distribution.
Output: out/fourpi_sim/compare_websmlm.json (+ printed table).
Usage: python analysis/fourpi_sim/compare_websmlm.py [--seeds 1 2 3 4 5 6 7 8]
"""
import argparse, base64, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mt_cells as M

HTML = os.path.expanduser('~/websmlm_src/webSMLM/webSMLM.html')
OUT = os.path.expanduser('~/simcode2/out/fourpi_sim')

JS = r"""
async (a) => {
  const b64 = (f64) => { const u = new Uint8Array(f64.buffer); let s = ''; const C = 0x8000;
    for (let i = 0; i < u.length; i += C) s += String.fromCharCode.apply(null, u.subarray(i, i + C)); return btoa(s); };
  const src = CellField.workerSource();
  const key = 'return { buildWindow,';
  if (src.split(key).length !== 2) throw new Error('hook not unique');
  const CF2 = (0, eval)(src.replace(key, 'return { M: () => M, getWorld, buildWindow,'));
  const out = [];
  for (const seed of a.seeds) {
    const opts = { pxnm: a.px, mtDensity: a.mtDensity, cellDensity: a.cellDensity, focusUm: a.focusUm, slabNm: a.slabNm };
    let t0 = performance.now();
    const st = mtStartPosition(seed, a.w, a.h, opts);
    const xUm = Math.round(st.x * 1000) / 1000, yUm = Math.round(st.y * 1000) / 1000;
    const tStart = performance.now() - t0; t0 = performance.now();
    const res = CellField.buildWindow(a.w, a.h, Object.assign({ seed, xUm, yUm }, opts));
    const tBuild = performance.now() - t0;
    const n = res.sites.length, flat = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) { flat[3*i] = res.sites[i][0]; flat[3*i+1] = res.sites[i][1]; flat[3*i+2] = res.sites[i][2]; }
    // core C API through the second instance
    CF2.buildWindow(1, 1, Object.assign({ seed, xUm, yUm }, opts));
    const Mo = CF2.M();
    const p = Object.assign({}, CF2.defaults, { mtDensity: a.mtDensity, density: a.cellDensity });
    const wd = CF2.getWorld(seed, p);
    const pxUm = a.px / 1000, hw = a.w * pxUm / 2, hh = a.h * pxUm / 2;
    const x0 = xUm - hw - 0.5, y0 = yUm - hh - 0.5, x1 = xUm + hw + 0.5, y1 = yUm + hh + 0.5;
    let cap = 256, ptr = Mo._malloc(cap * 14 * 8);
    const nc = Mo._isc_cells_in_window(wd, x0, y0, x1, y1, ptr, cap);
    const cells = Array.from(Mo.HEAPF64.slice(ptr / 8, ptr / 8 + Math.min(nc, cap) * 14));
    Mo._free(ptr);
    const cellOut = [];
    for (let c = 0; c < Math.min(nc, cap); c++) {
      const cx = cells[14*c], cy = cells[14*c+1];
      let capPts = 1 << 18, capMts = 8192, nm, tot, xyz, lens;
      for (;;) {
        const pX = Mo._malloc(capPts * 24), pL = Mo._malloc(capMts * 4), pT = Mo._malloc(4);
        nm = Mo._isc_cell_microtubules(wd, cx, cy, pX, capPts, pL, capMts, pT);
        tot = Mo.HEAP32[pT >> 2];
        if (nm >= 0 && nm <= capMts && tot <= capPts) {
          xyz = Mo.HEAPF64.slice(pX / 8, pX / 8 + tot * 3); lens = Mo.HEAP32.slice(pL >> 2, (pL >> 2) + nm);
        }
        Mo._free(pX); Mo._free(pL); Mo._free(pT);
        if (nm < 0 || xyz) break;
        capPts = Math.max(capPts, tot); capMts = Math.max(capMts, nm);
      }
      cellOut.push({ fields: cells.slice(14*c, 14*c+14), nMt: nm, lens: Array.from(lens || []), xyz: xyz ? b64(xyz) : '' });
    }
    out.push({ seed, st, xUm, yUm, nSites: n, nCells: res.nCells, nMt: res.nMt, sites: b64(flat), cells: cellOut,
               tStart, tBuild });
  }
  return out;
}
"""


def dec(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float64) if s else np.zeros(0)


def run_page(seeds, w, h, px, focus_um, slab_nm, mt_density, cell_density):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        r = []
        for seed in seeds:                      # a fresh page per seed (one long evaluate can exhaust the renderer)
            pg = br.new_page(viewport=dict(width=1600, height=1000))
            pg.goto('file://' + HTML)
            pg.wait_for_function('() => typeof CellField !== "undefined" && typeof mtStartPosition === "function"',
                                 timeout=120000)
            pg.set_default_timeout(1800000)
            try:
                r += pg.evaluate(JS, dict(seeds=[seed], w=w, h=h, px=px, focusUm=focus_um, slabNm=slab_nm,
                                          mtDensity=mt_density, cellDensity=cell_density))
            except Exception as e:
                print(f'seed {seed}: page failed ({e})', flush=True)
            pg.close()
        br.close()
    return r


def poly_len(m):
    return float(np.linalg.norm(np.diff(m, axis=0), axis=1).sum()) if len(m) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument('--w', type=int, default=141); ap.add_argument('--px', type=float, default=100.0)
    ap.add_argument('--focus', type=float, default=0.25); ap.add_argument('--mt_density', type=float, default=0.9)
    ap.add_argument('--tag', default=''); ap.add_argument('--slab', type=float, default=750.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    page = run_page(a.seeds, a.w, a.w, a.px, a.focus, a.slab, a.mt_density, 0.33)
    print(f'webSMLM page: {time.time() - t:.1f} s', flush=True)
    from scipy.spatial import cKDTree
    rows = []
    for pr in page:
        seed = pr['seed']
        t = time.time()
        st = M.mt_start_position(seed, a.w, a.w, a.px, a.focus, a.slab, a.mt_density)
        xu, yu = round(st['x'] * 1000) / 1000, round(st['y'] * 1000) / 1000
        bw = M.build_window(a.w, a.w, seed, xu, yu, a.px, a.focus, a.slab, a.mt_density)
        tpy = time.time() - t
        S_js = dec(pr['sites']).reshape(-1, 3); S_py = bw['sites']
        same_order = len(S_js) == len(S_py)
        maxdiff = float(np.abs(S_js - S_py).max()) if same_order and len(S_js) else None
        # unordered match (px, px, nm) -> nm units
        if len(S_js) and len(S_py):
            sc = np.array([a.px, a.px, 1.0])
            d, _ = cKDTree(S_py * sc).query(S_js * sc, k=1)
            frac_match = float((d < 1e-3).mean())
        else:
            frac_match = None
        # per cell: microtubules
        F = M.get_field(seed, dict(mtDensity=a.mt_density, density=0.33))
        cells_cmp = []
        for cj in pr['cells']:
            f = cj['fields']; cx, cy = int(f[0]), int(f[1])
            c = F.find_cell(cx, cy)
            xyz = dec(cj['xyz']).reshape(-1, 3)
            lens = np.array(cj['lens'], int)
            mts_js = np.split(xyz, np.cumsum(lens)[:-1]) if len(lens) else []
            A = F.assets(c)
            L_js = np.array([poly_len(m) for m in mts_js]); L_py = np.array([poly_len(m) for m in A['mts']])
            exact = (len(mts_js) == len(A['mts']) and
                     all(len(m1) == len(m2) and (len(m1) == 0 or np.abs(m1 - m2).max() < 1e-9)
                         for m1, m2 in zip(mts_js, A['mts'])))
            n_mt_same = sum(len(m1) == len(m2) and (len(m1) == 0 or np.abs(m1 - m2).max() < 1e-9)
                            for m1, m2 in zip(mts_js, A['mts']))
            pyf = [c[M.C_cx], c[M.C_cy], c[M.C_x], c[M.C_y], c[M.C_packRot], c[M.C_rOuter], c[M.C_height],
                   c[M.C_nucOffX], c[M.C_nucOffY], c[M.C_nucRot], c[M.C_nucLong], c[M.C_nucShort],
                   c[M.C_nucHeight], c[M.C_nucZ]]
            zs_js = np.concatenate(mts_js)[:, 2] if len(mts_js) else np.zeros(0)
            cells_cmp.append(dict(cx=cx, cy=cy, cell_fields_maxdiff=float(np.abs(np.array(f) - np.array(pyf)).max()),
                                  area_um2=A['area_um2'],
                                  n_mt_js=int(len(mts_js)), n_mt_py=len(A['mts']),
                                  mt_identical=int(n_mt_same), all_identical=bool(exact),
                                  length_um_js=float(L_js.sum()), length_um_py=float(L_py.sum()),
                                  length_per_um2_js=float(L_js.sum() / A['area_um2']),
                                  length_per_um2_py=float(L_py.sum() / A['area_um2']),
                                  mean_mt_len_js=float(L_js.mean()) if len(L_js) else 0.0,
                                  mean_mt_len_py=float(L_py.mean()) if len(L_py) else 0.0,
                                  z_centre_pct_js=[float(v) for v in np.percentile(zs_js, [5, 25, 50, 75, 95])],
                                  z_centre_pct_py=[float(v) for v in np.percentile(
                                      np.concatenate(A['mts'])[:, 2], [5, 25, 50, 75, 95])]))
        zj = S_js[:, 2] if len(S_js) else np.zeros(1); zp = S_py[:, 2] if len(S_py) else np.zeros(1)
        row = dict(seed=seed, start_js=pr['st'], start_py=st, nCells_js=pr['nCells'], nCells_py=bw['nCells'],
                   nMt_js=pr['nMt'], nMt_py=bw['nMt'], nSites_js=pr['nSites'], nSites_py=int(len(S_py)),
                   sites_same_order=same_order, sites_maxdiff_px_px_nm=maxdiff, sites_frac_matched_1pm=frac_match,
                   site_z_pct_js=[float(v) for v in np.percentile(zj + a.focus * 1000, [5, 25, 50, 75, 95])],
                   site_z_pct_py=[float(v) for v in np.percentile(zp + a.focus * 1000, [5, 25, 50, 75, 95])],
                   t_js_start_s=pr['tStart'] / 1000, t_js_build_s=pr['tBuild'] / 1000, t_py_s=tpy, cells=cells_cmp)
        rows.append(row)
        print(f"seed {seed}: start js {pr['st']['x']},{pr['st']['y']} cover {pr['st']['cover']:.2f} | py {st['x']},"
              f"{st['y']} {st['cover']:.2f}; nCells {pr['nCells']}/{bw['nCells']} nMt {pr['nMt']}/{bw['nMt']} "
              f"sites {pr['nSites']}/{len(S_py)} same_order {same_order} maxdiff {maxdiff} matched {frac_match}; "
              f"cells {[(c['n_mt_js'], c['n_mt_py'], c['mt_identical']) for c in cells_cmp]}; "
              f"time js {pr['tStart'] / 1000 + pr['tBuild'] / 1000:.1f} s py {tpy:.1f} s", flush=True)
    json.dump(dict(args=vars(a), rows=rows), open(f'{OUT}/compare_websmlm{a.tag}.json', 'w'), indent=1)
    print('wrote', f'{OUT}/compare_websmlm{a.tag}.json')


if __name__ == '__main__':
    main()
