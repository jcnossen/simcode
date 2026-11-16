"""Whole-movie comparison of the learned-window network table (post_w4pi.py) with the classical 4Pi-SIMFLUX table, using
the classical worker's comparison code (analysis/fourpi_classical/fig_cl.py; imported read-only, its output directory
redirected to <work>/net/cmp_<head>/ so nothing of the classical worker is overwritten): whole-field numbers (FRC, DAFL
on the same emitters, offsets), NPC averaged pore, and the individual pore sites of <work>/classical/sites/sites.json
rendered for the network and the classical pixel MLE with the same layout as npc_site.py.
Usage: python analysis/fourpi/cmp_full_w4pi.py --work <4Pi work folder> --cell NPC --head v10 [--files 2 3 10]
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/analysis/fourpi_classical')
import fig_cl as FC


def site_figure(D, sites, fn, head, nshow=6):
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    def img(px, py, cz, ext, bin_=1.5, sig=2.0, zr=None):
        e = np.arange(ext[0], ext[1] + bin_, bin_)
        Hn = np.histogram2d(py, px, bins=[e, e])[0]; Hn = gaussian_filter(Hn, sig / bin_)
        if cz is None: return Hn / max(Hn.max(), 1e-9)
        Hz = np.histogram2d(py, px, bins=[e, e], weights=cz)[0]; Hz = gaussian_filter(Hz, sig / bin_) / np.maximum(Hn, 1e-9)
        col = plt.get_cmap('turbo')(np.clip((Hz - zr[0]) / (zr[1] - zr[0]), 0, 1))[..., :3]
        v = np.clip(Hn / max(np.percentile(Hn[Hn > 0], 99.5), 1e-9), 0, 1) ** 0.7 if (Hn > 0).any() else Hn
        return col * v[..., None]
    meth = (('clp', 'classical pixel MLE'), ('net', f'network (head {head})'))
    T = {}
    for k, _ in meth:
        t = D[k]; m = t['ok'] & np.isfinite(t['x']) & np.isfinite(t['z'])
        T[k] = (cKDTree(np.c_[t['x'][m], t['y'][m]]), t['x'][m], t['y'][m], t['z'][m])
    n = min(nshow, len(sites)); fig, ax = plt.subplots(n, 6, figsize=(21, 3.6 * n), squeeze=False)
    ext = (-95, 95); zr = (-45, 45); stats = []
    for i, s in enumerate(sites[:n]):
        for j, (k, nm) in enumerate(meth):
            tr, x, y, z = T[k]
            ii = np.array(tr.query_ball_point([s['cx'], s['cy']], 95.0), dtype=int)
            px = x[ii] - s['cx']; py = y[ii] - s['cy']
            rr = np.hypot(px, py); ring = (rr > s['R'] - 16) & (rr < s['R'] + 16)
            zmid = np.median(z[ii][ring]) if ring.sum() > 20 else (np.median(z[ii]) if len(ii) else 0.0)
            pz = z[ii] - zmid
            lo = pz[ring] < 0
            zs = float(np.sqrt((pz[ring][lo].var() * lo.sum() + pz[ring][~lo].var() * (~lo).sum()) / max(ring.sum(), 1))) if lo.any() and (~lo).any() else np.nan
            sep = float(pz[ring][~lo].mean() - pz[ring][lo].mean()) if lo.any() and (~lo).any() else np.nan
            rsd = float(1.4826 * np.median(np.abs(rr[ring] - np.median(rr[ring])))) if ring.sum() > 5 else np.nan
            stats.append(dict(site=i + 1, method=k, n=int(len(ii)), ring_n=int(ring.sum()), radial_sd_nm=rsd, ring_sep_nm=sep, z_spread_per_ring_nm=zs))
            a0 = ax[i, 3 * j]
            a0.imshow(img(px, py, pz, ext, zr=zr), extent=[*ext, ext[1], ext[0]])
            a0.set_title(f'site {i + 1}, {nm}: top view, colour = z (±45 nm)\n{len(ii)} localizations, radial sd {rsd:.1f} nm', fontsize=8)
            for q, (pa, an) in enumerate(((px, 'x'), (py, 'y'))):
                e = np.arange(-95, 96.5, 1.5); ez = np.arange(-80, 81.5, 1.5)
                Hn = gaussian_filter(np.histogram2d(pz, pa, bins=[ez, e])[0], 2.0 / 1.5)
                a1 = ax[i, 3 * j + 1 + q]
                a1.imshow(Hn, cmap='inferno', extent=[-95, 95, 80, -80], aspect='equal'); a1.invert_yaxis()
                a1.set_title(f'side view {an}-z; ring separation {sep:.0f} nm, z spread per ring {zs:.1f} nm', fontsize=8)
        for a_ in ax[i]:
            a_.plot([-85, -35], [a_.get_ylim()[0] * 0.88] * 2, 'w-', lw=3); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle('NPC (Nup96) individual pore sites (picked on the classical table). Left: classical pixel MLE; right: network. Scale bar 50 nm; 1.5 nm bins, 2 nm blur; z drift removed per table.', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(fn, dpi=130); plt.close()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='NPC'); ap.add_argument('--head', default='v10'); ap.add_argument('--files', type=int, nargs='*')
    ap.add_argument('--cltag', default='_wv'); ap.add_argument('--sel', type=int, default=0, help="1: network rows = the 'sel' column (>= 1000 e, |z| <= 0.3 um) instead of 'good'")
    ap.add_argument('--work', required=True, help='4Pi work folder')
    a = ap.parse_args(); P4 = a.work
    CL = f'{P4}/classical'
    out = f'{P4}/net/cmp_{a.head}'; os.makedirs(f'{out}/fig', exist_ok=True); os.makedirs(f'{out}/real', exist_ok=True)
    FC.OUT = out
    FC.NETNAME = f'network pipeline, learned-window head {a.head} (4Pi-SIMCODE)'
    net_fn = f'{P4}/net/net_{a.cell}_{a.head}' + (f'_files{min(a.files)}-{max(a.files)}' if a.files else '') + '_all.npz'
    D = FC.load(f'{CL}/real/cl_{a.cell}{a.cltag}_all.npz', net_fn, None, None, zdrift=True)
    if a.files:     # restrict the classical tables to the same files
        for k in ('cl', 'clp'):
            m = np.isin(D[k]['f'], a.files); D[k]['ok'] = D[k]['ok'] & m
    tag = f'_{a.head}' + (f'_files{min(a.files)}-{max(a.files)}' if a.files else '')
    if a.sel:
        D['net']['ok'] = D['net']['ok'] & np.load(net_fn)['sel']; tag += '_sel'
    r0 = FC.numbers(D, f'{a.cell} whole field'); r0['drift_ref_offset_nm'] = D.get('drift_ref_offset_nm')
    L = FC.fmt_numbers(r0); res = [r0]
    zd = D['zdrift']; n_ = min(len(zd['net']), len(zd['cl']))
    L.append(f"\n## Sample z drift (structure-based, 500-window blocks, each table on its own)\nptp network {np.ptp(zd['net']):.0f} nm, classical {np.ptp(zd['cl']):.0f} nm; "
             f"rms difference of the two curves {np.sqrt(np.mean((zd['net'][:n_] - zd['cl'][:n_] - np.mean(zd['net'][:n_] - zd['cl'][:n_])) ** 2)):.1f} nm")
    if a.cell == 'NPC':
        po = FC.npc_figures(D, tag); res.append(po)
        L.append(f"\n## NPC pores ({po['n_pores']} pores, refitted per method)")
        L.append('| method | ring radius (peak / median, nm) | circle fit radius median | z peak separation (nm) | dip (valley / peak) |'); L.append('|---|---|---|---|---|')
        for k, m in po['methods'].items():
            L.append(f"| {k} | {m['ring_radius_peak_nm']:.1f} / {m['ring_radius_median_nm']:.1f} | {m['circle_fit_radius_median_nm']:.1f} | {m['peak_sep_nm']:.0f} | {m['dip_ratio']:.2f} |")
        sites = json.load(open(f'{CL}/sites/sites.json'))
        if a.files is None or len(a.files) > 30:
            # site coordinates are in the classical pixel-MLE drift frame; FC.load moved the classical tables by the drift-reference offset
            off = D.get('drift_ref_offset_nm', [0.0, 0.0])
            sites = [dict(s, cx=s['cx'] + off[0], cy=s['cy'] + off[1]) for s in sites]
            st = site_figure(D, sites, f'{out}/fig/NPC_sites{tag}.png', a.head); res.append(dict(sites=st))
            L.append('\n## Individual pore sites (same six sites as out/4pi/classical/sites/npc_sites.png)')
            L.append('| site | method | localizations | radial sd (nm) | ring separation (nm) | z spread per ring (nm) |'); L.append('|---|---|---|---|---|---|')
            L += [f"| {s['site']} | {s['method']} | {s['n']} | {s['radial_sd_nm']:.1f} | {s['ring_sep_nm']:.0f} | {s['z_spread_per_ring_nm']:.1f} |" for s in st]
    json.dump(res, open(f'{out}/compare_{a.cell}{tag}.json', 'w'), indent=1, default=float)
    open(f'{out}/compare_{a.cell}{tag}.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
