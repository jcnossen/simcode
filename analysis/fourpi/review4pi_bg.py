"""Review, part 3: where is the lateral information lost? Conditional error ratios of the production row, and
hand-built LINEAR read-outs with pass-1 knowledge only (diagnostic rows, not a pipeline proposal):
template = phase-averaged LUT channel PSF at the network prior (x, y, z), weights shared by the 6 frames,
background (a) true, (b) one value per channel from the joint linear LS over the 6 frames, (c) one value per
(frame, channel) from the joint LS of that image alone, (d) the border median per (frame, channel) (the head's b0).
C read-out and its variance from the network. Usage: python analysis/fourpi/review4pi_bg.py --head v3L
"""
import argparse, json, math, os, sys
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.sim4pi import Sim4Pi
from simcode2.fourpi.solve4pi import Solve4Pi
from eval_head4pi import run_net
from review4pi import solve_chunks, load_head, EDGES, V
dev = 'cuda'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--head', default='v3L'); a = ap.parse_args()
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/NPC_f_dxy15.json'))['psf']
    lut = LUT4Pi(st, P=15, dev=dev)
    S = Sim4Pi(lut, seed=11).sample(6000, p_nb=0.0, p_on=1.0, eps_range=math.pi)
    bc = torch.load(f'{REPO}/out/4pi/sim/bench_classical_epspi.pt', weights_only=False)
    assert torch.allclose(bc['S']['x'], S['x'])
    n = 6000; tr = torch.stack([S['x'], S['y'], S['z']], 1); sc = torch.tensor([128.0, 128.0, 1000.0], device=dev)
    cr = (bc['cr_p'][:, :3] * sc).cpu().numpy(); Nw = S['Nt'].sum(1).cpu().numpy()
    sv = Solve4Pi(lut); sv.build_table()
    net, ck = load_head(a.head); o = run_net(net, S)
    netp = dict(dxy=o['dxy'], z=o['z'], ls=o['ls'])
    img = S['img']; P = 15
    rows = {}
    r = solve_chunks(sv, dict(T=o['T'], C=o['C'], L=o['L'], **netp), S)
    rows['network (production)'] = ((r['th'][:, :3] - tr) * sc).cpu().numpy()
    # ---- conditional ratios of the production row
    e = rows['network (production)']; u = e / cr
    bgl = S['bg'][:, 0].mean((1, 2, 3)).cpu().numpy(); zz = S['z'].abs().cpu().numpy(); mm = S['m'].cpu().numpy()
    rr = torch.sqrt(S['x'] ** 2 + S['y'] ** 2).cpu().numpy()
    print('== production row, lateral (x, y pooled) and z error / CRLB (5-sigma trim) by condition (terciles: low | mid | high)')
    out = dict(cond={}, rows={})
    for nm, v in (('background level', bgl), ('|z|', zz), ('modulation depth', mm), ('offset from the candidate', rr)):
        print(nm)
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi); q = np.quantile(v[s], [1 / 3, 2 / 3]); cells = []
            for j, sj in enumerate((v <= q[0], (v > q[0]) & (v <= q[1]), v > q[1])):
                ul = u[s & sj][:, :2].ravel(); uz = u[s & sj][:, 2]
                cells.append(f'{np.sqrt(np.mean(ul[np.abs(ul) < 5] ** 2)):.2f} / {np.sqrt(np.mean(uz[np.abs(uz) < 5] ** 2)):.2f}')
            out['cond'].setdefault(nm, []).append(cells)
            print(f'  {lo}-{hi}: ' + ' | '.join(cells) + f'   (tercile edges {q[0]:.2f}, {q[1]:.2f})')
    # ---- hand-built linear read-outs
    border = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], -1)
    b0 = border.median(-1).values                                             # [n,6,4]
    btrue = S['bg'].mean((-1, -2))
    varC = o['L'][..., 1, 0] ** 2 + o['L'][..., 1, 1] ** 2
    G = torch.stack([lut.luts[0][c].evaluate(torch.stack([o['dxy'][:, 0] + S['doff'][:, c, 0], o['dxy'][:, 1] + S['doff'][:, c, 1], o['z']], 1)).clamp(min=1e-9)
                     for c in range(4)], 1)                                     # [n,4,P,P]
    G = G / G.sum((-1, -2), keepdim=True)

    def mf(bgv, var):      # matched filter with the template G and a given background [n,6,4]; var [n,6,4,P,P]
        w = G[:, None] / var; w = w / (w * G[:, None]).sum((-1, -2), keepdim=True)
        T = (w * (img - bgv[..., None, None])).sum((-1, -2)); vT = (w * w * var).sum((-1, -2))
        return T, vT

    def joint(shared):
        """weighted linear LS of img_tc = T_tc G_c + b; b per channel (shared over frames) or per (t, c); 2 reweightings"""
        var = (img.mean(1, keepdim=True).clamp(min=0) + V).expand_as(img) if shared else (img.clamp(min=0) * 0 + img.mean((-1, -2), keepdim=True).clamp(min=0.3) + V)
        b = b0.mean(1, keepdim=True).expand(-1, 6, -1) if shared else b0
        for _ in range(3):
            w = 1.0 / var
            Sgg = (w * G[:, None] ** 2).sum((-1, -2)); Sg = (w * G[:, None]).sum((-1, -2)); S1 = w.sum((-1, -2))
            Sgy = (w * G[:, None] * img).sum((-1, -2)); Sy = (w * img).sum((-1, -2))
            if shared:   # eliminate T_t: T_t = (Sgy_t - b Sg_t) / Sgg_t ; b from the pooled normal equation
                num = (Sy - Sg * Sgy / Sgg).sum(1); den = (S1 - Sg ** 2 / Sgg).sum(1)
                b = (num / den)[:, None].expand(-1, 6, -1)
            else:
                b = (Sy - Sg * Sgy / Sgg) / (S1 - Sg ** 2 / Sgg)
            T = (Sgy - b * Sg) / Sgg
            var = (T.clamp(min=0)[..., None, None] * G[:, None] + b.clamp(min=0.05)[..., None, None] + V)
        return T, 1.0 / Sgg, b

    def row(nm, T, vT):
        L = torch.zeros(n, 6, 4, 2, 2, device=dev); L[..., 0, 0] = vT.clamp(min=1e-6).sqrt(); L[..., 1, 1] = varC.sqrt()
        for pn, pr in (('network prior', netp),):
            r = solve_chunks(sv, dict(T=T, C=o['C'], L=L, **pr), S)
            rows[nm] = ((r['th'][:, :3] - tr) * sc).cpu().numpy()
    var_true = S['mu'] + S['bg'] + V
    T1, v1 = mf(btrue, var_true); row('template MF, true variance weights, true bg', T1, v1)
    Tj, vj, bj = joint(True); row('template LS, bg per channel (joint over frames)', Tj, vj)
    Tk, vk, bk = joint(False); row('template LS, bg per (frame, channel)', Tk, vk)
    var_e = (Tj.clamp(min=0)[..., None, None] * G[:, None] + bj.clamp(min=0.05)[..., None, None] + V)
    T3, v3 = mf(b0, var_e); row('template MF, border-median bg per (frame, channel)', T3, v3)
    T4, v4 = mf(o['bg'], var_e); row('template MF, network bg', T4, v4)
    # Gaussian template sd 1.3 at the prior position (the head's own anchor), shared bg
    print('\nbg error (e/px RMS): joint per channel %.3f | per (t,c) LS %.3f | border median %.3f | network %.3f' % (
        float((bj - btrue).pow(2).mean().sqrt()), float((bk - btrue).pow(2).mean().sqrt()), float((b0 - btrue).pow(2).mean().sqrt()),
        float((o['bg'] - btrue).pow(2).mean().sqrt())))
    print('network bg error: frame-common part %.3f, frame-specific part %.3f e/px' % (
        float((o['bg'] - btrue).mean(1).pow(2).mean().sqrt()), float(((o['bg'] - btrue) - (o['bg'] - btrue).mean(1, keepdim=True)).pow(2).mean().sqrt())))
    print('\n== RMS (5-sigma trim) / pixel CRLB, x / y / z   (hand-built T, network C, network prior)')
    for nm, e in rows.items():
        cells = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (Nw >= lo) & (Nw < hi); uu = e[s] / cr[s]
            q = [float(np.sqrt(np.mean(uu[np.abs(uu[:, k]) < 5, k] ** 2))) for k in range(3)]
            out['rows'].setdefault(nm, []).append(q); cells.append(' / '.join(f'{v:.2f}' for v in q))
        print(f'| {nm} | ' + ' | '.join(cells) + ' |')
    json.dump(out, open(f'{REPO}/out/4pi/sim/review_bg_{a.head}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
