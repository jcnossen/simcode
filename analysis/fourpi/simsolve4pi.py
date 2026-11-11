"""Pass 2 on a simulated 4Pi movie: the same code path as solve_real4pi.py (imported, unchanged): SIM pattern
re-estimated from the network intensities (reference k from the simulator's meta, as the real run uses the classical
estimate), per-candidate 4Pi phase with z by the prior, in-situ phase field M(x, y) + d(t), final kinetics / free solve.
Usage: CUDA_VISIBLE_DEVICES=2 python analysis/fourpi/simsolve4pi.py --tag d1 --model v2
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path: sys.path.insert(0, REPO)
import solve_real4pi as SR
from simcode2.fourpi.lut4pi import LUT4Pi
from simcode2.fourpi.solve4pi import Solve4Pi, kin_starts, best_of


def repattern_xy(I, X, ref, sel):
    """slot-DFT pattern estimate (as solve_real4pi.repattern) from per-frame intensities I [n,6] at positions X [n,2]"""
    from pattern import refine_k, slotZ, emitter_phase
    out = dict(n=int(sel.sum()))
    for a in range(2):
        I3 = I[sel, 3 * a:3 * a + 3]
        k = refine_k(I3, X[sel], np.array(ref[f'angle{a}']['k']), span=0.02, n=21)
        psi = -np.angle(slotZ(I3, X[sel], k)); ph, mm, _ = emitter_phase(I3, psi)
        out[f'angle{a}'] = dict(k=k.tolist(), psi_deg=np.degrees(psi).tolist(), m_median=float(np.nanmedian(np.clip(mm, 0, 1.5))),
                                 R=float(np.abs(np.mean(np.exp(1j * (ph - X[sel] @ k))))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='d1'); p.add_argument('--model', default='v2'); p.add_argument('--psf', default='NPC_f_dxy15')
    p.add_argument('--tile', type=float, default=12.0); p.add_argument('--tblk', type=int, default=50)
    p.add_argument('--dchi', type=float, default=7.81)
    p.add_argument('--phi', default='field', help="'field' (in-situ estimate) or 'oracle' (simulator truth at the prior position)")
    p.add_argument('--pattern', default='iter', help="'iter' (network intensities, re-estimated from solved positions), 'net' (solve_real4pi.repattern only), 'oracle' (simulator truth)")
    a = p.parse_args()
    from simcode2.util.proctitle import auto; auto('4pi-sim')
    d = f'{REPO}/out/4pi/simmovie/{a.tag}'; dev = 'cuda'
    meta = json.load(open(f'{d}/meta.json'))
    SR.OUT = d
    st = json.load(open(f'{REPO}/out/4pi/psf4pi/{a.psf}.json'))['psf']
    lut = LUT4Pi(st, P=15)
    t0 = time.time()
    o, mt = SR.load_front('SIM', 0, a.model, dev)
    n = len(mt['w']); cx, cy = mt['cx'].astype(np.float64), mt['cy'].astype(np.float64)
    ref = {f'angle{i}': dict(k=meta['k'][i], psi_deg=meta['psi_deg'][i], m_median=0.85) for i in range(2)}
    if a.pattern == 'oracle':
        patf = {f'angle{i}': dict(k=meta['k'][i], psi_deg=meta['psi_deg'][i], m_median=meta['m']) for i in range(2)}
    elif a.pattern == 'net':
        patf = SR.repattern(o, mt, ref)
    else:   # 'iter' / 'iter_gauge': relaxed selection at the prior positions
        I = o['T'].sum(2).cpu().numpy(); mI = I.mean(1)
        Xp = np.stack([cx + o['dxy'][:, 0].cpu().numpy(), cy + o['dxy'][:, 1].cpu().numpy()], 1)
        patf = repattern_xy(I, Xp, ref, (mI * 6 > 800) & (mt['nn'] > 3) & (I.min(1) > 0.25 * mI))
    psi, m, kk = SR.pattern_psi(patf, cx, cy, dev)
    sv = Solve4Pi(lut, k=kk, delta=(0.0, math.radians(meta['delta_deg']))); sv.build_table()
    idx = torch.arange(n, device=dev)
    rp = SR.solve_all(sv, o, idx, torch.zeros(n, device=dev), mt['doff'], m, psi, mode='free', fit_phi=True)
    phi = rp['th'][:, -1].cpu().numpy(); sphi = rp['sig'][:, -1].cpu().numpy()
    X = cx + o['dxy'][:, 0].cpu().numpy(); Y = cy + o['dxy'][:, 1].cpu().numpy()
    Tsum = o['T'].sum((1, 2)).cpu().numpy()
    good = (sphi < 0.8) & (Tsum > 800) & (mt['nn'] > 4)
    f, diag = SR.phase_field(X, Y, mt['w'], phi, good / np.maximum(sphi, 0.05) ** 2, tile=a.tile, tblk=a.tblk, H=meta['H'], Wd=meta['W'])
    Phi0 = torch.tensor(f(X, Y, mt['w']), device=dev, dtype=torch.float32)
    if a.phi == 'oracle':
        from simmovie4pi import phi0_field
        Ptrue = phi0_field(X, Y, 6 * mt['w'] + 3, meta['H'], meta['W'], meta['T'])
        print('field estimate vs truth: circ sd of (est - true) %.1f deg, mean offset %.1f deg' % (
            np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * (Phi0.cpu().numpy() - Ptrue))))))),
            np.degrees(np.angle(np.mean(np.exp(1j * (Phi0.cpu().numpy() - Ptrue)))))), flush=True)
        Phi0 = torch.tensor(Ptrue, device=dev, dtype=torch.float32)
    rfr = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='free')
    ks = [kin_starts({'th': rfr['th'][i:i + 4096]}, m[i:i + 4096], psi[i:i + 4096], sv) for i in range(0, n, 4096)]
    r1 = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='kin', kin0s=torch.cat([k[0] for k in ks]))
    r2 = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='kin', kin0s=torch.cat([k[1] for k in ks]))
    ron = best_of(r1, r2)
    use = (ron['chi2'] - rfr['chi2'] < a.dchi)
    if a.pattern in ('iter', 'iter_gauge'):   # one refinement (re-estimating from solved positions has no restoring force on the gauge): pattern from the solved positions of the kinetics-accepted, on-through rows
        I = o['T'].sum(2).cpu().numpy()
        Xs = np.stack([cx + ron['th'][:, 0].cpu().numpy(), cy + ron['th'][:, 1].cpu().numpy()], 1)
        onth = ((ron['th'][:, 4] <= 0) & (ron['th'][:, 5] >= 6)).cpu().numpy()
        patf = repattern_xy(I, Xs, patf, use.cpu().numpy() & onth & (I.sum(1) > 800))
        psi, m, kk = SR.pattern_psi(patf, cx, cy, dev)
        sv.k = kk
        rfr = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='free')
        ks = [kin_starts({'th': rfr['th'][i:i + 4096]}, m[i:i + 4096], psi[i:i + 4096], sv) for i in range(0, n, 4096)]
        r1 = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='kin', kin0s=torch.cat([k[0] for k in ks]))
        r2 = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='kin', kin0s=torch.cat([k[1] for k in ks]))
        ron = best_of(r1, r2)
        use = (ron['chi2'] - rfr['chi2'] < a.dchi)
    if a.pattern == 'iter_gauge':
        # fix ONLY the common pattern phase per angle (the gauge) with pattern-independent anchors: the filtered classical
        # summed-fit positions. Model I ~ 1 + m cos(k.r - psi): per-emitter phase phi_e (with the current psi) minus
        # k.r_anchor has circular mean g = -(psi_true - psi) -> psi_true = psi - g.
        from pattern import emitter_phase
        C = dict(np.load(f'{d}/classical_{a.model}_q.npz'))
        I = o['T'].sum(2).cpu().numpy(); mI = I.mean(1)
        q = (C['chi2px'] <= 4) & ~C['at_bound'] & C['conv'] & (C['N'] > 800) & (I.min(1) > 0.25 * mI)
        Xc = np.stack([C['x'], C['y']], 1)
        for an in range(2):
            A = patf[f'angle{an}']; k = np.array(A['k']); psi_ = np.radians(A['psi_deg'])
            ph, _, _ = emitter_phase(I[q, 3 * an:3 * an + 3], psi_)
            g = np.angle(np.mean(np.exp(1j * (ph - Xc[q] @ k))))
            A['psi_deg'] = np.degrees(psi_ - g).tolist(); A['gauge_deg'] = float(-np.degrees(g)); A['n_anchor'] = int(q.sum())
        psi, m, kk = SR.pattern_psi(patf, cx, cy, dev); sv.k = kk
        rfr = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='free')
        ks = [kin_starts({'th': rfr['th'][i:i + 4096]}, m[i:i + 4096], psi[i:i + 4096], sv) for i in range(0, n, 4096)]
        r1 = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='kin', kin0s=torch.cat([k[0] for k in ks]))
        r2 = SR.solve_all(sv, o, idx, Phi0, mt['doff'], m, psi, mode='kin', kin0s=torch.cat([k[1] for k in ks]))
        ron = best_of(r1, r2)
        use = (ron['chi2'] - rfr['chi2'] < a.dchi)
    pick = lambda A, B: torch.where(use.reshape(-1, *([1] * (A.dim() - 1))), A, B)
    th = pick(ron['th'][:, :3], rfr['th'][:, :3]); sg = pick(ron['sig'][:, :3], rfr['sig'][:, :3])
    tw = (torch.minimum(ron['th'][:, 5], torch.tensor(6.0, device=dev)) - torch.maximum(ron['th'][:, 4], torch.tensor(0.0, device=dev))).clamp(min=0)
    N = torch.where(use, ron['th'][:, 3] * tw, rfr['th'][:, 3:9].sum(1))
    out = dict(x=cx + th[:, 0].cpu().numpy(), y=cy + th[:, 1].cpu().numpy(), z=th[:, 2].cpu().numpy(),
               sx=sg[:, 0].cpu().numpy(), sy=sg[:, 1].cpu().numpy(), sz=sg[:, 2].cpu().numpy(), N=N.cpu().numpy(),
               on=use.cpu().numpy(), chi2_on=ron['chi2'].cpu().numpy(), chi2_free=rfr['chi2'].cpu().numpy(),
               phi=phi, sphi=sphi, Phi0=Phi0.cpu().numpy(), zprior=o['z'].cpu().numpy(), szprior=torch.exp(o['ls'][:, 2]).cpu().numpy(),
               xprior=X, yprior=Y, w=mt['w'], nn=mt['nn'], Tsum=Tsum)
    np.savez(f'{d}/locs_SIM_000_{a.model}_{a.pattern}' + ('' if a.phi == 'field' else '_phiO') + '.npz', **out, M=diag['M'], d=diag['d'])
    summ = dict(n=n, kin_frac=float(use.float().mean()), field_R=diag['R'], pattern=patf, t_s=time.time() - t0)
    json.dump(summ, open(f'{d}/solve_summary_{a.model}_{a.pattern}.json', 'w'), indent=1)
    print(summ, flush=True)


if __name__ == '__main__':
    main()
