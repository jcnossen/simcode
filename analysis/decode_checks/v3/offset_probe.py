"""Cause of DECODE's constant ~+3 nm x offset on the isolated-emitter movie (iso_v2psf).
Same model, same GT emitters (positions, heights, detected photons), three inputs, all through dec_common.process_padded:
  movie  : the movie itself (ADU -> photons with camera_sim.npz), as run_decode.py
  decsim : 6-sums rendered with DECODE's training renderer (dec_common.LutPSF = the training LUT), flat bg 15 ph / px / 6-sum,
           Poisson + 3.28 e read noise, in photons (gain 1, offset 0)
  transp : decsim transposed (x <-> y swapped in the image), outputs swapped back (x-y convention test)
  flipx  : decsim mirrored in x (column j -> W-1-j), output x -> W-1-x (a learned asymmetry flips sign; a coordinate offset does not)
Output: mean error x / y (nm), A_bias x / y, per photon level -> out/decode_checks/v3/offset_probe.json
usage: CUDA_VISIBLE_DEVICES=2 python analysis/decode_checks/v3/offset_probe.py [--name dec_sim1809v2] [--model_file <pt>]"""
import os, sys, json, argparse
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAIN = os.path.expanduser('~/simcode2')            # outputs (out/) live in the main tree
sys.path[:0] = [os.path.join(REPO, 'analysis', 'baselines', 'decode'), os.path.join(REPO, 'analysis', 'pixbias'), REPO]
import decode_paper as DP
D = DP.D
import pblib as B, sim as PS

ISO = os.path.join(MAIN, 'out', 'decode_checks', 'iso_v2psf')


def load_model(name, model_file=None):
    D.SPD = os.path.join(MAIN, 'out', 'paper2', 'decode'); D.LUTPSF = f'{D.SPD}/lutpsf.txt'
    mdir = f'{D.SPD}/lutpsf/{name}'
    pj = json.load(open(f'{mdir}/psf.json'))
    DP.setup_psf(pj['psf'], pj['halo'], pj.get('zmax_um', DP.ZMAX))
    mf = model_file or f'{mdir}/model_final.pt'
    return D.load_trained(name, pj.get('variant_name', 'sum6p'), model_file=mf), pj


def render_sums(gt, nsum, bg=15.0, read=3.28, seed=0):
    """6-sum images [nsum, 40, 40] in photons of the iso GT with the DECODE training renderer"""
    luts, zmax, halo = D._LUTS
    lp = D.LutPSF([-0.5, 39.5], [-0.5, 39.5], [40, 40], luts, zmax, halo=0.0)
    w = (gt['ev_t0'] // 6).astype(np.int64); k = w < nsum
    xyz = torch.tensor(np.c_[gt['ev_x'][k], gt['ev_y'][k], gt['ev_z_um'][k] * 1e3], dtype=torch.float32)
    I = torch.tensor(gt['ev_I6'][k] / gt['ev_collected'][k], dtype=torch.float32)    # intensity parameter (image sums to I6)
    out = []
    g = torch.Generator(device='cuda'); g.manual_seed(seed)
    for s in range(0, nsum, 1000):
        m = (w[k] >= s) & (w[k] < s + 1000); n = min(1000, nsum - s)
        mu = lp.forward(xyz[m], I[m], torch.tensor(w[k][m] - s), n).cuda() + bg
        img = torch.poisson(mu, generator=g) + read * torch.randn(mu.shape, device='cuda', generator=g)
        out.append(img.cpu().numpy())
    return np.concatenate(out)


def infer_sums(inst, sums, gain=1.0, offset=0.0):
    em, _ = D.process_padded(inst, iter(list(sums)), gain=gain, offset=np.zeros((40, 40), np.float32) + offset, sumframes=1, batch_size=64, infer_batch_size=16)
    em = em[em.frame_ix < len(sums)]
    return dict(f=6 * em.frame_ix.numpy().astype(np.int64), x=em.xyz[:, 0].numpy().astype(float), y=em.xyz[:, 1].numpy().astype(float),
                photons=em.phot.numpy().astype(float))


def movie_sums(nsum):
    from smlmtorch.util.multipart_tiff import tiff_read_file
    cam = np.load(os.path.join(ISO, 'camera_sim.npz')); gain = float(cam['gain']); off = cam['offset_adu'].astype(np.float32)
    fr = np.asarray(list(tiff_read_file(os.path.join(ISO, 'movie.tif'), startframe=0, maxframes=nsum * 6, use_progbar=False)), np.float32)
    return fr.reshape(nsum, 6, 40, 40).sum(1), gain, off * 6


def evaluate(T, gt, nf):
    tx, ty, tph = PS.gt_rows(gt, nf, T['f'], T['x'], T['y'])
    g = np.isfinite(tx)
    ex, ey = (T['x'][g] - tx[g]) * B.PX, (T['y'][g] - ty[g]) * B.PX
    r = dict(n=int(len(T['x'])), n_matched=int(g.sum()), mean_x_nm=float(ex.mean()), mean_y_nm=float(ey.mean()),
             se_x_nm=float(ex.std() / np.sqrt(len(ex))), se_y_nm=float(ey.std() / np.sqrt(len(ey))),
             A_bias_x_nm=PS.bias_curve(ex / B.PX, B.sub(tx[g]))['A_nm'], A_bias_y_nm=PS.bias_curve(ey / B.PX, B.sub(ty[g]))['A_nm'])
    ph = {}
    for L in np.unique(tph[g]):
        m = tph[g] == L
        ph[str(int(L))] = dict(n=int(m.sum()), mean_x_nm=float(ex[m].mean()), mean_y_nm=float(ey[m].mean()))
    r['per_photons'] = ph
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='dec_sim1809v2'); ap.add_argument('--model_file', default=None)
    ap.add_argument('--nsum', type=int, default=6000); ap.add_argument('--out', default=os.path.join(MAIN, 'out', 'decode_checks', 'v3', 'offset_probe.json'))
    a = ap.parse_args()
    inst, pj = load_model(a.name, a.model_file)
    gt = dict(np.load(os.path.join(ISO, 'gt.npz'))); nf = a.nsum * 6
    R = {}
    ms, gain, off = movie_sums(a.nsum)
    R['movie'] = evaluate(infer_sums_cam(inst, ms, gain, off), gt, nf)
    S = render_sums(gt, a.nsum)
    R['decsim'] = evaluate(infer_sums(inst, S), gt, nf)
    T = infer_sums(inst, S.transpose(0, 2, 1).copy()); T['x'], T['y'] = T['y'], T['x']
    R['transp'] = evaluate(T, gt, nf)
    T = infer_sums(inst, S[:, :, ::-1].copy()); T['x'] = 39.0 - T['x']
    R['flipx'] = evaluate(T, gt, nf)
    for k, v in R.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'per_photons'}, flush=True)
        print('   per photons', {p: (round(q['mean_x_nm'], 2), round(q['mean_y_nm'], 2)) for p, q in v['per_photons'].items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(model=a.name, model_file=a.model_file, results=R), open(a.out, 'w'), indent=1)


def infer_sums_cam(inst, sums, gain, off):
    em, _ = D.process_padded(inst, iter(list(sums)), gain=gain, offset=off, sumframes=1, batch_size=64, infer_batch_size=16)
    em = em[em.frame_ix < len(sums)]
    return dict(f=6 * em.frame_ix.numpy().astype(np.int64), x=em.xyz[:, 0].numpy().astype(float), y=em.xyz[:, 1].numpy().astype(float),
                photons=em.phot.numpy().astype(float))


if __name__ == '__main__':
    main()
