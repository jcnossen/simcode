"""R24: simulated 40 nm / 20 nm nanoruler movie matched to the nr40 / nr20 movie = analysis/nanoruler/sim_nr.py run UNCHANGED
except for one line (its source is read, the output-directory assert is relaxed to out/nanoruler/nr40|nr20, and it is executed;
no file is edited). This wrapper prepares the movie-specific inputs first:
  <out>/../calib_src_<mv>.json  = out/nanoruler/calib_<mv>.json with (a) the absolute camera-map path, (b) the pattern depth per
       stored step set to the paper's estimated depths of that movie (r2040/FACTS.md section 3, report.txt, reordered to the
       stored order [0,2,4,1,3,5]); the calib json itself carries the nr80 placeholder 0.88 / 0.92
  <out>/../preset_<mv>.json     = out/nanoruler/nr80_preset.json with the movie's k vectors and camera maps
Movie-matched defaults (overridable on the command line, as sim_nr.py): site spacing, ON density (nr80 default 9.5e-5 scaled by
the quick-fit density ratio of FACTS.md section 4), window photons (2150 scaled by the quick-fit photon ratio), frac3 0.75
(nr80: 95 of 126 automatic rulers have three sites), min_dist 300 nm, sample drift rms.
GRID structures (the real nr40 / nr20 samples are 2D origami: nr40 sites on the corners of a ~40 nm square, nr20 a ~20 nm grid): options
  --grid nx,ny --occ p   (handled here, not passed on): sim_nr.place_rulers is replaced in the executed source by a version that places
  nx x ny grids of spacing --site_nm with random position / orientation, every grid site present with probability p (at least one site per
  structure); --n_rulers is set so that the expected ON density is --dens. 'ruler' in sites.npz = the structure; ax / ay are not meaningful.
usage: CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/r24_sim.py <nr40|nr20> <out_dir> [n_frames] [seed] [sim_nr options] [--grid 3,4 --occ 0.45]"""
import json, os, sys
ND = '/home/jc/simcode2/out/nanoruler/'; HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH = dict(nr40=[0.86, 0.93, 0.89, 0.91, 0.94, 0.95], nr20=[0.80, 0.83, 0.85, 0.80, 0.85, 0.85])   # stored steps 0-5
DEF = dict(nr40=dict(site_nm=40.0, dens=1.65e-4, win_med=2000.0, frac3=0.75, min_dist=300.0, drift_nm=6.0),
           nr20=dict(site_nm=20.0, dens=4.2e-4, win_med=1790.0, frac3=0.75, min_dist=300.0, drift_nm=12.0))
mv, out = sys.argv[1], os.path.abspath(sys.argv[2]); rest = sys.argv[3:]
GRID = None; OCC = 1.0
if '--grid' in rest: i = rest.index('--grid'); GRID = tuple(int(v) for v in rest[i + 1].split(',')); del rest[i:i + 2]
if '--occ' in rest: i = rest.index('--occ'); OCC = float(rest[i + 1]); del rest[i:i + 2]
base = os.path.dirname(out); os.makedirs(base, exist_ok=True)
C = json.load(open(ND + f'calib_{mv}.json')); C['camera']['maps'] = ND + f'camera_maps_{mv}.npz'; C['pattern']['depth'] = DEPTH[mv]
C['source'] += f' | r24_sim.py: pattern depth = paper estimate per step {DEPTH[mv]}, absolute camera-map path'
cs = f'{base}/calib_src_{mv}.json'; json.dump(C, open(cs, 'w'), indent=1)
P = json.load(open(ND + 'nr80_preset.json')); P['common']['k'] = C['pattern']['k_rad_per_px']; P['common']['cam'] = C['camera']['maps']
ps = f'{base}/preset_{mv}.json'; json.dump(P, open(ps, 'w'), indent=1)
argv = ['sim_nr.py', out] + [a for a in rest if not a.startswith('--') and rest.index(a) < 2 and a.lstrip('-').isdigit()]
opts = rest[len(argv) - 2:]
for k, v in DEF[mv].items():
    if '--' + k not in opts: opts += ['--' + k, str(v)]
if '--calib' not in opts: opts += ['--calib', cs]
if '--preset_json' not in opts: opts += ['--preset_json', ps]
if GRID and '--n_rulers' not in opts:
    g = lambda k, d: float(opts[opts.index('--' + k) + 1]) if '--' + k in opts else d
    duty = (g('on_flong', 0.2) * g('on_tlong', 160.0) + (1 - g('on_flong', 0.2)) * g('on_tshort', 6.0)) * g('ev_per_site_30k', 40.0) / 30000.0
    nsg = GRID[0] * GRID[1]; mean_sites = nsg * OCC / (1 - (1 - OCC) ** nsg)
    opts += ['--n_rulers', str(max(1, int(round(g('dens', 0) * g('H', 256) * g('W', 256) / duty / mean_sites))))]
sys.argv = argv + opts; print('r24_sim:', ' '.join(sys.argv), 'grid', GRID, 'occ', OCC, flush=True)
src = open(os.path.join(HERE, 'sim_nr.py')).read()
old = "assert os.path.abspath(out).startswith('/home/jc/simcode2/out/nanoruler/sim'), 'outputs only under out/nanoruler/sim/'"
assert old in src, 'sim_nr.py changed'
src = src.replace(old, "assert os.path.abspath(out).startswith('/home/jc/simcode2/out/nanoruler/nr'), 'outputs only under out/nanoruler/nr40|nr20'")
if GRID:
    assert 'def place_rulers(rng, a, n_target):' in src
    src = src.replace('def place_rulers(rng, a, n_target):', 'def place_rulers_line(rng, a, n_target):')
    src = src.replace('def sfield(n, H, W, sig, g):', """def place_rulers(rng, a, n_target):
    # r24_sim.py: nx x ny grid structures (spacing site_nm), each site present with probability occ; same return format as the line version
    H, W = a.H, a.W; d = a.site_nm / a.pixel_nm; md = a.min_dist / a.pixel_nm; nx, ny = %d, %d; occ = %r
    gx, gy = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2); g0 = np.c_[gx.ravel(), gy.ravel()] * d
    S = []; R = []
    for it in range(400000):
        if len(R) >= n_target: break
        th = rng.uniform(0, math.pi); c = np.array([rng.uniform(0, W - 1), rng.uniform(0, H - 1)])
        keep = rng.rand(nx * ny) < occ
        if not keep.any(): keep[rng.randint(nx * ny)] = True
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pts = c[None] + g0[keep] @ rot.T
        if (pts[:, 0] < a.margin).any() or (pts[:, 0] > W - 1 - a.margin).any() or (pts[:, 1] < a.margin).any() or (pts[:, 1] > H - 1 - a.margin).any(): continue
        if S and np.min(np.hypot(*(np.concatenate([s[0] for s in S])[:, None] - pts[None]).transpose(2, 0, 1))) < md: continue
        S.append((pts, len(R))); R.append((c, th, len(pts)))
    assert len(R) == n_target, 'could not place the structures (density too high for min_dist)'
    return S, R


def sfield(n, H, W, sig, g):""" % (GRID[0], GRID[1], OCC))
sys.path.insert(0, HERE)
exec(compile(src, os.path.join(HERE, 'sim_nr.py'), 'exec'), {'__name__': '__main__', '__file__': os.path.join(HERE, 'sim_nr.py')})
