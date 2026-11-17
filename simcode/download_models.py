"""
Download the network weights of the example models (they are not in git): GitHub release assets of the simcode
repository, one zip per model set, verified by sha256 and unpacked into the models folder.

  python -m simcode.download_models                         # m1809 and g13 into <repo>/models
  python -m simcode.download_models --set pretrain          # the pretraining networks (needed to train a new model)
  python -m simcode.download_models --set all --dest /path/to/models
  python -m simcode.download_models --base-url file:///path/to/release_assets   # a local / mirrored copy of the assets

Sets: m1809 (in-situ PSF of the 1809/2 movie), g13 (Gaussian PSF sigma 1.3 px): the four networks of the model,
<dest>/<set>/{detector,emitter}_pass{1,2}.pt; pretrain: <dest>/pretrain_detector_pass1.pt, pretrain_emitter_pass1.pt (the
starting point of a new pass-1 emitter network and the default detector of the emitter trainer, simcode.config.PRETRAIN_*).
The calibrations, PSFs and model.json of the sets are in git (models/<set>/). A file that is already present with the
right sha256 is not downloaded again (--force: download anyway). --dest other than <repo>/models: pass the model as a
folder, e.g. python -m simcode.run --model <dest>/m1809 (the folder needs the files of models/m1809 from git as well).
"""
import argparse, hashlib, os, shutil, sys, tempfile, urllib.request, zipfile

TAG = 'models-v2'          # 2026-11-17: the emitter networks with the joint prior / intensity covariance (v1: before)
BASE_URL = f'https://github.com/qnano/simcode/releases/download/{TAG}'
MIRROR_URL = f'https://github.com/jcnossen/simcode/releases/download/{TAG}'   # the same files (mirror)
# asset -> sha256 of the zip; members (relative to the models folder) -> sha256 of the file
ASSETS = {
    'm1809': ('simcode_model_m1809.zip', 'd0d8da735fa24202f4c5f1e4c8e7cab12f542165e37551f67f78acc992fe2d84'),
    'g13': ('simcode_model_g13.zip', '66e4abfafbd17af98b2919aed83f64cc2b9f31f9b5dd90ec2a69bbeb78d26d29'),
    'pretrain': ('simcode_pretrain.zip', 'f22fd0e73dbf926d9fe93a8e36a3c00c26b00fe526c1239e8163df07c97fe7d6'),
}
MEMBERS = {
    'm1809': {'m1809/detector_pass1.pt': '18bd16f99613a67b3c73d543833397b57deeb278b5568083d15731231428ca04',
              'm1809/emitter_pass1.pt': '92ab26c0fd0799cd0c76088fa9763065199dac3ee00d19a7005b53b4f48514d4',
              'm1809/detector_pass2.pt': '92d4487a7fe8ced56aec492e2972c8a00bd95e4cf43515714517df85bba51a07',
              'm1809/emitter_pass2.pt': '5329890b80311d0288b9a5e305621fdb0caf7df9297320fe4413a79ce441a1cb'},
    'g13': {'g13/detector_pass1.pt': '7a6130f3d34ca111bd8d79b0be3e7caaf75f4ce41203ac249b43762f5884d30c',
            'g13/emitter_pass1.pt': '17be7aa32028e232598b10f70d3bf7f7ba823c8e5ea77d6d4a9888d4e676b5b6',
            'g13/detector_pass2.pt': 'd1037970932898d304ed7ffbd6867b42928f08e047a96ade0c13d5d52de3e2ea',
            'g13/emitter_pass2.pt': '0e23e4a66f8e6e06232a19bdc597446a6eaea65c1ed7eaaed932323a9474d085'},
    'pretrain': {'pretrain_detector_pass1.pt': '3bfc6c64a75e63d8c7a7631d07810cb6a285453a6bc7578965ac1213514b756e',
                 'pretrain_emitter_pass1.pt': '238cea16191109f55aa6053f31aa01c9f81b97f4f2e2f08281c478d6c6c9fea6'},
}
DEFAULT_SETS = ('m1809', 'g13')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def present(s, dest):
    """True when every file of set s is in dest with its sha256"""
    return all(os.path.isfile(os.path.join(dest, m)) and sha256(os.path.join(dest, m)) == h for m, h in MEMBERS[s].items())


def fetch(url, out):
    """url (https:// or file://) -> file out, with a progress line"""
    with urllib.request.urlopen(url) as r, open(out, 'wb') as f:
        n = int(r.headers.get('Content-Length') or 0); got = 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); got += len(b)
            if n: print(f'\r  {got / 1e6:.1f} / {n / 1e6:.1f} MB', end='', flush=True)
        if n: print()


def install(s, dest, base_url=BASE_URL, force=False):
    """download, verify and unpack set s into dest; returns the list of installed files"""
    name, zsha = ASSETS[s]
    if not force and present(s, dest):
        print(f'{s}: already present in {dest}'); return [os.path.join(dest, m) for m in MEMBERS[s]]
    os.makedirs(dest, exist_ok=True)
    url = f'{base_url.rstrip("/")}/{name}'
    print(f'{s}: {url}')
    with tempfile.TemporaryDirectory(dir=dest, prefix='.download_') as tmp:
        z = os.path.join(tmp, name)
        fetch(url, z)
        h = sha256(z)
        if h != zsha: raise RuntimeError(f'{name}: sha256 {h}, expected {zsha} (incomplete download or another release than {TAG})')
        with zipfile.ZipFile(z) as Z:
            names = [i.filename for i in Z.infolist() if not i.is_dir()]
            if sorted(names) != sorted(MEMBERS[s]): raise RuntimeError(f'{name}: members {names}, expected {sorted(MEMBERS[s])}')
            out = []
            for m in names:
                t = os.path.join(tmp, 'x', m)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                with Z.open(m) as src, open(t, 'wb') as dst: shutil.copyfileobj(src, dst)
                if sha256(t) != MEMBERS[s][m]: raise RuntimeError(f'{name}:{m}: sha256 mismatch')
                f = os.path.join(dest, m); os.makedirs(os.path.dirname(f), exist_ok=True)
                os.replace(t, f); out.append(f)
    print(f'{s}: {len(out)} files -> {dest}')
    return out


def main(argv=None):
    from simcode import config
    ap = argparse.ArgumentParser(prog='python -m simcode.download_models', description=__doc__.strip().split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--set', nargs='+', default=list(DEFAULT_SETS), help=f'sets: {", ".join(ASSETS)} or all (default: {" ".join(DEFAULT_SETS)})')
    ap.add_argument('--dest', default=config.MODELS, help='models folder to unpack into (default: the repository\'s models/)')
    ap.add_argument('--base-url', default=BASE_URL, help=f'where the assets are (default: the GitHub release {TAG} of qnano/simcode; the same files are mirrored at {MIRROR_URL})')
    ap.add_argument('--force', action='store_true', help='download even if the files are present')
    a = ap.parse_args(argv)
    sets = list(ASSETS) if 'all' in a.set else a.set
    bad = [s for s in sets if s not in ASSETS]
    if bad: ap.error(f'unknown set(s) {bad}; choose from {list(ASSETS)} or all')
    for s in sets: install(s, os.path.abspath(os.path.expanduser(a.dest)), a.base_url, a.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
