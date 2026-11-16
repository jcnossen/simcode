"""Cache of the Cell11 channel crops (the raw dcimg files are on a saturated spinning disk: each file is read once) and
a reader of the cache for the processing driver.

  python analysis/fourpi_plain/p4cache.py --work <plain-4Pi work folder> --data <extracted Zenodo folder> [--files 0 1 2 3]
      -> <work>/p4psf/cache/Cell11_<file>_crops.npy: [n, 4, 168, 192] uint16 camera ADU (the channel crops of the
         fourpi_plain config, exactly what ChannelReader.read returns, as uint16)

  cache_dir(work), cache_path(fi, work): the cache of the plain-4Pi work folder work;  acq(work, data, cfg): the Cell11
  acquisition of a config file (configs(work, data); data = the folder of the extracted Zenodo data)
  CachedReader(path).read(f0, f1) -> [n, 4, H, W] float32 ADU (a drop-in for ChannelReader: .n, .shape, .read)
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

CFG = os.path.join(os.path.dirname(__file__), '..', '..', 'simcode', 'pipeline', 'configs', 'fourpi_plain.py')


def cache_dir(work):
    """the crop cache of the plain-4Pi work folder work"""
    return os.path.join(work, 'p4psf', 'cache')


def cache_path(fi, work, cell='Cell11'):
    return os.path.join(cache_dir(work), f'{cell}_{fi:03d}_crops.npy')


def acq(work, data, cfg=CFG):
    """(resolved config, setup) of the Cell11 acquisition of a config file (default fourpi_plain.py; work = the plain-4Pi
    work folder, data = the folder of the extracted Zenodo data)"""
    from simcode.pipeline import acquisition_config as AC
    from simcode.setups import get
    L = AC.load_all(cfg, dict(work=work, data=data))
    c = [u for u in L if u['name'].endswith('Cell11')][0]
    return c, get(c['setup'])


class CachedReader:
    def __init__(self, path):
        self.a = np.load(path, mmap_mode='r')
        self.n = self.a.shape[0]
        self.shape = self.a.shape[1:]
        self.parts = [path]

    def read(self, f0, f1):
        return np.asarray(self.a[f0:f1], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--work', required=True, help='plain-4Pi work folder')
    p.add_argument('--data', required=True, help='folder of the extracted Zenodo data (zenodo_4022827/extracted)')
    a = p.parse_args()
    c, S = acq(a.work, a.data)
    os.makedirs(cache_dir(a.work), exist_ok=True)
    for fi in a.files:
        fo = cache_path(fi, a.work)
        if os.path.exists(fo):
            print(fo, 'exists'); continue
        rd = S.reader(c, S.movie_file(c, fi))
        t0 = time.time()
        out = np.lib.format.open_memmap(fo + '.tmp.npy', mode='w+', dtype=np.uint16, shape=(rd.n, *rd.shape))
        for f0 in range(0, rd.n, 500):
            f1 = min(f0 + 500, rd.n)
            x = rd.read(f0, f1)
            assert x.min() >= 0 and x.max() < 65536
            out[f0:f1] = x.astype(np.uint16)
        out.flush(); del out
        os.replace(fo + '.tmp.npy', fo)
        print(f'file {fi}: {rd.n} frames -> {fo} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
