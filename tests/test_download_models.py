"""
CPU test of simcode.download_models on a local file:// copy of a (fake) release asset: download, sha256 check of the zip and
of every member, unpack into a models folder, skip when present, refuse a wrong checksum / unexpected members.
Run: python tests/test_download_models.py (or pytest tests)
"""
import hashlib, os, sys, tempfile, zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode import download_models as DM

H = lambda b: hashlib.sha256(b).hexdigest()


def _fake_release(d):
    files = {'mx/detector_pass1.pt': b'det' * 1000, 'mx/emitter_pass1.pt': b'em' * 1000}
    z = os.path.join(d, 'simcode_model_mx.zip')
    with zipfile.ZipFile(z, 'w') as Z:
        for k, v in files.items(): Z.writestr(k, v)
    return z, files


def test_download_file_url():
    with tempfile.TemporaryDirectory() as d:
        rel = os.path.join(d, 'release'); os.makedirs(rel)
        z, files = _fake_release(rel)
        A, M = dict(DM.ASSETS), dict(DM.MEMBERS)
        try:
            DM.ASSETS['mx'] = (os.path.basename(z), H(open(z, 'rb').read()))
            DM.MEMBERS['mx'] = {k: H(v) for k, v in files.items()}
            dest = os.path.join(d, 'models')
            out = DM.install('mx', dest, base_url='file://' + rel)
            assert sorted(out) == sorted(os.path.join(dest, k) for k in files)
            assert all(open(os.path.join(dest, k), 'rb').read() == v for k, v in files.items())
            assert DM.present('mx', dest) and not [f for f in os.listdir(dest) if f.startswith('.download_')]
            os.remove(z)                                                 # present: no download (the asset is gone)
            DM.install('mx', dest, base_url='file://' + rel)
            _fake_release(rel)
            DM.ASSETS['mx'] = (os.path.basename(z), '0' * 64)            # wrong zip checksum
            try: DM.install('mx', os.path.join(d, 'm2'), base_url='file://' + rel); raise AssertionError('accepted a wrong sha256')
            except RuntimeError as e: assert 'sha256' in str(e)
            assert not os.path.exists(os.path.join(d, 'm2', 'mx'))
            DM.ASSETS['mx'] = (os.path.basename(z), H(open(z, 'rb').read()))
            DM.MEMBERS['mx'] = {'mx/other.pt': H(b'')}                   # unexpected members
            try: DM.install('mx', os.path.join(d, 'm3'), base_url='file://' + rel); raise AssertionError('accepted other members')
            except RuntimeError as e: assert 'members' in str(e)
            DM.MEMBERS['mx'] = {k: H(v) for k, v in files.items()}
            assert DM.main(['--set', 'mx', '--dest', os.path.join(d, 'm4'), '--base-url', 'file://' + rel]) == 0 and DM.present('mx', os.path.join(d, 'm4'))
        finally:
            DM.ASSETS.clear(); DM.ASSETS.update(A); DM.MEMBERS.clear(); DM.MEMBERS.update(M)


def test_asset_table():
    assert DM.TAG and DM.BASE_URL.endswith('/releases/download/' + DM.TAG)
    assert set(DM.ASSETS) == set(DM.MEMBERS) >= {'m1809', 'g13', 'pretrain'}
    assert all(len(h) == 64 for _, h in DM.ASSETS.values()) and all(len(h) == 64 for M in DM.MEMBERS.values() for h in M.values())
    from simcode import config
    for s in ('m1809', 'g13'):                                           # the members are the model's networks
        S = config.model_set(s)
        assert {S[k] for k in ('detector', 'emitter', 'detector_pass2', 'emitter_pass2')} == set(DM.MEMBERS[s])
    assert set(DM.MEMBERS['pretrain']) == {config.PRETRAIN_DETECTOR, config.PRETRAIN_EMITTER}


if __name__ == '__main__':
    test_download_file_url(); test_asset_table(); print('ok')
