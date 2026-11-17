"""sigma study (2026-11-17): a view of a pipeline run with tables built WITHOUT the model's sigma rescale g(u) (the posterior sigma as
the solve reports it): <out>/ = symbolic links to every file of <run> except the tables, + locs_p1.hdf5 / locs_p2.hdf5 from
simcode.table.build_table(sig_u=None). For the figure scripts (they read the run directory).
usage: python nog_view.py <run dir> <out dir>"""
import glob, os, sys
from simcode import table as SP

run, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
for f in glob.glob(f'{run}/*'):
    b = os.path.basename(f)
    if b.startswith('locs_'): continue
    if not os.path.exists(f'{out}/{b}'): os.symlink(os.path.abspath(f), f'{out}/{b}')
for tag, nm in (('est', 'locs_p1'), ('est2', 'locs_p2')):
    T, info = SP.build_table(run, tag, sig_u=None)
    info['note'] = 'table without the sigma rescale g(u) (analysis/sigma_joint/nog_view.py)'
    SP.write_table(T, info, f'{out}/{nm}')
    print(tag, info['n_main'], 'rows in the 12 nm selection')
