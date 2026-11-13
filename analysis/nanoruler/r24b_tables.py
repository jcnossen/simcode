"""Baseline run of r24b_run.py -> tables, pattern json and sanity report for nr40 / nr20 (CPU only): bl_tables.py with the
movie-specific constants adapted. bl_tables.py is NOT edited and not copied: its source is read, patched IN MEMORY and
executed. Exact changes (each textual replacement is asserted to occur the stated number of times):

  text patches of the bl_tables.py source
    1. "'/results/sim4_1_reord/'" -> "'/results/movie_reord/'"            (1 x; result directory = name of the movie file)
    2. " / 5000" -> " / NCYC"                                              (4 x; localizations per cycle: 6666 / 16666 cycles)
    3. "t['frame'][i] // 1000, 4)" -> "t['frame'][i] * 5 // NCYC, 4)"      (1 x; time bins of the matched-pair offsets =
       fifths of the movie instead of blocks of 1000 cycles)
    4. "median_offset_px_per_1000_cycles" -> "median_offset_px_per_fifth_of_movie" (2 x) and
       "per 1000 cycles median offset px" -> "per fifth of the movie median offset px" (1 x; labels of 3)
  module globals set after executing the patched source
    5. OUT = out/nanoruler/<movie>/ref/ (run directory = OUT/<tag>/, tables next to it)
    6. PUB = the movie's published results folder (g2d-nodriftc.hdf5, simflux-nodriftc.hdf5, report.txt: same names)
    7. CALIB = out/nanoruler/calib_<movie>.json
    8. NCYC
Unchanged: table format, matching (mutual nearest neighbours per cycle within --rmatch px), automatic half-pixel rule,
pattern conventions, the report's fixed depth 0.85 (true for both movies).

usage: python analysis/nanoruler/r24b_tables.py nr40 --tag bl_run [--suffix ''] [--log <run log>]
outputs in out/nanoruler/<movie>/ref/: tab_bl_smlm<suffix>.npz, tab_bl_simflux<suffix>.npz, pattern_bl<suffix>.json,
  pattern_bl<suffix>_phase_offset.npy, sanity_bl<suffix>.json / .txt"""
import sys
REPO = '/home/jc/simcode2'
SRC = f'{REPO}/analysis/nanoruler/bl_tables.py'
B = '/mnt/data/SIMFLUX/data/fig2/'
SETS = dict(nr40=(B + '40nm-panel-q-r/results/3Phase_3_MMStack_Pos0.ome/', 6666), nr20=(B + '20nm-panel-s-t/results/1_2_MMStack_Pos0.ome/', 16666))

if __name__ == '__main__':
    movie = sys.argv[1]; assert movie in SETS
    src = open(SRC).read()
    for old, new, n in (("'/results/sim4_1_reord/'", "'/results/movie_reord/'", 1), (' / 5000', ' / NCYC', 4),
                        ("t['frame'][i] // 1000, 4)", "t['frame'][i] * 5 // NCYC, 4)", 1),
                        ('median_offset_px_per_1000_cycles', 'median_offset_px_per_fifth_of_movie', 2),
                        ('per 1000 cycles median offset px', 'per fifth of the movie median offset px', 1)):
        assert src.count(old) == n, (old, src.count(old)); src = src.replace(old, new)
    ns = dict(__name__='bl_tables_patched', __file__=SRC)
    exec(compile(src, SRC, 'exec'), ns)
    ns.update(OUT=f'{REPO}/out/nanoruler/{movie}/ref/', PUB=SETS[movie][0], CALIB=f'{REPO}/out/nanoruler/calib_{movie}.json', NCYC=SETS[movie][1])
    sys.argv = [SRC] + sys.argv[2:]
    ns['main']()
