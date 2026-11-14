"""A/B of the pass-1 detector --band option (2026-11-13): run simcode.det.up2 train under the process title 'band-ab <tag>'.
usage: python analysis/detband/train.py <tag> <up2 train args...>"""
import sys
from simcode.util import proctitle
tag = sys.argv[1]
proctitle.set_title('band-ab', tag)
proctitle.auto = lambda name: name          # keep the band-ab title (up2.train calls auto('train-det'))
from simcode.det import up2
up2.train(sys.argv[2:])
