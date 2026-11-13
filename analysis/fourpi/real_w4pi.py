"""4Pi-SIMCODE (learned-window network pipeline on raw dcimg): the code is simcode/fourpi/run.py; this file keeps the old
command working.
Usage: CUDA_VISIBLE_DEVICES=1 python analysis/fourpi/real_w4pi.py --cell NPC --files 10 --head v10 [--nframes 3600]
       (= python -m simcode.fourpi.run ...; whole movies on several GPUs: analysis/fourpi/run_full_w4pi.sh)
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.fourpi.run import main

if __name__ == '__main__':
    main()
