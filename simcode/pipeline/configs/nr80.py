"""nr80.yaml as a Python config (same resolved config, without the yaml's `process.pass2: false`, a key that is no longer
accepted since 2026-11-14: pass 2 always runs).
Original SIMFLUX setup (Pockels-cell orientation switch, mechanical phase steps, interleaved orientation order),
nanoruler movie sim4_1 (80 nm), Hamamatsu ORCA-Flash4.0 V2, 65 nm px. Reference: out/nanoruler (calib_nr80.json,
camera_maps_nr80.npz, tracks_nr80.npz; model nr80 out/nanoruler/models/netset_nr80.json).

  python -m simcode.pipeline simcode/pipeline/configs/nr80.py
"""
from simcode.util.cfgdict import ConfigDict

NANORULER = '/home/jc/simcode2/out/nanoruler'
PIXEL_UM = 0.065
PITCH_PX = 3.38                            # pattern pitch of this setup (220 nm)

config = ConfigDict(
    name='nr80',
    movie='/mnt/data/SIMFLUX/data/fig2/80nm-panel-a-e/sim4_1_MMStack_Pos0.ome.tif',
    out='/mnt/data/dmdsf/simcode2_out/auto/nr80',
    camera=dict(gain_e_per_adu=0.45,       # ASSUMED (paper processing; not measurable: background ~3 ADU above offset)
                offset_adu=100.0,          # ASSUMED
                map_frames=6000),
    optics=dict(pixel_um=PIXEL_UM, wavelength_um=0.68),
    pattern=dict(
        frame_order='interleaved',         # acquisition index = frame % 6, orientation = index % 2, step = index // 2
        cache=f'{NANORULER}/raw/sim4_1_reord.tif',      # existing blocked-order cache (verified by stage 'order')
        pitch_px=[round(PITCH_PX - 0.48, 2), round(PITCH_PX + 0.52, 2)]),            # search range [2.9, 3.9]
    spots=dict(maxframe=30000,
               thr=5.0,                    # the validated run (reference tracks of out/nanoruler: thr 5); default 3.5
               track_roi=15, isolation_px=10.0),
    psf=dict(init=['zero', 'f0.1', 'f0.2', 'f0.3'],     # no bead stack of this setup; multi-start, lowest base-fit NLL wins
             readnoise_e=1.6),
    process=dict(model=f'{NANORULER}/models/netset_nr80.json', crop=[5, 5, 390], chunk=1500, det_batch=8,
                 cycle_aligned_only=True),
    gpus=[1, 2, 0],
)
