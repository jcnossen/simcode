"""The one place for names that only exist in stored data: what earlier checkpoints and run records contain, mapped to the
present names when they are read. Nothing else in the package knows an old name (file names of models, modules, options and
config keys have no aliases; renames: docs/LAYOUT.md "Renamed").

Used only by the readers of stored data:
  EMITTER_ARCH_TAGS     emitter.model.load: the architecture tag args['arch'] of the pass-1 emitter-network checkpoints trained
                        before 2026-11-15 ('sc4n', a development name of the same network; every shipped model has pass-1
                        checkpoints with this tag, rewriting them would change their recorded sha256)
  FOURPI_ARCH_TAGS      emitter.fourpi.load: the 4Pi checkpoints trained before 2026-11-15 (head4pi_v10 / v11: 'w', the
                        FourPiPrior checkpoints v2 .. v8 / v3L: no tag); both with T = 6 and the default zmax / vread
  RUN_RECORD_FORMAT1_*  util.runconfig.upgrade: run records 'simcode.run_config/1' (written before 2026-11-15) named the
                        model files det / head / p2_det / p2_head / p2_cal (+ sig_u_p1 / sig_u_p2) and two fixed values
                        head_batch / compile_head
"""
EMITTER_ARCH_TAGS = {'sc4n': 'emitter_net'}
FOURPI_ARCH_TAGS = {'w': 'FourPiEmitterNet', None: 'FourPiPrior'}
RUN_RECORD_FORMAT1_KEYS = dict(det='detector', head='emitter', p2_det='detector_pass2', p2_head='emitter_pass2', p2_cal='cal_pass2',
                               p2_cal_sim='cal_sim_pass2', p2_cal_real='cal_real_pass2', sig_u_p1='sig_u_pass1', sig_u_p2='sig_u_pass2')
RUN_RECORD_FORMAT1_FIXED = dict(head_batch='emitter_batch', compile_head='compile_emitter')
