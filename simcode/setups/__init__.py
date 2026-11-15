"""Set-ups of the automatic pipeline (simcode/pipeline/localization_pipeline.py): what differs between acquisition types
is supplied by a Setup object, selected by the acquisition config key 'setup'; there is one LocalizationPipeline.

  widefield   SIMCODE on DMD-SIMFLUX movies (one channel; the stages of docs/AUTO_PIPELINE.md)        widefield.py
  fourpi      4Pi-SIMFLUX: four interference channels on one camera, dcimg files, the 4Pi PSF, the 4Pi detector and
              emitter network, the 4Pi solve, per-file registration / SIM pattern / 4Pi phase field    fourpi.py

get(name) -> the Setup object; base.py: the interface."""
from simcode.setups.base import Setup


def get(name):
    if name == 'widefield':
        from simcode.setups.widefield import WidefieldSetup
        return WidefieldSetup()
    if name == 'fourpi':
        from simcode.setups.fourpi import FourPiSetup
        return FourPiSetup()
    raise ValueError(f"unknown setup {name!r} (widefield | fourpi)")


NAMES = ('widefield', 'fourpi')
