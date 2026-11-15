"""SIMCODE on DMD-SIMFLUX movies (one camera channel): the stages of docs/AUTO_PIPELINE.md, implemented by the pipeline
itself (simcode/pipeline/localization_pipeline.py _st_<stage>). With more than one channel in the config ('channels')
the per-movie channel registration (stage registration, in situ) runs after the camera stage; the processing of a
multi-channel widefield movie (two-colour, biplane) is not part of the pipeline yet."""
from simcode.setups.base import Setup

STAGES = ['order', 'camera', 'spots', 'cammaps', 'tracks', 'beads', 'psf', 'train', 'process', 'report']


class WidefieldSetup(Setup):
    name = 'widefield'

    def stages(self, c):
        if c.get('channels') and len(c['channels']) > 1:
            return ['order', 'camera', 'registration']
        return list(STAGES)
