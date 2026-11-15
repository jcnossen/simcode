"""Interface of a set-up (simcode/setups/__init__.py). A Setup supplies to the one LocalizationPipeline:

  name                    the config value of 'setup'
  defaults()              its own config section(s) {key: defaults} (added to the schema of acquisition_config.py)
  validate(c)             checks of a resolved config
  stages(c)               the stage names in run order (simcode/pipeline/localization_pipeline.py METHODS)
  st_<stage>(pipe)        a stage implemented by the set-up (else the pipeline's own _st_<stage>); returns its summary
  movie_files(c)          the movie files of the acquisition (one movie may be several files)
  reader(c, path)         channel reader of one file: read(f0, f1) -> [n, C, H, W] ADU (simcode/channels/layout.py)
"""
import glob


class Setup:
    name = None

    def defaults(self):
        return {}

    def validate(self, c):
        pass

    def stages(self, c):
        raise NotImplementedError

    def movie_files(self, c):
        fs = sorted(glob.glob(c['movie']))
        return fs if fs else [c['movie']]

    def reader(self, c, path):
        from simcode.io.movie import open_movie
        from simcode.channels.layout import ChannelLayout, ChannelReader
        mv = open_movie(path)
        return ChannelReader(mv, ChannelLayout.from_config(c, mv.shape[-2:]))
