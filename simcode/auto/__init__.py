"""SIMCODE v2 automatic per-acquisition pipeline: config -> camera, spots, tracks, in-situ PSF, model, processing, report.
Design: docs/AUTO_PIPELINE.md. Entry points: python -m simcode.auto <config>, simcode.auto.LocalizationPipeline;
configs in code: simcode.auto.ConfigDict (simcode/util/cfgdict.py), examples in simcode/auto/configs/*.py."""


def __getattr__(name):
    if name == 'LocalizationPipeline':
        from simcode.auto.pipeline import LocalizationPipeline
        return LocalizationPipeline
    if name == 'ConfigDict':
        from simcode.util.cfgdict import ConfigDict
        return ConfigDict
    raise AttributeError(name)
