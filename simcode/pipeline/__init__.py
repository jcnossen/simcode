"""SIMCODE v2 automatic per-acquisition pipeline: config -> camera, spots, tracks, in-situ PSF, model, processing, report.
Design: docs/AUTO_PIPELINE.md. Entry points: python -m simcode.pipeline <config>, simcode.pipeline.LocalizationPipeline;
configs in code: simcode.pipeline.ConfigDict (simcode/util/config_dict.py), examples in simcode/pipeline/configs/*.py."""


def __getattr__(name):
    if name == 'LocalizationPipeline':
        from simcode.pipeline.localization_pipeline import LocalizationPipeline
        return LocalizationPipeline
    if name == 'ConfigDict':
        from simcode.util.config_dict import ConfigDict
        return ConfigDict
    raise AttributeError(name)
