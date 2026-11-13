"""SIMCODE v2 automatic per-dataset pipeline: config -> camera, spots, tracks, in-situ PSF, network set, processing, report.
Design: docs/AUTO_PIPELINE.md. Entry points: python -m simcode.auto <config>, simcode.auto.Dataset."""


def __getattr__(name):
    if name == 'Dataset':
        from simcode.auto.pipeline import Dataset
        return Dataset
    raise AttributeError(name)
