"""SIMCODE v2 automatic per-acquisition pipeline: config -> camera, spots, tracks, in-situ PSF, model, processing, report.
Design: docs/AUTO_PIPELINE.md. Entry points: python -m simcode.auto <config>, simcode.auto.Acquisition (old name: Dataset)."""


def __getattr__(name):
    if name in ('Acquisition', 'Dataset'):
        from simcode.auto.pipeline import Acquisition
        return Acquisition
    raise AttributeError(name)
