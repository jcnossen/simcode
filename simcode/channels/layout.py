"""Channel layout of an acquisition: how one camera frame (or a group of interleaved pages) is cut into the detection
channels. The acquisition config lists the channels (simcode/pipeline/acquisition_config.py, key 'channels'):

  channels: [{name, crop: [x0, y0, w, h], mirror: none | lr | ud | both, page: 0, psf: null, weight: 1.0}, ...]

  crop    the channel's window in the camera frame (px; x = column, y = row); all channels have the same w x h.
          null: the whole frame (one channel).
  mirror  the channel crop is flipped (lr: left-right, ud: up-down, both) so that all channels overlay the reference
          channel up to the registration map (the 4Pi channels 2 and 3 are mirrored by the optics).
  page    for movies that store the channels as interleaved pages (page k of frame f = stack page f * pages + k;
          the two-colour TIFFs): the page of this channel; 0 when every frame is one page.
  psf     a per-channel PSF file (two-colour, biplane); null when the set-up has one multi-channel PSF (4Pi).
  weight  weight of the channel in the registered channel sum of the detector (1.0).
Channel 0 is the reference channel of the registration (simcode/channels/registration.py).

  ChannelLayout.from_config(cfg)   the layout of a resolved acquisition config (one channel if 'channels' is null)
  layout.split(a)                  camera frames [n, H, W] (or [n, pages, H, W]) -> channel crops [n, C, h, w] float32
  ChannelReader(reader, layout)    any movie reader (simcode/io/movie.py) -> read(f0, f1) = channel crops [n, C, h, w]
"""
from dataclasses import dataclass
import numpy as np

MIRRORS = ('none', 'lr', 'ud', 'both')


@dataclass
class Channel:
    name: str = 'c0'
    crop: tuple = None          # (x0, y0, w, h) px in the camera frame; None = the whole frame
    mirror: str = 'none'
    page: int = 0
    psf: str = None
    weight: float = 1.0

    def __post_init__(self):
        if self.mirror not in MIRRORS: raise ValueError(f'channel {self.name}: mirror must be one of {MIRRORS}, not {self.mirror!r}')
        if self.crop is not None:
            self.crop = tuple(int(v) for v in self.crop)
            if len(self.crop) != 4 or min(self.crop[2:]) <= 0 or min(self.crop[:2]) < 0:
                raise ValueError(f'channel {self.name}: crop must be [x0, y0, w, h] with x0, y0 >= 0 and w, h > 0, not {self.crop}')

    def cut(self, a):
        """[..., H, W] -> this channel's crop [..., h, w] (mirrored), a view"""
        if self.crop is not None:
            x0, y0, w, h = self.crop
            if y0 + h > a.shape[-2] or x0 + w > a.shape[-1]:
                raise ValueError(f'channel {self.name}: crop {self.crop} outside the frame {a.shape[-2:]}')
            a = a[..., y0:y0 + h, x0:x0 + w]
        if self.mirror in ('lr', 'both'): a = a[..., ::-1]
        if self.mirror in ('ud', 'both'): a = a[..., ::-1, :]
        return a


class ChannelLayout:
    def __init__(self, channels, frame_shape=None):
        """channels: list of Channel (or dicts with the Channel fields); frame_shape (H, W) of the camera frame (needed
        only for a channel without crop)"""
        self.channels = [c if isinstance(c, Channel) else Channel(**c) for c in channels]
        if not self.channels: raise ValueError('a channel layout needs at least one channel')
        self.pages = max(c.page for c in self.channels) + 1
        sizes = {(c.crop[2], c.crop[3]) if c.crop is not None else tuple(frame_shape[::-1]) if frame_shape else None for c in self.channels}
        if len(sizes) != 1: raise ValueError(f'all channel crops must have the same size, got {sorted(map(str, sizes))}')
        s = sizes.pop()
        self.W, self.H = (s if s is not None else (None, None))
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names): raise ValueError(f'channel names must be unique: {names}')

    @classmethod
    def from_config(cls, cfg, frame_shape=None):
        ch = cfg.get('channels')
        if not ch: return cls([Channel()], frame_shape)
        return cls([dict(c, name=c.get('name') or f'c{i}') for i, c in enumerate(ch)], frame_shape)

    @property
    def C(self):
        return len(self.channels)

    @property
    def weights(self):
        return [c.weight for c in self.channels]

    def split(self, a):
        """camera frames [n, H, W] (pages == 1) or the interleaved pages [n * pages, H, W] / [n, pages, H, W] ->
        [n, C, h, w] float32"""
        a = np.asarray(a)
        if self.pages > 1:
            if a.ndim == 3: a = a.reshape(-1, self.pages, *a.shape[-2:])
            parts = [c.cut(a[:, c.page]) for c in self.channels]
        else:
            if a.ndim == 4: a = a[:, 0]
            parts = [c.cut(a) for c in self.channels]
        return np.stack(parts, -3).astype(np.float32)

    def to_config(self):
        return [dict(name=c.name, crop=list(c.crop) if c.crop is not None else None, mirror=c.mirror, page=c.page, psf=c.psf,
                     weight=c.weight) for c in self.channels]


class ChannelReader:
    """a movie reader whose read(f0, f1) returns the channel crops [n, C, h, w] (float32, camera ADU). The wrapped reader
    supplies raw(f0, n) (uint16 frames; DcimgMovie) or read(f0, f1); with pages > 1 its frames are the interleaved pages."""
    def __init__(self, reader, layout):
        self.reader, self.layout = reader, layout
        self.n = reader.n // layout.pages
        self.parts = getattr(reader, 'parts', None)
        self.shape = (layout.C, layout.H, layout.W)

    def read(self, f0, f1):
        p = self.layout.pages
        if hasattr(self.reader, 'raw'):
            a = self.reader.raw(f0 * p, (f1 - f0) * p)
        else:
            a = self.reader.read(f0 * p, f1 * p)
        return self.layout.split(a)
