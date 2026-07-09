"""
SFLocalizationModelExpN
=======================

Variant of SFLocalizationModel with **exp-scaled output activations** for the
per-frame intensities (N0..N5) and the per-spot background (bg).  Everything
else -- the U-Net topology, the loss, the training loop -- is identical to the
base class, so results can be compared directly.

Motivation
----------
The base ``SFLocalizationModel`` puts bg and every N_k through tanh(x) * scale.
Both quantities are physically non-negative and the training data is either
log-normal / log-uniform (intensities) or bounded positive (bg), so half of the
tanh output range is wasted on unreachable negative values.  Concretely for the
current shipped model:

  * ``max_bg=100`` but training bg range is [0.5, 20] -> the target sits in the
    lowest 10% of the tanh output range, giving the network almost no gradient
    budget in the useful region.
  * ``max_intensity=20000`` but training intensity range is [50, 10000] -> same
    story, half the output range unreachable.

Exp activation removes the wasted range: ``exp(raw) * scale`` maps R -> (0, inf).
The network directly learns log-photons which matches the log-normal / log-uniform
training distribution and gives full gradient control across every decade.

Output pipeline (compared to SFLocalizationModel)
-------------------------------------------------
For each per-pixel output feature:

  feature       | base model                 | ExpN
  --------------+----------------------------+----------------------------------
  prob (p)      | sigmoid(x)                 | sigmoid(x)                    [unchanged]
  x, y (position| tanh(x) * xy_scale         | tanh(x) * xy_scale            [unchanged]
  bg            | tanh(x) * max_bg           | exp(x) * bg_scale
  N0..N5        | tanh(x) * max_intensity    | exp(x) * intensity_scale
  {p,x,y,...}_sig| sigmoid(x)*scale + 1e-3   | sigmoid(x)*scale + 1e-3       [unchanged]

Raw logits going into exp() are clamped to [-8, 8] for numerical safety.  With
``bg_scale=5, intensity_scale=700`` the reachable output ranges are
[1.7e-3, 15e3] for bg and [0.24, 2.1e6] for intensity -- covers realistic
extremes without either saturating the activation or losing gradient in the
useful region.

Constructor parameters
----------------------
Same as SFLocalizationModel plus:

  bg_scale : float, default 5.0
      Multiplier applied to exp() of the raw bg logit.  Pick ~ median of the
      training bg distribution.
  intensity_scale : float, default 700.0
      Multiplier applied to exp() of the raw N_k logit.  Pick ~ geometric mean
      of the training intensity range (sqrt(intensity_min * intensity_max)).
"""

import numpy as np
import torch.nn as nn

from smlmtorch.nn.sf_model import SFLocalizationModel


class SFLocalizationModelExpN(SFLocalizationModel):
    def __init__(self,
                 num_intensities,
                 xyz_scale,
                 enable3D,
                 bg_scale=5.0,
                 intensity_scale=700.0,
                 **kwargs):

        # ExpN uses exp() * scale for bg and intensity -- these arguments are
        # semantic labels only; the SFLocalizationModel base class still expects
        # max_bg and max_intensity as the output multipliers.  Pass through and
        # override the feature-index lists below so the exp() branch is picked
        # up instead of the default tanh.
        kwargs.setdefault('max_bg', bg_scale)
        kwargs.setdefault('max_intensity', intensity_scale)

        # Compute feature index layout so we can route bg + N_k through exp().
        # Parameter layout: [x, y, (z if 3D), bg, N0..N_{K-1}]
        xyz_dim = 3 if enable3D else 2
        n_main = xyz_dim + 1 + num_intensities  # x, y, [z], bg, N0..
        bg_ix = xyz_dim                          # index of bg within params
        n_ixs = list(range(xyz_dim + 1, xyz_dim + 1 + num_intensities))

        # Output feature layout: [prob, *params, *sigmas].  Index 0 is prob,
        # 1..n_main are the parameter means, n_main+1..2*n_main are the sigmas.
        tanh_features = [1 + i for i in range(xyz_dim)]   # x, y (and z if 3D)
        exp_features = [1 + bg_ix] + [1 + i for i in n_ixs]  # bg + all N_k
        sigma_features = list(1 + n_main + np.arange(n_main))
        sigmoid_features = [0] + list(sigma_features)     # prob + all sigmas

        # These override the LocalizationModel defaults.  Don't let the caller
        # accidentally override them from kwargs -- ExpN has a specific layout.
        kwargs['tanh_features'] = tanh_features
        kwargs['sigmoid_features'] = sigmoid_features
        kwargs['exp_features'] = exp_features

        super().__init__(num_intensities=num_intensities,
                         xyz_scale=xyz_scale,
                         enable3D=enable3D,
                         **kwargs)
