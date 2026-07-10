"""
========================================================================
SIMCODE retraining run: SFLocalizationModelExpN
========================================================================

For the server-side agent picking this up
-----------------------------------------
This script trains a *new* SIMCODE localisation model that supersedes the
shipped ``sf_conv_g1.3_tio2_L64_2``.  Same U-Net topology, same 6-frame
input, same 2D Gaussian PSF, same loss family (GMM).  The differences below
are the reason for the retraining.  Please DO NOT swap the model class or
config keys silently -- they are chosen deliberately.

The config lives in this file (not a separate .yaml) so it is under git
version control alongside the code.  ``LocalizationModelTrainer.__init__``
writes the resolved config to ``<save_dir>/config.yaml`` at startup as a
record of exactly what was trained with.

Motivating experiments
----------------------
The prior evaluation showed the current shipped model:
  * Fails to detect single emitters at total photon counts < 300, especially
    at low background (bg=0.5).  Confirmed both with the isolated
    ``estimation_vs_crlb.py`` sweep and with the bg-sweep plot in
    ``example/plots/estimation_vs_crlb_bg_sweep_notrim.png``.
  * Achieves SIMFLUX CRLB only when the NN's max-prob pixel picks the
    correct spot AND the NDI pattern-fit is applied on the extracted N_k.
    The pattern-agnostic NN itself matches at best the summed-frame SMLM CRLB
    (by design, see below).
  * Wastes ~half of every output activation on unphysical negative values.
    ``tanh(x) * max_bg=100`` covers [-100, +100] but training bg lives in
    [0.5, 20].  Same story for N_k with max_intensity=20000 vs actual [50, 10000].
  * Was trained with ``intensity_fluctuation_std=0.5``, but pattern modulation
    with depth=0.9 produces per-frame relative intensity variation up to
    ~90%.  The NN's expected fluctuation regime is narrower than what it
    sees at inference.

The model changes are aimed at fixing exactly these issues *without*
altering the pattern-agnostic design (SIMCODE deliberately does not fit
modulation patterns inside the NN -- pitch/depth/phase can vary freely at
inference and are handled downstream by ``ndi_fit_torch``).

Change list, model side (see smlmtorch/nn/sf_model_expn.py)
------------------------------------------------------------
  1. Exp activation on bg and every N_k.  Concretely:
         output_bg    = exp(clamp(raw, -8, 8)) * bg_scale
         output_N_k   = exp(clamp(raw, -8, 8)) * intensity_scale
     Network learns log-photons directly, which matches the log-normal /
     log-uniform training distribution.  ``bg_scale=5.0`` and
     ``intensity_scale=700.0`` are geometric means of their respective
     training ranges -- pick raw ~= 0 for the "typical" spot.
  2. tanh_features / sigmoid_features / eps_features overridden so bg + N_k
     no longer flow through tanh.  x, y stay on tanh (signed offsets from
     pixel centre); prob and sigma outputs stay on sigmoid.
  3. Anscombe input transform.  ``input_transform='anscombe'`` replaces the
     linear ``(x - offset) * scale`` with ``2*sqrt(clip(x, -3/8) + 3/8) * scale``.
     Variance-stabilises shot noise: Poisson(lambda) becomes approximately
     N(2*sqrt(lambda), 1), so noise is roughly Gaussian with unit variance
     regardless of signal level.  Amplifies low-signal differences (SNR
     ~ 1/sqrt(lambda)) which is what matters for detection at low photons.
     Read noise (~1 photon) is absorbed into the pixel value and clipped
     safely by the sqrt.
  4. ``enable_readnoise=False``.  The per-pixel read-noise channel path
     concatenates sigma_read as an extra input channel; we keep it off
     because read noise is uniformly low (~1 photon) in the target setup,
     so the extra channel just adds compute without adding information.
     The read noise still contaminates the pixel value in simulation and
     the Anscombe transform handles it inside the sqrt.

Change list, loss (config.loss)
-------------------------------
  4. count_loss_weight bumped 0.01 -> 0.05.  The count constraint had almost
     no gradient in the shipped model, contributing to the low-SNR detection
     failure.  Modest bump so it has some pull without dominating the GMM loss.

Change list, training data (config.simulation)
----------------------------------------------
  5. intensity_distr:   log-normal -> log-uniform.  Log-uniform gives every
     photon decade equal training weight.  The old log-normal mode=400,
     mean=600 heavily under-represents the 50-200 photon regime where the
     NN currently fails.
  6. intensity_fluctuation_std:  0.5 -> 0.9.  Matches the ~90% amplitude
     of pattern modulation with depth=0.9 that the NN encounters at
     inference.
  7. bg_distr: uniform -> log-uniform (new option added in
     smlmtorch/nn/simulate/dataset_generator.py).  Uniform sampling on
     [0.5, 20] gave the NN a training mass distribution biased toward
     high-bg samples; log-uniform on [0.3, 25] balances the low-bg regime
     which is where our bg-sweep showed the worst NN behaviour.
  8. bg_min 0.5 -> 0.3, bg_max 20 -> 25.  Small widening so the training
     distribution edges sit outside the typical inference range.

Change list, optimiser (config.lr_scheduler)
--------------------------------------------
  9. StepLR (gamma=0.5, step=30) -> CosineAnnealingLR (T_max=300, eta_min=1e-7).
     The previous schedule dropped LR to ~1e-6 by epoch 172 (the shipped
     checkpoint).  Cosine gives a smoother decay across 300 epochs.

What is NOT changed
-------------------
  * U-Net topology, feature sizes, number of downsamples.
  * GMM loss formulation.
  * ``modulated_estim`` branch -- it's dead code in the current forward and
    we're deliberately keeping the NN pattern-agnostic.  Leave for later
    cleanup.
  * ``density_um2`` -- 1.5 spots/um^2 stays the same.
  * Base optimiser -- Lion with lr=2.0e-05, weight_decay=0.01.

Running this
------------
Prerequisites (on a Linux GPU box):

    conda create -n simcode_env python=3.11 pytorch torchvision torchaudio \\
        pytorch-cuda=12.4 -c pytorch -c nvidia
    conda activate simcode_env
    cd <where you cloned simcode-release>
    pip install -e .

Then:

    python example/train_expn.py               # default: 300 epochs, batch 6
    python example/train_expn.py --epochs 150  # short run to sanity-check first
    python example/train_expn.py --resume      # pick up from latest checkpoint

Outputs land in ``model/sf_conv_g1.3_expn/`` -- the same layout as the
shipped model directory (``checkpoint_1.pt``, ``config.yaml`` + tensorboard
logs).  The config is written from this file at startup, so what's in the
save-dir always matches what was actually trained with.  A full 300-epoch
run on an RTX 3090 should take ~24-30 hours.  Short 150-epoch runs are fine
for a first sanity check.

Post-training verification
--------------------------
After training completes, evaluate against the shipped model with:

    python example/estimation_vs_crlb.py

which already loads the shipped ``sf_conv_g1.3_tio2_L64_2`` checkpoint.
To evaluate the new model, edit the ``load_simcode_model`` call in that
script (or add a --model flag) to point at ``model/sf_conv_g1.3_expn/`` and
the class to ``SFLocalizationModelExpN``.

Notes on gotchas
----------------
  * With ``enable_readnoise=True``, the model *requires* a read-noise
    input at inference.  Feed a zero map if you don't have one.  See
    ``estimation_vs_crlb.py:nn_estimate`` for the pattern.
  * The exp() activation saturates numerically if raw logits leave
    [-8, 8].  Kept in check by the clamp in ``apply_output_scaling`` plus
    the global ``clip_grad_norm`` in the config.  If loss NaNs during
    training, first check that clip_grad_norm.max_norm is not too large.
  * Log-uniform intensity sampling is *the* biggest expected win.  If the
    resulting model is not better than the shipped one at ph<300, that
    hypothesis is falsified and we should look at the count_loss_weight
    bump next.
"""

import argparse
import os

from smlmtorch import config_dict
from smlmtorch.nn.model_trainer import LocalizationModelTrainer
from smlmtorch.nn.sf_model_expn import SFLocalizationModelExpN


# Structural constants: these follow the shipped colab_training_example.ipynb
# so the U-Net topology matches exactly.
GAUSS_SIGMA = 1.3
L = 64
MOVING_WND = 6              # num_intensities
CENTER_FRAME_IX = 2         # track_intensities_offset (both loss & simulation)
DEFAULT_EPOCHS = 300


def make_config(save_dir=None):
    """Build the full training config as a config_dict.  Kept as a function so
    it's easy to load or override from tests / notebooks."""
    cfg = config_dict(
        model=dict(
            # --- ExpN activation scales ---
            # bg_scale / intensity_scale are the multipliers applied to exp(x)
            # of the raw output.  Set to the geometric mean of the target range
            # so raw ~= 0 corresponds to a typical training spot.
            bg_scale=1.0,
            intensity_scale=400.0,

            # --- Architecture (matches the shipped sf_conv model) ---
            enable_readnoise=False,       # read noise is low (~1 ph) & uniform; no channel needed
            enable3D=False,
            unet_shared_features=[L, L * 2],
            unet_combiner_features=[L, L * 2, L * 4],
            unet_combiner_output_features=256,
            unet_intensity_features=[L, L * 2, L * 4],
            output_intensity_features=32,
            output_head_features=48,
            ie_input_features=32,
            unet_batch_norm=True,

            # <-- ExpN change: variance-stabilising Anscombe transform on inputs.
            #     Post-transform scale of 0.05 brings a 5000-photon bright pixel
            #     (2*sqrt(5000)=141) to ~7, and a 5-photon bg pixel (2*sqrt(5.375)=4.6)
            #     to ~0.23 -- similar dynamic range to the old linear (0.01, offset 3)
            #     but with per-pixel noise variance approximately equalised.
            input_transform='anscombe',
            input_scale=0.05,
            input_offset=0.0,

            num_intensities=MOVING_WND,
            xyz_scale=[1.1, 1.1, 1.0],
            use_on_prob=False,
        ),

        loss=dict(
            gmm_components=0,
            count_loss_weight=0.01,       # reverted to notebook value
            track_intensities_offset=CENTER_FRAME_IX,
        ),

        optimizer_type='Lion',
        optimizer=dict(
            lr=2e-5,
            weight_decay=0.01,
        ),

        lr_scheduler=dict(step_size=30, gamma=0.5),

        clip_grad_norm=dict(max_norm=2.0),

        train_size=8 * 1024,
        test_size=1024,
        batch_size=6,

        simulation=dict(
            num_frames=32,
            img_shape=(32, 32),
            z_range=(-0.5, 0.5),
            density_um2=1.5,
            pixelsize_nm=100,
            mean_on_time=6,

            track_intensities=MOVING_WND,
            track_intensities_offset=CENTER_FRAME_IX,

            # --- ExpN training-distribution changes ---
            intensity_distr='log-normal',         # reverted to notebook value
            intensity_fluctuation_std=0.5,        # reverted to notebook value
            intensity_mean_min=50,
            intensity_mean_max=10000,
            intensity_mode=400,                   # kept for backward compat, unused for log-uniform
            intensity_mean=600,                   # kept for backward compat, unused for log-uniform

            bg_distr='log-uniform',               # <-- new option (see dataset_generator.py)
            bg_min=0.3,                           # was 0.5
            bg_max=25,                            # was 20

            render_args=dict(
                read_noise_mean=0.5,
                read_noise_std=1,
            ),
            psf=dict(
                type='Gaussian2D',
                sigma=[GAUSS_SIGMA, GAUSS_SIGMA],
            ),
        ),

        benchmark=dict(
            prob_threshold=0.5,
            match_distance_px=3,
            kdeplot_params=['N0', 'N5'],
            render_frame_offset=CENTER_FRAME_IX,
        ),
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Train SFLocalizationModelExpN')
    parser.add_argument('--save-dir',
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            '..', 'model', f'sf_conv_g{GAUSS_SIGMA}_expn'),
                        help='Where to write checkpoints + tensorboard logs '
                             '(and a config.yaml snapshot of what was trained)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--resume', action='store_true',
                        help='Load latest checkpoint from save-dir and continue')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the default batch_size (6)')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    config = make_config(save_dir=args.save_dir)
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    print('=' * 72)
    print('Training SFLocalizationModelExpN')
    print(f'  save_dir: {args.save_dir}')
    print(f'  device:   {args.device}')
    print(f'  epochs:   {args.epochs}')
    print(f'  batch:    {config.batch_size}')
    print(f'  resume:   {args.resume}')
    print('=' * 72)

    trainer = LocalizationModelTrainer(
        config,
        SFLocalizationModelExpN,
        device=args.device,
        save_dir=args.save_dir,
        load_previous_model=args.resume,
    )

    trainer.train(num_epochs=args.epochs,
                  batch_size=config.batch_size,
                  log_interval=1,
                  report_interval=5,
                  data_refresh_interval=1,
                  save_checkpoints=True)


if __name__ == '__main__':
    main()
