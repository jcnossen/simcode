"""
========================================================================
SIMCODE comparison run: old SFLocalizationModel (tanh activations)
========================================================================

Companion to ``train_expn.py``.  Trains the *original* shipped model
architecture (``SFLocalizationModel`` -- tanh(x)*scale on bg and N_k, linear
input scaling) instead of the exp-activation ``SFLocalizationModelExpN``.

Purpose: train_expn.py's ExpN model appeared to converge very slowly under
the tuned training-loop settings below.  This script isolates the model
architecture as the variable: everything that is not intrinsically tied to
the choice of output activation (optimizer, lr, scheduler, clip_grad_norm,
batch size, data-refresh/log/report intervals, loss weights, simulation
distribution) is kept identical to the current ``train_expn.py`` config, so
any convergence-speed difference between the two runs can be attributed to
the model activation change rather than a training-loop difference.

Model-side settings that only make sense for the tanh-based model are
reverted to the original (pre-ExpN) values:
  * max_bg=100, max_intensity=20000 instead of bg_scale/intensity_scale.
  * input_transform='linear', input_scale=0.01, input_offset=3 instead of
    the Anscombe transform.

Running this
------------
    python example/train.py               # default: 300 epochs, batch 6
    python example/train.py --epochs 150  # short run to sanity-check first
    python example/train.py --resume      # pick up from latest checkpoint

Outputs land in ``model/sf_conv_g1.3_oldmodel/`` -- kept separate from both
the ExpN run's ``model/sf_conv_g1.3_expn/`` and the originally shipped
``sf_conv_g1.3_tio2_L64_2`` checkpoint.
"""

import argparse
import os

from smlmtorch import config_dict
from smlmtorch.nn.model_trainer import LocalizationModelTrainer
from smlmtorch.nn.sf_model import SFLocalizationModel


GAUSS_SIGMA = 1.3
L = 64
MOVING_WND = 6              # num_intensities
CENTER_FRAME_IX = 2         # track_intensities_offset (both loss & simulation)
DEFAULT_EPOCHS = 300


def make_config(save_dir=None):
    """Build the full training config as a config_dict.  Mirrors
    train_expn.py's make_config() except for the model activation block --
    see module docstring for exactly what's reverted vs kept identical."""
    cfg = config_dict(
        model=dict(
            # --- Original tanh(x) * scale activation (pre-ExpN) ---
            max_bg=100,
            max_intensity=20000,

            enable_readnoise=False,
            enable3D=False,
            unet_shared_features=[L, L * 2],
            unet_combiner_features=[L, L * 2, L * 4],
            unet_combiner_output_features=256,
            unet_intensity_features=[L, L * 2, L * 4],
            output_intensity_features=32,
            output_head_features=48,
            ie_input_features=32,
            unet_batch_norm=True,

            # --- Original linear input scaling (pre-ExpN) ---
            input_transform='linear',
            input_scale=0.01,
            input_offset=3,

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
            lr=2e-5,                      # kept same as train_expn.py
            weight_decay=0.01,
        ),

        lr_scheduler=dict(step_size=30, gamma=0.5),

        clip_grad_norm=dict(max_norm=2.0),  # kept same as train_expn.py

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

            intensity_distr='log-normal',         # reverted to notebook value
            intensity_fluctuation_std=0.5,        # reverted to notebook value
            intensity_mean_min=50,
            intensity_mean_max=10000,
            intensity_mode=400,
            intensity_mean=600,

            bg_distr='log-uniform',
            bg_min=0.3,
            bg_max=25,

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
    parser = argparse.ArgumentParser(description='Train SFLocalizationModel (old, tanh activations)')
    parser.add_argument('--save-dir',
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            '..', 'model', f'sf_conv_g{GAUSS_SIGMA}_oldmodel'),
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
    print('Training SFLocalizationModel (old, tanh activations)')
    print(f'  save_dir: {args.save_dir}')
    print(f'  device:   {args.device}')
    print(f'  epochs:   {args.epochs}')
    print(f'  batch:    {config.batch_size}')
    print(f'  resume:   {args.resume}')
    print('=' * 72)

    trainer = LocalizationModelTrainer(
        config,
        SFLocalizationModel,
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
