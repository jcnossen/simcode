import numpy as np
import torch
from smlmtorch.nn.simulate.dataset_generator import SimulatedSMLMDataset, RandomSpotDataset, SpotDataset, SMLMDataGenerator
from smlmtorch.simflux.pattern_estimator import ModulationPattern
from smlmtorch import config_dict


class ModulatedSpotDataset(RandomSpotDataset):
    def __init__(self,num_movies, num_frames, min_depth=0.1, max_depth=0.9, min_relint=0.5, max_relint=1,
                 min_pitch_px=2, max_pitch_px=2, phasesteps=3, mod_axes=2,
                  **kwargs):
        
        super().__init__(num_movies=num_movies, 
                            num_frames=num_frames,
                          **kwargs)
        
        # generate modulation patterns
        angles = torch.rand(num_movies) * 2*torch.pi
        mod_angles = angles[:,None] + np.linspace(0,2*np.pi, mod_axes, endpoint=False)[None]
        relint = min_relint + torch.rand(num_movies, mod_axes) * (max_relint-min_relint)

        phase_offsets = torch.rand(num_movies, mod_axes) * 2*torch.pi
        phases = phase_offsets[:,:,None] + np.linspace(0,2*np.pi, phasesteps, endpoint=False)[None,None]

        angle_ix = np.arange(num_frames) % mod_axes
        step_ix = (np.arange(num_frames)//mod_axes)%phasesteps
        frame_phases = phases[:, angle_ix, step_ix]
        frame_angles = mod_angles[:, angle_ix]

        freq = 2*np.pi/(min_pitch_px + torch.rand(num_movies) * (max_pitch_px-min_pitch_px))
        self.k = torch.zeros((num_movies, num_frames, 2))
        self.k[:,:,0] = torch.cos(frame_angles) * freq[:,None]
        self.k[:,:,1] = torch.sin(frame_angles) * freq[:,None]
        depths = min_depth + torch.rand(num_movies,mod_axes) * (max_depth-min_depth)
        self.frame_depths = depths[:, angle_ix]
        self.frame_relint = relint[:, angle_ix]
        self.frame_phases = frame_phases

        # [active, intensity, x, y, z, t_start, t_end, bg]
        # spots fmt: [movies, frames, max_spots, 8]
        xy = self.spots[..., 2:4]
        modulation = (1 + self.frame_depths[:,:,None] * torch.sin(
            (xy*self.k[:,:,None]).sum(-1)-frame_phases[:,:,None])) * self.frame_relint[:,:,None]
        self.spots[..., 1] *= modulation

        self.update_tracked_intensities()
        self.apply_max_spots()
        

    def __getitem__(self, idx): 
        d = super().__getitem__(idx)
        return dict(**d, k=self.k[idx], 
                    depths=self.frame_depths[idx],
                    relint=self.frame_relint[idx],
                    phases=self.frame_phases[idx])
        


if __name__ == '__main__':

    gen_cfg = config_dict(num_frames=32, 
                          num_spots=100, max_spots=50,
                    min_depth=0.1, max_depth=0.9)

    ds_gen = SMLMDataGenerator(psf = config_dict(type='Gaussian2D', sigma=1.8), 
                    device='cuda:0',
                      img_shape = [40,40],
                      spot_ds_type=ModulatedSpotDataset,
                      **gen_cfg
                      )
    
    images = ds_gen.generate(4).data

    from smlmtorch.ui.array_view_pyqt import image_view
    image_view(images)
