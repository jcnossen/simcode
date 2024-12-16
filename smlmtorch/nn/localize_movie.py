
import torch
import yaml
from smlmtorch.util.config_dict import config_dict
from smlmtorch.util.multipart_tiff import tiff_read_file, tiff_get_movie_size
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
from smlmtorch.nn.localization_detector import LocalizationDetector
from smlmtorch.nn.checkpoint import CheckpointManager

from smlmtorch.simflux.dataset import SFDataset
from smlmtorch import Dataset

def iterate_moving_window(src, wndsize):
    buf = []
    for img in src:
        buf.append(img)
        if len(buf) < wndsize:
            continue
        
        for wndimg in buf:
            yield wndimg
        buf.pop(0)


class MovieIterableDataset(IterableDataset):
    def __init__(self, tif_fn = None, data = None, batch_size=1, batch_overlap=0, maxframes=None,
                  crop_frame_margin=None, tiff_read_args={}):

        if tif_fn is None:
            img_stream = data
            self.shape = data.shape[1:]
            self.numframes = data.shape[0]
        else:
            print('Scanning ', tif_fn)
            self.shape, self.numframes = tiff_get_movie_size(tif_fn)
            print(f'Processing {self.numframes} frames from {tif_fn}. Shape: {self.shape}')
            img_stream = self._stream(tif_fn, maxframes)

        self.crop_frame_margin = crop_frame_margin
        self.tiff_read_args = tiff_read_args

        # apply frame margins
        def make_batches(img_stream):
            batch = []
            for img in img_stream:
                batch.append(img)
                if len(batch) == batch_size:
                    yield torch.stack(batch)
                    if batch_overlap > 0 and len(batch) > batch_overlap:
                        batch = batch[-batch_overlap:]
                    else:
                        return

            if len(batch) > 0:
                yield torch.stack(batch)

        self.iterator = make_batches(img_stream)
        self.length = min(maxframes, self.numframes) if maxframes is not None else self.numframes

    def _stream(self, tif_fn, max_frames):
        for frame in tiff_read_file(tif_fn, maxframes=max_frames, **self.tiff_read_args):
            if self.crop_frame_margin is not None:
                frame = frame[self.crop_frame_margin:-self.crop_frame_margin, self.crop_frame_margin:-self.crop_frame_margin]

            # Prevent issues with convolutional network sizes
            h = frame.shape[0] - frame.shape[0] % 4
            w = frame.shape[1] - frame.shape[1] % 4
            frame = frame[:h, :w]

            yield torch.tensor(frame.astype(np.float32))

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.iterator

class MovieProcessor:
    def __init__(self, model_class, config, model_chkpt, device):
        self.config = config
        self.model = model_class(**self.config.model)
        
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.detector = LocalizationDetector(**config.detector)

        checkpoint = torch.load(model_chkpt, map_location=torch.device('cpu'))
        model_state = checkpoint['model_state_dict']
        if 'output_scale' in model_state:
            model_state.pop('output_scale')
        self.model.load_state_dict(model_state)

        #checkpoint_manager = CheckpointManager(None, None, model_dir)
        #chkpt = checkpoint_manager.find_latest_checkpoint()
        #self.load_from_checkpoint(checkpoint_fn=chkpt)

    def process(self, ds, gain=1, offset=0, return_outputs=False, state_reset_interval=None, **kwargs):
        #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        if type(ds) == str:
            ds = MovieIterableDataset(ds, **kwargs)

        outputs=[]
        states = []
        self.model.eval()
        model_state = None

        locs = []
        frame_ix = []

        cur_frame = 0
        last_state_reset = 0
        states = []
        input_batches = []
        for img_batch in ds:
            with torch.no_grad():
                img_batch = (img_batch.to(self.device) - offset) * gain
                img_batch = torch.clamp(input=img_batch, min=0)

                if img_batch.shape[0] < self.model.min_frames:
                    break
                
                output, new_state = self.model.forward(img_batch[None], camera_calib=None, model_state=model_state)
                #states.append(new_state[0].reshape(100,100,-1).detach().cpu())

                #input_batches.append(img_batch.detach().cpu())

                #print(f"Batch {cur_frame} - {cur_frame+len(img_batch)}. Output shape: {output.shape} ")

                #batchsize, frames, features, h,w = output.shape
                #states.append(model_state.detach().cpu().reshape(batchsize, h, w, -1).permute(0,3,1,2))
                #locs = self.model.to_locs(output, moving_window//2)
                if return_outputs:
                    outputs.append(output[0].cpu())
                src_info, locs_ = self.detector.detect(output)
                if locs_.shape[0]>0:
                    frame_ix.append(src_info['frame'] + cur_frame)
                    locs.append(locs_)
                cur_frame += output.shape[1]

                # For now make sure to not use RNN in length beyond what it is trained on
                if state_reset_interval is not None and cur_frame > last_state_reset + state_reset_interval:
                    model_state = None
                    last_state_reset = cur_frame
                else:
                    model_state = new_state


            # Process the image batch using your model here

        #from smlmtorch import image_view
        #image_view(torch.stack(states).permute(0,3,1,2))
        #image_view(torch.cat(outputs))

        #image_view(torch.cat(states))
        #image_view(torch.cat(outputs))
        frame_ix = torch.cat(frame_ix)
        dims = 3 if self.model.enable3D else 2
        locs = torch.cat(locs)
        if self.model.num_intensities > 1:
            ds = SFDataset(len(locs), dims, numPatterns=self.model.num_intensities, imgshape=img_batch.shape[-2:])
        else:
            ds = Dataset(len(locs), dims, imgshape=img_batch.shape[-2:])
        nparam = (locs.shape[1]-1)//2
        xy_ix = np.array([self.model.param_names.index(n)+1 for n in ['x', 'y']])
        bg_ix = self.model.param_names.index('bg') + 1

        if ('N0' in self.model.param_names):
            N_ix = self.model.param_names.index('N0') + 1
        else:
            N_ix = self.model.param_names.index('N') + 1
            
        ds.frame = frame_ix.numpy()
        ds.pos[:,:2] = locs[:, xy_ix]
        ds.photons = locs[:, N_ix]
        ds.background = locs[:, bg_ix]
        ds.crlb.pos[:,:2] = locs[:, xy_ix+nparam]
        if dims==3:
            ds.pos[:,2] = locs[:, self.model.param_names.index('z')+1]
            ds.crlb.pos[:,2] = locs[:, self.model.param_names.index('z')+1+nparam]
        ds.crlb.photons = locs[:, N_ix+nparam]
        ds.crlb.background = locs[:, bg_ix+nparam]
        if self.model.num_intensities > 1:
            ds.ibg[:,:,0] = locs[:, N_ix:N_ix+self.model.num_intensities].numpy()
            ds.ibg_crlb[:,:,0] = locs[:, N_ix+nparam:N_ix+nparam+self.model.num_intensities].numpy()
            ds.photons = ds.ibg[:,:,0].sum(-1)
        #ds.frame = 
        if return_outputs:
            return ds, torch.cat(outputs)
        
        return ds

