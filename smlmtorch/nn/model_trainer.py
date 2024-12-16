"""

"""
#%%
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from torch.utils.tensorboard import SummaryWriter

from smlmtorch.nn.gmm_loss import GMMLoss
from smlmtorch.nn.simulate.dataset_generator import RandomSpotDataset, SimulatedSMLMDataset, SMLMDataGenerator
from smlmtorch.nn.localization_model import LocalizationModel
from smlmtorch.nn.benchmark.performance_logger import PerformanceLogger
from smlmtorch.nn.localization_detector import LocalizationDetector
from smlmtorch.nn.checkpoint import CheckpointManager

from lion_pytorch import Lion
from torch.optim import Adam
import os
import pickle
from smlmtorch import config_dict
from smlmtorch.nn.utils.batch_utils import move_to_device, batch_cat
import platform
from smlmtorch import progbar

class LocalizationModelTrainer:
    def __init__(self, config, model_class_or_inst, 
                 device, save_dir, data_generator = None, load_previous_model=False,
                 spot_ds_type = None):

        #self.model = model.to(device)
        #self.loss = loss.to(device)
        self.device = device
        self.save_dir = save_dir
        self.load_previous_model = load_previous_model

        config = config_dict.from_dict(config)

        self.train_size = config['train_size']
        self.test_size = config['test_size']

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        config.save(os.path.join(save_dir, 'config.yaml'))

        self.config = config
        if isinstance(model_class_or_inst, type):
            self.model = model_class_or_inst(**config.model)
        else:
            self.model = model_class_or_inst

        self.loss = self.model.create_loss(**config.loss)

        if data_generator is None:
            if spot_ds_type is None:
                spot_ds_type = RandomSpotDataset
            data_generator = SMLMDataGenerator(device=device, spot_ds_type = spot_ds_type, **config.simulation)
        self.data_generator = data_generator
        
        self.model.to(self.device)
        self.loss.to(self.device)

        if config.optimizer_type == 'Lion':
            self.optimizer = Lion(self.model.parameters(), **config.optimizer)
        else:
            self.optimizer = torch.optim.__dict__[config.optimizer_type](self.model.parameters(), **config.optimizer)

        self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, save_dir)

        if not ('type' in config.lr_scheduler):
            lr_scheduler_type = 'StepLR'
        else:
            lr_scheduler_type = config.lr_scheduler.type
            config.lr_scheduler.pop('type')
        
        self.lr_scheduler = torch.optim.lr_scheduler.__dict__[lr_scheduler_type](self.optimizer, **config.lr_scheduler)

        if load_previous_model:
            self.epoch = self.checkpoint_manager.load_latest()
            if self.epoch is None:
                self.epoch = 0
        else:
            self.epoch = 0

        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard_logs'))
        self.performance_logger = PerformanceLogger(self.writer, 
                    param_idx_map = config_dict(self.loss.param_idx_map),
                    **self.config.benchmark)

        self.crlb_plot = None

    @property
    def psf(self):
        return self.data_generator.psf

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def load_checkpoint(self, checkpoint_path):
        self.epoch = self.checkpoint_manager.load(checkpoint_path)

    def test(self, dataset, epoch=0, test_callback=None, close_figs=True, batch_size=1, log=True):
        # Log the test loss and accuracy
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        loss = 0
        loss_info = []
        data = []
        camera_calib = []
        with torch.no_grad():
            for data_, camera_calib_, labels_ in dataloader:
                (loss_, loss_info_), output_ = self.eval_batch(data_, camera_calib_, labels_, return_loss_dict=True)
                loss += loss_.item()
                loss_info.append(loss_info_)
                data.append(data_)
                camera_calib.append(camera_calib_)

        loss_info = batch_cat(loss_info)
        data = batch_cat(data)
        sel_data = data#data[:, :data.shape[1]] - self.model.lookahead_frames] # ugly
        camera_calib = batch_cat(camera_calib)

        if test_callback is not None:
            test_callback(epoch, (sel_data, camera_calib, loss_info['targets']), loss_info['outputs'])

        if log and self.performance_logger is not None:
            print('Writing benchmark info to tensorboard')
            self.performance_logger.log(epoch, sel_data, 
                loss_info['outputs'], self.model, loss_info['targets'], close_figs=close_figs)

        loss /= len(dataloader)
        self.writer.add_scalar("Test Loss", loss, epoch)
        return loss

    def train(self, num_epochs, log_interval=10, batch_size=16,
               data_refresh_interval=20, report_interval=20, test_callback=None, save_checkpoints=True, use_compile=False):

        train_dataloader = None
        test_dataset = None

        if use_compile and platform.system() == 'Linux':
            model = torch.compile(self.model)
            calc_loss = torch.compile(self.loss)
        else:
            model = self.model
            calc_loss = self.loss

        for i in range(num_epochs):
            epoch = self.epoch + i
            if train_dataloader is None or epoch % data_refresh_interval == 0:
                dataset = self.data_generator.forward(self.train_size + self.test_size)
                if self.test_size>0:
                    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])                   
                else:
                    train_dataset = dataset
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

                #if test_callback is not None:
                #    data, labels = train_dataset[0]
                #    output = self.model(data.to(self.device)[None]).detach().cpu()
                #    test_callback(data, labels, output)

            train_loss = 0
            with progbar(total = len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
                for batch_idx, (data, camera_calib, labels) in enumerate(train_dataloader):
                    camera_calib = move_to_device(camera_calib, self.device)
                    labels = move_to_device(labels, self.device)
                    data = move_to_device(data, self.device)

                    #torch.autograd.set_detect_anomaly(True)
                    model.train() # set model to training mode
                    output, _ = model(data, camera_calib)
                    loss = calc_loss(output, labels)
                    if torch.isnan(loss):
                        print("NAN loss")
                        break
                    self.optimizer.zero_grad()
                    loss.backward()

                    train_loss += loss.detach().cpu().item()
                    
                    if self.config['clip_grad_norm']:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), **self.config.clip_grad_norm)

                    self.optimizer.step()

                    pbar.update()
                    pbar.set_description(f"Epoch {epoch} [{batch_idx}/{len(train_dataloader)}], Train Loss: {loss.item():.4e}")
                self.lr_scheduler.step()

            train_loss /= len(train_dataloader)

            if epoch % log_interval == 0 and test_dataset is not None:
                print('Evaluating on test dataset')

                # Log performance every log_interval epochs
                if report_interval>0 and epoch % report_interval == 0:
                    test_loss = self.test(test_dataset, epoch, test_callback, batch_size=batch_size)

                    self.writer.add_scalar("Train Loss", train_loss, epoch)
                    print(f"Epoch {epoch} [{batch_idx}/{len(train_dataloader)}], Train Loss: {train_loss:.4e}. Test Loss: {test_loss:.4e}")

            if loss.isnan().sum()>0:
                print("Loss is NaN. Stopping training")
                break

            #h = hash(pickle.dumps(list(self.model.parameters())))
            #print('model hash: ', h)

            self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], epoch)

            if save_checkpoints:
                print('saving checkpoint')
                self.checkpoint_manager.save(epoch)

        self.writer.close()


    def eval_batch(self, data, camera_calib = None, labels = None, return_loss_dict=False):
        self.model.eval()
        if camera_calib is None:
            camera_calib = torch.zeros_like(data).to(self.device)
        with torch.no_grad():
            data, camera_calib = data.to(self.device), camera_calib.to(self.device)
            output, hidden_state = self.model(data, camera_calib)
            if labels is not None:
                labels = move_to_device(labels, self.device)
                loss_and_optional_target = self.loss(output, labels, return_loss_dict=return_loss_dict)
                return loss_and_optional_target, output
            else:
                return output

