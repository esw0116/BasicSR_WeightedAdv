import nsml
from nsml import DATASET_PATH
import os
import torch
from collections import OrderedDict

from archs import build_network
from losses import build_loss
from utils import get_root_logger
from utils.registry import MODEL_REGISTRY
from .stochastic_sep_sr_model import StoSepSRModel


@MODEL_REGISTRY.register()
class StoSepSRGANModel(StoSepSRModel):
    """SRGAN model for single image super-resolution."""

    def init_training_settings(self):
        self.net_g.train()
        self.net_w.train()
        train_opt = self.opt['train']

        def nsml_load_ema(filename):
            save_filename = 'G.pth'
            filename_g = os.path.join(filename, save_filename)
            print('g_ema loaded!!')
            param_g = torch.load(filename_g)
            self.net_g_ema.load_state_dict(param_g, strict=self.opt['path'].get('strict_load_g', True))

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                if load_path.startswith('NSML'):
                    print(load_path.split('_')[-1], 'KR80934/CVLAB_SR6/{}'.format(load_path.split('_')[-2]))
                    nsml.load(checkpoint=load_path.split('_')[-1], load_fn=nsml_load_ema, session='KR80934/CVLAB_SR6/{}'.format(load_path.split('_')[-2]))
                else:
                    load_path = os.path.join(DATASET_PATH, load_path)
                    self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('pixel_opt_w'):
            self.cri_pix_w = build_loss(train_opt['pixel_opt_w']).to(self.device)
        else:
            self.cri_pix_w = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer w
        optim_type = train_opt['optim_w'].pop('type')
        self.optimizer_w = self.get_optimizer(optim_type, self.net_w.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_w)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

            self.optimizer_w.zero_grad()
            self.outputw = self.net_w(self.output.detach(), self.gt, exploit=self.exploit)

            l_w_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix_w:
                l_pix_w = self.cri_pix_w(self.outputw, self.gt)
                l_w_total += l_pix_w
                loss_dict['l_pix_w'] = l_pix_w

            l_w_total.backward()
            self.optimizer_w.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        with torch.no_grad():
            self.pos_weight = self.net_w(self.ouput.detach(), self.gt, get_std=True)
            self.pos_weight = self.pos_weight.mean(dim=1, keepdim=True)
        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
