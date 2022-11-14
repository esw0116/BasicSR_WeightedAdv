from nsml import DATASET_PATH
import os
import torch
from collections import OrderedDict

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class WeightedESRGANModel(SRModel):
    """ESRGAN model for single image super-resolution."""
    def init_training_settings(self):
        train_opt = self.opt['train']

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
            print(load_path)
            if load_path is not None:
                load_path = os.path.join(DATASET_PATH, load_path)
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_w
        self.net_w = build_network(self.opt['network_w'])
        self.net_w = self.model_to_device(self.net_w)
        self.print_network(self.net_w)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_w', None)
        if load_path is not None:
            load_path = os.path.join(DATASET_PATH, load_path)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_w', 'params')
            self.load_network(self.net_w, load_path, self.opt['path'].get('strict_load_w', True), param_key)

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            load_path = os.path.join(DATASET_PATH, load_path)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()
        self.net_w.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

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
        self.optimizer_g = self.get_optimizer(optim_type, list(self.net_g.parameters())+list(self.net_w.parameters()), **train_opt['optim_g'])
        # self.optimizer_g = self.get_optimizer(optim_type, self.net_w.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
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
        pos_weight = self.net_w(self.lq)
        gray_coeffs = torch.Tensor([65.738 / 256, 129.057 / 256, 25.064 / 256]).reshape((1,3,1,1)).to(self.lq.device)
        self.pos_weight = torch.sum(pos_weight * gray_coeffs, dim=1, keepdim=True)

        norm_quantile = self.opt['train']['weightgan']['quantile']
        gamma = self.opt['train']['weightgan']['gamma']

        # Normalize (low 10% -> 0, high 10% -> 1)
        b, c, h, w = self.pos_weight.shape
        self.pos_weight = self.pos_weight.reshape(b, c, h*w)
        low10 = torch.quantile(self.pos_weight, norm_quantile, dim=2, keepdim=True)
        hi10 =  torch.quantile(self.pos_weight, 1-norm_quantile, dim=2, keepdim=True)
        self.pos_weight = (self.pos_weight -low10) / (hi10 - low10)
        self.pos_weight = self.pos_weight.clamp(0, 1)
        self.pos_weight = self.pos_weight.reshape(b,c,h,w)

        # Add non-linearity
        self.pos_weight = torch.pow(self.pos_weight, gamma)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt, pos_weight=1-0.01*self.pos_weight)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt, pos_weight=self.pos_weight)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt).detach()
            fake_g_pred = self.net_d(self.output)

            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False, pos_weight=self.pos_weight)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False, pos_weight=self.pos_weight)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_pred = self.net_d(self.output).detach()
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_w, 'net_w', current_iter)
        self.save_training_state(epoch, current_iter)
