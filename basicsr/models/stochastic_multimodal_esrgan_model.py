import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

from utils.registry import MODEL_REGISTRY
from .stochastic_srgan_model import StoSRGANModel


@MODEL_REGISTRY.register()
class StoMMESRGANModel(StoSRGANModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output, std_weights, dist_weights = self.net_g(self.lq, get_stat=True)

        use_pos = self.opt['train']['weightgan']['use_pos']

        if use_pos == 'std':
            self.pos_weight = std_weights
        elif use_pos == 'dist':
            self.pos_weight = dist_weights
        elif use_pos == 'mindist':
            self.pos_weight = 1 - dist_weights
        self.pos_weight = self.pos_weight.mean(dim=1, keepdims=True)

        l_g_total = 0
        loss_dict = OrderedDict()
        weight_vgg = self.opt['train']['weightgan']['apply_vgg'] if 'apply_vgg' in self.opt['train']['weightgan'] else 0
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                if self.pos_weight is None:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                else:
                    l_g_pix = self.cri_pix(self.output, self.gt, pos_weight=1-weight_vgg*self.pos_weight)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt, pos_weight=1-weight_vgg*self.pos_weight)
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
        apply_weight_d = self.opt['train']['weightgan']['apply_d'] if 'apply_d' in self.opt['train']['weightgan'] else True
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
        if apply_weight_d:
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True, pos_weight=self.pos_weight) * 0.5
        else:
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        if apply_weight_d:
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True, pos_weight=self.pos_weight) * 0.5
        else:
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        loss_dict['weight_average'] = torch.mean(self.pos_weight)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
