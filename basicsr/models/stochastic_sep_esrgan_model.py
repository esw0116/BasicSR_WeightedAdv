import nsml
from nsml import DATASET_PATH
import os
import torch
from collections import OrderedDict

from archs import build_network
from losses import build_loss
from utils import get_root_logger
from utils.registry import MODEL_REGISTRY
from .stochastic_sep_srgan_model import StoSepSRGANModel


@MODEL_REGISTRY.register()
class StoSepESRGANModel(StoSepSRGANModel):
    """SRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        encoder_loss_d = self.net_d.encoder_loss if hasattr(self.net_d, 'encoder_loss') else False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        with torch.no_grad():
            self.pos_weight = self.net_w(self.output.detach(), self.gt, get_std=True)
            self.pos_weight = self.pos_weight.mean(dim=1, keepdim=True)

        pos_coeff = self.opt['train']['weightgan']['pos_coeff'] if 'pos_coeff' in self.opt['train']['weightgan'] else 1
        self.pos_weight = self.pos_weight * pos_coeff

        l_g_total = 0
        loss_dict = OrderedDict()
        weight_vgg = self.opt['train']['weightgan']['apply_vgg'] if 'apply_vgg' in self.opt['train']['weightgan'] else 0

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
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
            
            # gan loss
            if encoder_loss_d:
                real_d_pred, real_d_enc = self.net_d(self.gt)
                real_d_pred, real_d_enc = real_d_pred.detach(), real_d_enc.detach()
                fake_g_pred, fake_g_enc = self.net_d(self.output)
            else:
                real_d_pred = self.net_d(self.gt).detach()
                fake_g_pred = self.net_d(self.output)

            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False, pos_weight=self.pos_weight)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False, pos_weight=self.pos_weight)
            l_g_gan = (l_g_real + l_g_fake) / 2
            if encoder_loss_d:
                l_g_gan += self.cri_gan(real_d_enc - torch.mean(fake_g_enc), False, is_disc=False) * 0.5
                l_g_gan += self.cri_gan(fake_g_enc - torch.mean(real_d_enc), True, is_disc=False) * 0.5
                l_g_gan = l_g_gan / 2
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        apply_weight_d = self.opt['train']['weightgan']['apply_d'] if 'apply_d' in self.opt['train']['weightgan'] else True
        for p in self.net_d.parameters():
            p.requires_grad = True
        
        self.optimizer_d.zero_grad()
        if encoder_loss_d:
            fake_d_pred, fake_d_enc = self.net_d(self.output)
            fake_d_pred, fake_d_enc = fake_d_pred.detach(), fake_d_enc.detach()
            real_d_pred, real_d_enc = self.net_d(self.gt)
        else:
            fake_d_pred = self.net_d(self.output).detach()
            real_d_pred = self.net_d(self.gt)
        
        if apply_weight_d:
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True, pos_weight=self.pos_weight) * 0.5
        else:
            l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        
        if encoder_loss_d:
            l_d_real += self.cri_gan(real_d_enc - torch.mean(fake_d_enc), True, is_disc=True) * 0.5
            l_d_real = l_d_real / 2
        l_d_real.backward()

        # fake
        if encoder_loss_d:
            fake_d_pred, fake_d_enc = self.net_d(self.output.detach())
        else:
            fake_d_pred = self.net_d(self.output.detach())
        
        if apply_weight_d:
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True, pos_weight=self.pos_weight) * 0.5
        else:
            l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        if encoder_loss_d:
            l_d_fake += self.cri_gan(fake_d_enc - torch.mean(real_d_enc.detach()), False, is_disc=True) * 0.5
            l_d_fake = l_d_fake / 2
        
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        loss_dict['trained_weight'] = self.pos_weight.mean()

        # optimize net_w
        self.optimizer_w.zero_grad()
        self.outputw = self.net_w(self.output.detach(), self.gt, exploit=self.exploit)
        l_w_total = 0
        # pixel loss
        if self.cri_pix_w:
            l_pix_w = self.cri_pix_w(self.outputw, self.gt)
            l_w_total += l_pix_w
            loss_dict['l_pix_w'] = l_pix_w

        l_w_total.backward()
        self.optimizer_w.step()

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
