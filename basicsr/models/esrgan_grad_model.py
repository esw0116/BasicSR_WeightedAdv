import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

from utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class ESRGANGradModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        if hasattr(self, 'coeff'):
            self.pos_weight = self.coeff
            norm_quantile = self.opt['train']['weightgan']['quantile']
            gamma = self.opt['train']['weightgan']['gamma']

            map_process = self.opt['train']['weightgan']['blur'] if 'blur' in self.opt['train']['weightgan'] else None
            if map_process == 'Average' or map_process is True:
                blur_k = torch.ones((1,1,7,7)).type_as(self.pos_weight) / 49
                self.pos_weight = F.conv2d(self.pos_weight, blur_k, padding=3)
            elif map_process == 'Median':
                m = nn.ReplicationPad2d(3)
                h, w = self.pos_weight.shape[-2:]
                self.pos_weight = m(self.pos_weight)
                u = nn.Unfold(kernel_size=(7,7))
                self.pos_weight = u(self.pos_weight)
                self.pos_weight = torch.median(self.pos_weight, dim=1, keepdim=True)[0]
                self.pos_weight = self.pos_weight.reshape((-1, 1, h, w))

            # Normalize (low 10% -> 0, high 10% -> 1)
            # b, c, h, w = self.pos_weight.shape
            # self.pos_weight = self.pos_weight.reshape(b, c, h*w)
            # low10 = torch.quantile(self.pos_weight, norm_quantile, dim=2, keepdim=True)
            # hi10 =  torch.quantile(self.pos_weight, 1-norm_quantile, dim=2, keepdim=True)
            # self.pos_weight = (self.pos_weight -low10) / (hi10 - low10)
            self.pos_weight = self.pos_weight.clamp(0, 1)
            # self.pos_weight = self.pos_weight.reshape(b,c,h,w)

            # Add non-linearity
            self.pos_weight = torch.pow(self.pos_weight, gamma)
        else:
            self.pos_weight = None

        l_g_total = 0
        loss_dict = OrderedDict()
        weight_vgg = self.opt['train']['weightgan']['apply_vgg'] if 'apply_vgg' in self.opt['train']['weightgan'] else 0
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                if self.pos_weight is None:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                else:
                    l_g_pix = self.cri_pix(self.output, self.gt, pos_weight=None) # 1-0.01*self.pos_weight)
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
            l_g_total.backward(retain_graph=True)
            
            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt).detach()
            fake_g_pred = self.net_d(self.output)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False, pos_weight=None)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False, pos_weight= None)
            l_g_gan = (l_g_real + l_g_fake) / 2
            # l_g_gan.backward(retain_graph=True, inputs=self.output)
            out_g_grad = autograd.grad(outputs=l_g_gan, inputs=self.output, retain_graph=True)[0]
            self.output.backward(gradient=out_g_grad * self.pos_weight)

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        apply_weight_d = self.opt['train']['weightgan']['apply_d'] if 'apply_d' in self.opt['train']['weightgan'] else False
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
        # out_d1_grad = autograd.grad(outputs=l_g_gan, inputs=self.output, retain_graph=True)[0]
        # self.output.backward(gradient=out_d1_grad * self.pos_weight)

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

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
