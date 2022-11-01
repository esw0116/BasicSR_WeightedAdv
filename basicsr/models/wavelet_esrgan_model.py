import torch
from torch.nn import functional as F
from collections import OrderedDict
from pytorch_wavelet import Haar2

from utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class WaveletESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        gray_coeffs = torch.Tensor([65.738 / 256, 129.057 / 256, 25.064 / 256]).reshape((1,3,1,1)).to(self.lq.device)
        # gray_gt = torch.sum(self.gt * gray_coeffs, dim=1, keepdim=True)
        gray_out = torch.sum(self.output * gray_coeffs, dim=1, keepdim=True)
        h, w = gray_out.shape[-2:]
        xfm = Haar2(device=self.lq.device)
        gray_out = gray_out.permute(2,3,0,1)
        wavelet_out = xfm.forward(gray_out, dtype=self.output.dtype)
        wavelet_out = torch.stack(wavelet_out[1:], dim=0)
        wavelet_out = torch.max(wavelet_out, dim=0)[0]
        wavelet_out = wavelet_out.permute(2,3,0,1)
        self.pos_weight = F.interpolate(wavelet_out, size=(h,w), mode='bilinear', align_corners=True)
        # xfm = DWTForward(J=1, mode='zero', wave='haar')
        # xfm = xfm.to(self.lq.device)
        # gol, goh = xfm(gray_out)

        # Normalize (low 10% -> 0, high 10% -> 1)
        b, c, h, w = self.pos_weight.shape
        self.pos_weight = self.pos_weight.reshape(b, c, h*w)
        low10 = torch.quantile(self.pos_weight, 0.1, dim=2, keepdim=True)
        hi10 =  torch.quantile(self.pos_weight, 0.9, dim=2, keepdim=True)
        self.pos_weight = (self.pos_weight -low10) / (hi10 - low10)
        self.pos_weight = self.pos_weight.clamp(0,1)
        self.pos_weight = self.pos_weight.reshape(b,c,h,w)

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
