import torch
from collections import OrderedDict

from utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class ResidualESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        res_gt = torch.abs(self.output - self.gt)
        gray_coeffs = torch.Tensor([65.738 / 256, 129.057 / 256, 25.064 / 256]).reshape((1,3,1,1)).to(self.lq.device)
        self.pos_weight = torch.sum(res_gt * gray_coeffs, dim=1, keepdim=True)

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
