import torch
from torch import autograd
from collections import OrderedDict

from losses.loss_util import get_refined_artifact_map
from utils.registry import MODEL_REGISTRY
from .srgan_iccv_model import SRGANICCVModel


@MODEL_REGISTRY.register()
class ESRGANICCVModel(SRGANICCVModel):
    """ESRGAN model for single image super-resolution."""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)
        
        l_g_total = 0
        loss_dict = OrderedDict()
        if hasattr(self, 'coeff'):
            no_dc = self.opt['train']['no_dc'] if 'no_dc' in self.opt['train'] else False
            if self.coeff.shape[1] == 3:
                convert_y = torch.Tensor([65.481/255, 128.553/255, 24.966/255]).reshape(3,1,1).to(self.device)
                if no_dc:
                    self.pos_weight = torch.sum(self.coeff * convert_y, dim=1, keepdims=True)
                else:
                    self.pos_weight = torch.sum(self.coeff * convert_y, dim=1, keepdims=True) + 16/255
            elif self.coeff.shape[1] == 1:
                if no_dc:
                    self.pos_weight = self.coeff - 16/255
                else:
                    self.pos_weight = self.coeff

            weight_policy = self.opt['train']['weightpolicy'] if 'weightpolicy' in self.opt['train'] else None
            if weight_policy == 'clamp':
                self.pos_weight = self.pos_weight.clamp(0, 1)
            loss_dict['weight_average'] = self.pos_weight.mean()
        else:
            self.pos_weight = None

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.cri_ldl:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # out_recon_grad = autograd.grad(outputs=l_g_total, inputs=self.output, retain_graph=True)[0]
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
        attenuate_d = self.opt['train']['attn_d'] if 'attn_d' in self.opt['train'] else False
        if attenuate_d:
            attn_d = self.pos_weight.mean()
        else:
            attn_d = 1

        # real
        fake_d_pred = self.net_d(self.output).detach()
        real_d_pred = self.net_d(self.gt)
        l_d_real = attn_d * self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = attn_d * self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
