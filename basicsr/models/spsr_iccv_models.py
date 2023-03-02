import nsml
from nsml import DATASET_PATH
import os
import torch
from torch import autograd
from collections import OrderedDict

from archs import build_network
from losses import build_loss
from utils import get_root_logger
from utils.registry import MODEL_REGISTRY
from losses.loss_util import get_refined_artifact_map
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SPSRModel(BaseModel):
    def __init__(self, opt):
        super(SPSRModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netD_grad = networks.define_D_grad(opt).to(self.device) # D_grad
            self.netG.train()
            self.netD.train()
            self.netD_grad.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            # Branch_init_iters
            self.Branch_pretrain = train_opt['Branch_pretrain'] if train_opt['Branch_pretrain'] else 0
            self.Branch_init_iters = train_opt['Branch_init_iters'] if train_opt['Branch_init_iters'] else 1

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # gradient_pixel_loss
            if train_opt['gradient_pixel_weight'] > 0:
                self.cri_pix_grad = nn.MSELoss().to(self.device)
                self.l_pix_grad_w = train_opt['gradient_pixel_weight']
            else:
                self.cri_pix_grad = None

            # gradient_gan_loss
            if train_opt['gradient_gan_weight'] > 0:
                self.cri_grad_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_grad_w = train_opt['gradient_gan_weight']
            else:
                self.cri_grad_gan = None

            # G_grad pixel loss
            if train_opt['pixel_branch_weight'] > 0:
                l_pix_type = train_opt['pixel_branch_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix_branch = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix_branch = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_branch_w = train_opt['pixel_branch_weight']
            else:
                logger.info('Remove G_grad pixel loss.')
                self.cri_pix_branch = None

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            optim_params = []
            for k, v in self.netG.named_parameters():  # optimize part of the model

                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))

            self.optimizers.append(self.optimizer_D)

            # D_grad
            wd_D_grad = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D_grad = torch.optim.Adam(self.netD_grad.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))

            self.optimizers.append(self.optimizer_D_grad)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            self.get_grad = Get_gradient()
            self.get_grad_nopadding = Get_gradient_nopadding()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)



    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        for p in self.netD_grad.parameters():
            p.requires_grad = False


        if(self.Branch_pretrain): 
            if(step < self.Branch_init_iters):
                for k,v in self.netG.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=False
            else:
                for k,v in self.netG.named_parameters():
                    if 'f_' not in k :
                        v.requires_grad=True


        self.optimizer_G.zero_grad()

        self.fake_H_branch, self.fake_H, self.grad_LR = self.netG(self.var_L)

        
        self.fake_H_grad = self.get_grad(self.fake_H)
        self.var_H_grad = self.get_grad(self.var_H)
        self.var_ref_grad = self.get_grad(self.var_ref)
        self.var_H_grad_nopadding = self.get_grad_nopadding(self.var_H)
        

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            
            if self.cri_pix_grad: #gradient pixel loss
                l_g_pix_grad = self.l_pix_grad_w * self.cri_pix_grad(self.fake_H_grad, self.var_H_grad)
                l_g_total += l_g_pix_grad


            if self.cri_pix_branch: #branch pixel loss
                l_g_pix_grad_branch = self.l_pix_branch_w * self.cri_pix_branch(self.fake_H_branch, self.var_H_grad_nopadding)
                l_g_total += l_g_pix_grad_branch


            # G gan + cls loss
            pred_g_fake = self.netD(self.fake_H)
            pred_d_real = self.netD(self.var_ref).detach()
            
            l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            # grad G gan + cls loss
            
            pred_g_fake_grad = self.netD_grad(self.fake_H_grad)
            pred_d_real_grad = self.netD_grad(self.var_ref_grad).detach()

            l_g_gan_grad = self.l_gan_grad_w * (self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_g_fake_grad), False) + 
                                                self.cri_grad_gan(pred_g_fake_grad - torch.mean(pred_d_real_grad), True)) /2
            l_g_total += l_g_gan_grad


            l_g_total.backward()
            self.optimizer_G.step()


        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

        l_d_total = (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.var_ref.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
            interp.requires_grad = True
            interp_crit, _ = self.netD(interp)
            l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit) 
            l_d_total += l_d_gp

        l_d_total.backward()

        self.optimizer_D.step()

        
        for p in self.netD_grad.parameters():
            p.requires_grad = True

        self.optimizer_D_grad.zero_grad()
        l_d_total_grad = 0

        
        pred_d_real_grad = self.netD_grad(self.var_ref_grad)
        pred_d_fake_grad = self.netD_grad(self.fake_H_grad.detach())  # detach to avoid BP to G
        
        l_d_real_grad = self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True)
        l_d_fake_grad = self.cri_grad_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False)

        l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2


        l_d_total_grad.backward()

        self.optimizer_D_grad.step()

        

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

            if self.cri_pix_branch: #branch pixel loss
                self.log_dict['l_g_pix_grad_branch'] = l_g_pix_grad_branch.item()
                
        # D
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()

        # D_grad 
        self.log_dict['l_d_real_grad'] = l_d_real_grad.item()
        self.log_dict['l_d_fake_grad'] = l_d_fake_grad.item()

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        # D_grad outputs
        self.log_dict['D_real_grad'] = torch.mean(pred_d_real_grad.detach())
        self.log_dict['D_fake_grad'] = torch.mean(pred_d_fake_grad.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H_branch, self.fake_H, self.grad_LR = self.netG(self.var_L)
            
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['SR_branch'] = self.fake_H_branch.detach()[0].float().cpu()
        out_dict['LR_grad'] = self.grad_LR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict


    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        self.save_network(self.netD_grad, 'D_grad', iter_step)

class SPSRICCVModel(SRModel):
    """SRGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        def nsml_load(filename):
            save_filename = 'D.pth'
            filename_d = os.path.join(filename, save_filename)
            param_d = torch.load(filename_d)
            print('D loaded!!')
            self.net_d.load_state_dict(param_d, strict=self.opt['path'].get('strict_load_g', True))

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

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            if load_path.startswith('NSML'):
                print(load_path.split('_')[-1], 'KR80934/CVLAB_SR6/{}'.format(load_path.split('_')[-2]))
                nsml.load(checkpoint=load_path.split('_')[-1], load_fn=nsml_load, session='KR80934/CVLAB_SR6/{}'.format(load_path.split('_')[-2]))
                # print(load_net)
                # self.net_g.load_state_dict(load_net, strict=self.opt['path'].get('strict_load_g', True))

            else:
                load_path = os.path.join(DATASET_PATH, load_path)
                param_key = self.opt['path'].get('param_key_d', 'params')
                if param_key == 'None':
                    param_key = None
                print(param_key)
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

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
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
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
        if self.cri_ldl:
            self.output_ema = self.net_g_ema(self.lq)

        if hasattr(self, 'coeff'):
            convert_y = torch.Tensor([65.481/255, 128.553/255, 24.966/255]).reshape(3,1,1).to(self.device)
            self.pos_weight = torch.sum(self.coeff * convert_y, dim=1, keepdims=True) + 16/255

            weight_policy = self.opt['train']['weightpolicy'] if 'weightpolicy' in self.opt['train'] else None
            if weight_policy == 'clamp':
                self.pos_weight = self.pos_weight.clamp(0, 1)
        else:
            self.pos_weight = None

        l_g_total = 0
        loss_dict = OrderedDict()
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
            l_g_total.backward(retain_graph=True)
            
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            
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
