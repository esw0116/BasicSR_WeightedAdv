import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import block as B
from . import spectral_norm as SN


class SPSRNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(SPSRNet, self).__init__()

        n_upscale = int(math.log(upscale, 2))

        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]        
        
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        self.HR_conv0_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, self.HR_conv0_new)

        self.get_g_nopadding = Get_gradient_nopadding()

        self.b_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_concat_1 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_1 = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')
        

        self.b_concat_2 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_2 = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')


        self.b_concat_3 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_3 = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')


        self.b_concat_4 = B.conv_block(2*nf, nf, kernel_size=3, norm_type=None, act_type = None)
        self.b_block_4 = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        b_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        b_HR_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)

        self.conv_w = B.conv_block(nf, out_nc, kernel_size=1, norm_type=None, act_type=None)

        self.f_concat = B.conv_block(nf*2, nf, kernel_size=3, norm_type=None, act_type=None)

        self.f_block = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')

        self.f_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.f_HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        

    def forward(self, x):    

        x_grad = self.get_g_nopadding(x)
        x = self.model[0](x)  

        x, block_list = self.model[1](x)

        x_ori = x
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x 

        for i in range(5):
            x = block_list[i+5](x)
        x_fea2 = x

        for i in range(5):
            x = block_list[i+10](x)
        x_fea3 = x
        
        for i in range(5):
            x = block_list[i+15](x)
        x_fea4 = x
        
        x = block_list[20:](x)
        #short cut
        x = x_ori+x
        x= self.model[2:](x)
        x = self.HR_conv1_new(x)

        x_b_fea = self.b_fea_conv(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)
        
        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1)
        
        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)

        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)
        
        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)

        x_cat_4 = torch.cat([x_cat_3, x_fea4], dim=1)
        
        x_cat_4 = self.b_block_4(x_cat_4)
        x_cat_4 = self.b_concat_4(x_cat_4)

        x_cat_4 = self.b_LR_conv(x_cat_4)

        #short cut
        x_cat_4 = x_cat_4+x_b_fea
        x_branch = self.b_module(x_cat_4)

        x_out_branch = self.conv_w(x_branch)
        ########
        x_branch_d = x_branch
        x_f_cat = torch.cat([x_branch_d, x], dim=1)
        x_f_cat = self.f_block(x_f_cat)
        x_out = self.f_concat(x_f_cat)
        x_out = self.f_HR_conv0(x_out)
        x_out = self.f_HR_conv1(x_out)
        
        #########
        return x_out_branch, x_out, x_grad