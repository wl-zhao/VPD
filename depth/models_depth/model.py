# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F

from vpd.models import UNetWrapper, TextAdapterDepth

class VPDDepthEncoder(nn.Module):
    def __init__(self, out_dim=1024, ldm_prior=[320, 640, 1280+1280], sd_path=None, text_dim=768,
                 dataset='nyu'
                 ):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        self.apply(self._init_weights)

        ### stable diffusion layers 

        config = OmegaConf.load('./v1-inference.yaml')
        if sd_path is None:
            config.model.params.ckpt_path = '../checkpoints/v1-5-pruned-emaonly.ckpt'
        else:
            config.model.params.ckpt_path = f'../{sd_path}'

        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model

        self.unet = UNetWrapper(sd_model.model, use_attn=False)
    
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

        if dataset == 'nyu':
            self.text_adapter = TextAdapterDepth(text_dim=text_dim)
            class_embeddings = torch.load('nyu_class_embeddings.pth')
        else:
            raise NotImplementedError
        
        self.register_buffer('class_embeddings', class_embeddings)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x, class_ids=None):
        with torch.no_grad():
            latents = self.encoder_vq.encode(x).mode().detach()

        if class_ids is not None:
            class_embeddings = self.class_embeddings[class_ids.tolist()]
        else:
            class_embeddings = self.class_embeddings
        
        c_crossattn = self.text_adapter(latents, class_embeddings, self.gamma)  # NOTE: here the c_crossattn should be expand_dim as latents
        t = torch.ones((x.shape[0],), device=x.device).long()
        # import pdb; pdb.set_trace()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)

class VPDDepth(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth

        embed_dim = 192
        
        channels_in = embed_dim*8
        channels_out = embed_dim

        if args.dataset == 'nyudepthv2':
            self.encoder = VPDDepthEncoder(out_dim=channels_in, dataset='nyu')
        else:
            raise NotImplementedError
            
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x, class_ids=None):    
        # import pdb; pdb.set_trace() 
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]
        if h == 480 and w == 480:
            new_x = torch.zeros(b, c, 512, 512, device=x.device)
            new_x[:, :, 0:480, 0:480] = x
            x = new_x
        elif h==352 and w==352:
            new_x = torch.zeros(b, c, 384, 384, device=x.device)
            new_x[:, :, 0:352, 0:352] = x
            x = new_x
        elif h == 512 and w == 512:
            pass
        else:
            raise NotImplementedError
        conv_feats = self.encoder(x, class_ids)

        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]      

        out = self.decoder([conv_feats])
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels

        # import pdb; pdb.set_trace()
        
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        # import pdb; pdb.set_trace()
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

