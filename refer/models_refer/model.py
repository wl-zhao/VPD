import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from ldm.util import instantiate_from_config

from omegaconf import OmegaConf
from lib.mask_predictor import SimpleDecoding

from vpd.models import UNetWrapper, TextAdapterRefer

class VPDRefer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 sd_path=None,
                 base_size=512,
                 token_embed_dim=768,
                 neck_dim=[320,680,1320,1280],
                 **args):
        super().__init__()
        config = OmegaConf.load('./v1-inference.yaml')
        config.model.params.ckpt_path = f'{sd_path}'
        # import pdb; pdb.set_trace()
        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, base_size=base_size)
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder

        self.text_adapter = TextAdapterRefer(text_dim=token_embed_dim)

        self.classifier = SimpleDecoding(dims=neck_dim)

        self.gamma = nn.Parameter(torch.ones(token_embed_dim) * 1e-4)

    def forward(self, img, l_feats):
        input_shape = img.shape[-2:]
        with torch.no_grad():
            latents = self.encoder_vq.encode(img).mode().detach()
        c_crossattn = self.text_adapter(latents, l_feats, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
        t = torch.ones((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        
        x_c1, x_c2, x_c3, x_c4 = outs  
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x
