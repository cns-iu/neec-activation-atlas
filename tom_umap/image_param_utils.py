from auto_image_param import BaseImageParam

import cv2 
import torch
import numpy as np


from utils import (
    lucid_colorspace_to_rgb, 
    normalize,
    get_fft_scale_custom_img,
    denormalize,
    rgb_to_lucid_colorspace,
    chw_rgb_to_fft_param,
    fft_to_rgb_custom_img
)

class ImageParam(BaseImageParam):
    def __init__(self, image, device):
        super().__init__()
        self.device = device
        if isinstance(image, str):
            self.image_path = image
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)/255.
            image = torch.tensor(image).permute(-1,0,1).unsqueeze(0)
        self.set_param(image)

    def normalize(self,x, device):
        return normalize(x = x, device= device)

    def postprocess(self, device):
        out = fft_to_rgb_custom_img(height = self.height, width = self.width, image_parameter= self.param, device= device)
        out = lucid_colorspace_to_rgb(t = out, device= device).clamp(0,1)
        return out

    def forward(self, device):
        return self.normalize(self.postprocess(device = device), device= device).clamp(0,1)

    def to_chw_tensor(self, device = 'cpu'):
        t = self.forward(device= device).squeeze(0).clamp(0,1).detach()
        return t
        
    def to_hwc_tensor(self, device = 'cpu'):
        t = self.forward(device= device).squeeze(0).clamp(0,1).permute(1,2,0).detach()
        return t

    def to_nchw_tensor(self, device = 'cpu'):
        t = self.forward(device= device).clamp(0,1).detach()
        return t

    def set_param(self, tensor):
        assert len(tensor.shape) == 4
        self.height, self.width = tensor.shape[-2], tensor.shape[-1]
        self.param = chw_rgb_to_fft_param(tensor.squeeze(0), device = self.device)  / get_fft_scale_custom_img(h = self.height, w = self.width, device= self.device)
        self.param.requires_grad_()
        self.optimizer = None
