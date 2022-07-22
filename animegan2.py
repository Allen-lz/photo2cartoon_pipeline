import os
import argparse

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Animegan2(object):
    def __init__(self, checkpoint, device="cuda:0", upsample_align=False):
        super().__init__()
        self.device = device
        self.net = Generator()
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.net.to(device).eval()
        self.upsample_align = upsample_align

    def forward(self, input):

        # 将numpy转image
        input = Image.fromarray(input)
        input = self.preprocess_image(input)
        with torch.no_grad():
            image = to_tensor(input).unsqueeze(0) * 2 - 1
            out = self.net(image.to(self.device), self.upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = np.array(out * 255, np.uint8).transpose((1, 2, 0))
        return out


    def preprocess_image(self, input, x32=False):
        if x32:
            def to_32s(x):
                return 256 if x < 256 else x - x % 32

            w, h = input.size
            input = input.resize((to_32s(w), to_32s(h)))
        return input

