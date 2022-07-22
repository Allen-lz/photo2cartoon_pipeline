"""
先对人脸进行age的判断, 只有在大于[25 - 32]的这个档位的人脸才需要进行年龄编辑和磨皮美白(可选)
"""
import yaml
import sys
sys.path.append(".")
sys.path.append("..")
from agenet.trainer import *
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

class AGENet(object):
    def __init__(self, device):
        config = yaml.load(open("agenet/configs/001.yaml", 'r'), Loader=yaml.FullLoader)
        img_size = (config['input_w'], config['input_h'])
        # Initialize trainer
        self.trainer = Trainer(config)
        self.device = device
        # Load pretrained model
        self.trainer.load_checkpoint("agenet/logs/001/checkpoint")
        self.trainer.to(self.device)
        self.resize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        self.normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1, 1, 1])
        self.target_age = 20

    # Preprocess
    def preprocess(self, img):
        # 换成numpy格式的输入, 并将BGR ---> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img = self.resize(img_pil)
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim=0)
        img = self.normalize(img)
        return img

    def beauty_face(self, img, v1=5, v2=1, p=0.1):
        dx = v1 * 5  # 双边滤波参数之一
        fc = v1 * 12.5  # 双边滤波参数之一
        temp1 = cv2.bilateralFilter(img, dx, fc, fc)
        temp2 = cv2.subtract(temp1, img)
        temp2 = cv2.add(temp2, (10, 10, 10, 128))
        temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
        temp4 = cv2.add(img, temp3)
        dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
        dst = cv2.add(dst, (10, 10, 10, 255))
        return dst

    def forward(self, img):
        with torch.no_grad():
            image_A = self.preprocess(img)
            image_A = image_A.unsqueeze(0).to(self.device)
            age_modif = torch.tensor(self.target_age).unsqueeze(0).to(self.device)
            # 这里的推理是支持batch的推理的, 所以返回的是一个list
            image_A_modif = self.trainer.test_eval(image_A, age_modif, target_age=self.target_age, hist_trans=True)
            res_img_list = clip_img(image_A_modif)

            # 这个滤波美白磨皮算法会使得图片有点失真, 所以这里先不用
            if False:
                img = self.beauty_face(
                    res_img_list[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

            img = res_img_list[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            return img



