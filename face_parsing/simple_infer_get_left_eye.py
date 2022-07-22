#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np

import sys
sys.path.append(".")
sys.path.append("..")

from face_parsing.model import BiSeNet
import torch
import matplotlib.pylab as plt
import os
from PIL import Image
import torchvision.transforms as transforms
import cv2

class FaceParsing:
    def __init__(self, weight):
        self.net = BiSeNet(n_classes=19)
        self.net.cuda()
        self.net.load_state_dict(torch.load(weight))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_square(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        down_pad = w - h
        bbox[3] += down_pad

    def get_bbox(self, contour):
        """
        通过轮廓来得到bbox
        :return:
        """

        contour = contour.reshape(-1, 2)
        xmin = np.min(contour[:, 0])
        ymin = np.min(contour[:, 1])

        xmax = np.max(contour[:, 0])
        ymax = np.max(contour[:, 1])

        bbox = np.array([xmin, ymin, xmax, ymax])

        self.get_square(bbox)

        return bbox

    def get_counter(self, binary, img):
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 轮廓绘制
        # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

        eye_bboxes = []
        for contour in contours:
            bbox = self.get_bbox(contour)
            # img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            eye_bboxes.append(bbox[np.newaxis, :])
        eye_bboxes = np.concatenate(eye_bboxes, axis=0)

        # 得到左眼
        left_eye = self.get_crop_img(eye_bboxes, img)

        return left_eye

    def get_crop_img(self, eye_bboxes, img):
        crop_imgs = [0] * len(eye_bboxes)
        for i, bbox in enumerate(eye_bboxes):
            crop_imgs[i] = img[int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2]), :]

        return crop_imgs[1]

    def forward(self, img_cv):
        with torch.no_grad():
            # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(img_cv)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            parsing[parsing == 14] = 0
            parsing[parsing == 16] = 0
            parsing[parsing == 17] = 0

            # 4, 5 眼睛
            eyes_mask = parsing.copy()
            eyes_mask[eyes_mask == 4] = 255
            eyes_mask[eyes_mask == 5] = 255
            eyes_mask[eyes_mask != 255] = 0
            eyes_mask = np.array(eyes_mask, dtype=np.uint8)

            if np.sum(eyes_mask) > 100:
                resize_img = cv2.resize(img_cv, (512, 512))
                kernel = np.ones((33, 33), np.uint8)
                eyes_mask_dilate = cv2.dilate(eyes_mask // 255, kernel)  # 腐蚀
                left_eye = self.get_counter(eyes_mask_dilate, resize_img)
                return left_eye

            else:
                return 0

if __name__ == "__main__":
    # 使用face_parse进行人脸解析
    faceparse_model_path = "weights/face_parse.pth"
    faceparse = FaceParsing(weight=faceparse_model_path)
    img_dir = r"E:\datasets\frontalization_male"
    # img_dir = r"E:\datasets\real_face\glass"
    images = os.listdir(img_dir)
    for name in images:
        image_path = os.path.join(img_dir, name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_img = faceparse.forward(img)
        if isinstance(crop_img, int):
            continue
        else:
            plt.imshow(crop_img)
            plt.show()


