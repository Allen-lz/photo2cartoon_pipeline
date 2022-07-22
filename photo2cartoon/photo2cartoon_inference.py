import os
import cv2
import torch
import numpy as np
from photo2cartoon.generator import ResnetGenerator
import matplotlib.pyplot as plt

class Photo2Cartoon:
    def __init__(self, weight_path):
        # self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        assert os.path.exists(weight_path), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        # print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None

        # print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face * mask + (1 - mask) * 255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        # cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        # print('[Step3: photo to cartoon] success!')
        return cartoon


if __name__ == '__main__':
    test_img = r"E:/styletransfer/animegan2/samples/inputs/bobo.png"
    img = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(img)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    if cartoon is not None:
        plt.subplot(121), plt.imshow(img)
        plt.subplot(122), plt.imshow(cartoon)
        plt.show()
        print('Cartoon portrait has been saved successfully!')
