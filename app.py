import os

import numpy as np
import torch

import u2net
import cv2
import animegan2
from skimage import io
from face_detector.infer_camera import infer_image, draw_face
import matplotlib.pyplot as plt
from face_parsing.simple_infer import FaceParsing
from photo2cartoon.photo2cartoon_inference import Photo2Cartoon
from agenet.age_prediction.age_perdict import AGEPredict

from utils.histogram_matching import api as color_transfer

from agenet.inference import AGENet

def ratio_resize(img, size=336):
    """
    保持图像的长宽比对图像进行缩放
    :param img:
    :return:
    """
    min_l = min(img.shape[:2])
    if min_l < size:
        ratio = size / min_l
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img

def ratio_resize_large(img, size=512):
    """
    保持图像的长宽比对图像进行缩放
    :param img:
    :return:
    """
    min_l = min(img.shape[:2])

    ratio = size / min_l
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img

def image_stitching(plane, kernel_size = (5, 5), sigma = 1.5):
    """
    对主要的连接处进行高斯模糊处理
    :param plane:
    :return:
    """
    plane = cv2.GaussianBlur(plane, kernel_size, sigma)
    return plane

def get_body_area(image, mask, face_bbox=None):
    """
    :param image:
    :return:
    """
    index = mask.nonzero()
    # print(index[0].shape, index[1].shape)
    min_y = np.min(index[0])
    max_y = np.max(index[0])
    min_x = np.min(index[1])
    max_x = np.max(index[1])

    # plt.imshow(body_img)
    # plt.show()
    body_bbox = [min_x, min_y, max_x, max_y]
    if face_bbox is not None:
        body_bbox[1] = max(0, int(face_bbox[3] - (face_bbox[3] - face_bbox[1]) * 0.2))
    body_img = image[body_bbox[1]:body_bbox[3], body_bbox[0]:body_bbox[2], :]

    return body_img, body_bbox

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    img_dir = "samples/inputs"
    img_list = os.listdir(img_dir)

    # 使用u2net进行人像的抠图
    model_path = "weights/u2net.quant.onnx"
    mask_net = u2net.ONNXModel(model_path)

    # 使用face_parse进行人脸解析
    faceparse_model_path = "weights/face_parse.pth"
    faceparse = FaceParsing(weight=faceparse_model_path)

    # 使用GAN将老年人脸转成20岁的(很多时候不会调用, 只有检测到人脸是老年人的时候才会调用)
    prototxtPathA = "./agenet/logs/models/age_detector/age_deploy.prototxt"
    weightsPathA = "./agenet/logs/models/age_detector/age_net.caffemodel"
    age_perdict = AGEPredict(prototxtPathA, weightsPathA)
    agegan = AGENet(device=device)

    # 使用animegan2进行整张图的动漫化
    checkpoint = "./weights/face_paint_512_v2_0.pt"
    animegan2 = animegan2.Animegan2(checkpoint)

    # 使用腾讯api采集的数据来对脸进行卡通画(或者使用自己的风格对人脸进行卡通化)
    weight_path = "weights/photo2cartoon_weights.pt"
    cartoon = Photo2Cartoon(weight_path)

    for name in img_list:
        img_path = os.path.join(img_dir, name)
        img = io.imread(img_path)

        if img.shape[2] == 4:
            # img = Image.open(img_path).convert("RGB")
            # img = np.array(img, np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # 对输入的图像进行等比例缩放(不然如果机器的显卡不够大的话会爆现存)
        img = ratio_resize_large(img)
        bboxes, _ = infer_image(img)

        # 默认只有一个人
        if bboxes is not None:
            bbox = draw_face(img=img, boxes_c=bboxes)[0]

            # 对整张图进行卡通化
            input_img, h, w = mask_net.preprocess(img)
            masks = mask_net.forward(input_img)
            mask = mask_net.postprocess(masks[0])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (w, h)) / 255
            mask[mask > 0.5] = 1

            origin_out = animegan2.forward(img)
            origin_out = cv2.resize(origin_out, (w, h))

            # 为了获得白色的衣服, 将背景设置为黑色的, 提升前景与背景的对比度
            mask_h, mask_w = mask.shape[:2]
            img_black = np.zeros([mask_h, mask_w, 3], np.uint8)
            img_black[:, :, 1] = np.zeros([mask_h, mask_w])

            kernel = np.ones((9, 9), np.uint8)
            dilation_mask = cv2.dilate(mask, kernel, iterations=1)
            input_img = np.array((1 - dilation_mask) * img_black + dilation_mask * img, dtype=np.uint8)
            # 针对衣服获得相对白一点的输出
            the_whole_out = animegan2.forward(input_img)
            the_whole_out = cv2.resize(the_whole_out, (w, h))

            vis_origin_out = the_whole_out.copy()

            # 然后对人脸区域进行卡通化
            face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            face_h, face_w = face.shape[0], face.shape[1]
            face_input = ratio_resize(face)

            # 对年龄大的人脸进行年轻化，消除一些褶皱
            if age_perdict.forward(face_input):
                face_input = agegan.forward(face_input)

            # 对人脸进行卡通化
            face_out = cartoon.inference(face_input)
            face_out_animegan2 = animegan2.forward(face_input)
            face_out = cv2.resize(face_out, (face_w, face_h))
            face_out_animegan2 = cv2.resize(face_out_animegan2, (face_w, face_h))

            # plt.subplot(131), plt.imshow(face_input)
            # plt.subplot(132), plt.imshow(face_out)
            # plt.subplot(133), plt.imshow(face_out_animegan2)
            # plt.show()

            # 对人脸进行解析, 得到人脸, 面皮, 和五官的mask
            face_mask, eyes_mask, mouth_mask, face_skin = faceparse.forward(face)
            face_mask = np.array(np.repeat(face_mask[:, :, np.newaxis], repeats=3, axis=-1), dtype=np.uint8)
            # cv2的resize需要将numpy中的数据类型转化为uint8的，不然会报错
            face_mask = cv2.resize(face_mask, (face_w, face_h)) / 255

            # 为了在两种风格进行粘贴的时候更加的自然一点, 在两个风格人脸的面皮之间进行肤色的迁移
            # _, _, _, face_skin_out = faceparse.forward(face_out)
            # _, _, _, face_skin_out_animegan2 = faceparse.forward(face_out_animegan2)
            # face_skin_out = np.array(face_skin_out, np.uint8)
            # face_skin_out_animegan2 = np.array(face_skin_out_animegan2, np.uint8)
            # face_skin_out = cv2.resize(face_skin_out, (face_w, face_h)) / 255
            # face_skin_out_animegan2 = cv2.resize(face_skin_out_animegan2, (face_w, face_h)) / 255
            # face_out = color_transfer(face_skin_out, face_skin_out_animegan2, face_out, face_out_animegan2)
            # plt.subplot(131), plt.imshow(face_skin_out)
            # plt.subplot(132), plt.imshow(face_skin_out_animegan2)
            # plt.subplot(133), plt.imshow(face_out)
            # plt.show()

            # plt.subplot(121), plt.imshow(face_skin_out * face_out)
            # plt.subplot(122), plt.imshow(face_skin_out_animegan2 * face_out_animegan2)
            # plt.show()

            # 不使用五官和面部的颜色匹配
            # eyes_mask = cv2.resize(eyes_mask, (face_w, face_h)) / 255
            # mouth_mask = cv2.resize(mouth_mask, (face_w, face_h)) / 255

            # face_out = color_transfer(mouth_mask, mouth_mask, face_out, face)
            # face_out = color_transfer(eyes_mask, eyes_mask, face_out, face)

            # plt.imshow(face_mask)
            # plt.show()

            # 之后对人体区域进行卡通化
            # body_img, body_bbox = get_body_area(img, mask, bbox)
            # body_h, body_w = body_img.shape[0], body_img.shape[1]
            # body_img = ratio_resize(body_img)
            # body_out = animegan2.forward(body_img)
            # body_out = cv2.resize(body_out, (body_w, body_h))
            # body_mask = np.zeros(mask.shape)
            # body_mask[body_bbox[1]:body_bbox[3], body_bbox[0]:body_bbox[2]] = mask[body_bbox[1]:body_bbox[3], body_bbox[0]:body_bbox[2]]
            # out[body_bbox[1]:body_bbox[3], body_bbox[0]:body_bbox[2]] = body_out
            # plt.imshow(body_out)
            # plt.show()

            # plt.subplot(151), plt.imshow(out[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            # plt.subplot(152), plt.imshow(face_out)
            # plt.subplot(153), plt.imshow(face_mask)
            the_whole_out[bbox[1]:bbox[3], bbox[0]:bbox[2]] = face_mask * face_out + (1 - face_mask) * the_whole_out[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # plt.subplot(154), plt.imshow(out[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            # plt.subplot(155), plt.imshow(out)
            # plt.show()

            # 对缝合的部位进行模糊处理
            # plane_box = [max(0, int(bbox[0] - 30)),
            #              max(0, int(bbox[3] - 30)),
            #              min(img.shape[1] - 1, int(bbox[2] + 30)),
            #              min(img.shape[0] - 1, int(bbox[3] + 30))
            #             ]
            # plane = image_stitching(out[plane_box[1]: plane_box[3], plane_box[0]: plane_box[2], :])
            # out[plane_box[1]: plane_box[3], plane_box[0]: plane_box[2], :] = plane

            # plt.subplot(141), plt.imshow(img)
            # vis_mask = mask.copy()
            # cv2.rectangle(vis_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
            # cv2.rectangle(vis_mask, (body_bbox[0], body_bbox[1]), (body_bbox[2], body_bbox[3]), (255, 0, 0), 5)
            # plt.subplot(142), plt.imshow(vis_mask)
            # plt.subplot(143), plt.imshow(vis_origin_out)

            # plt.subplot(141), plt.imshow(mask)
            # plt.subplot(142), plt.imshow(img)
            # plt.subplot(143), plt.imshow(out)
            fusion = np.array(mask * the_whole_out + (1 - mask) * img, np.uint8)
            # plt.subplot(144), plt.imshow(fusion)
            # plt.show()

            # plt.subplot(144), plt.imshow(fusion)
            # plt.show()

            plt.subplot(121), plt.imshow(img)
            plt.subplot(122), plt.imshow(fusion)
            plt.show()

            cv2.imwrite("samples/results/" + name, cv2.cvtColor(fusion, cv2.COLOR_RGB2BGR))

            # plt.imshow(fusion)
            # plt.show()
