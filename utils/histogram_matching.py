import numpy as np
import torch
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib


def vis_hist(hist):
    """
    绘制水平条形图方法barh
    参数一：y轴
    参数二：x轴
    """
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = list(range(len(hist)))  # 横坐标刻度显示值

    # plt.subplot(121), plt.bar(label_list, hist)

    pdf = hist.copy()
    for i in range(1, 256):
        # 这个从0~255的像素值开始累加是什么意思我有点蒙蔽(是为了增加特征的表达能力吗)
        pdf[i] = pdf[i - 1] + pdf[i]
    # plt.subplot(122), plt.bar(label_list, pdf)
    # plt.show()

def cal_hist(image):
    """
    image: numpy格式的
    cal cumulative hist for channel list
    """
    hists = []
    # 计算每一个通道的直方图
    for i in range(0, 3):
        channel = image[i]
        # channel = image[i, :, :]
        channel = torch.FloatTensor(channel)
        # hist, _ = np.histogram(channel, bins=256, range=(0,255))
        # hist中的下标记录的是像素的值, 里面的元素表示的是像素的频度
        hist = torch.histc(channel, bins=256, min=0, max=256)
        hist = hist.numpy()
        # refHist=hist.view(256,1)
        # 对直方图进行可视化
        # vis_hist(hist)
        # 计算一个channel上的像素的个数
        sum = hist.sum()
        # 对每个像素的频度进行归一化
        pdf = [v / sum for v in hist]
        # 从0~255开始遍历
        for i in range(1, 256):
            # 这个从0~255的像素值开始累加是什么意思我有点蒙蔽(是为了增加特征的表达能力吗)
            """
            这么做的主要原理是:
            因为人脸中颜色的分别是相对
            可以知道像素所处的位置(频率档位), 方便在ref和dst中进行颜色的迁移
            (不知道其在颜色匹配上的是否work)
            """
            pdf[i] = pdf[i - 1] + pdf[i]

            # vis_hist(pdf)

        hists.append(pdf)
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            # 就是ref上直方图的值是在adj[j - 1] ~ adj[j]之间的, 将其的值赋值给table
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        index应该是mask, 记录了在transfer的时候dstImg和refImg上的有效像素
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    # index = [x.cpu().numpy() for x in index]
    index = [x for x in index]

    # tensor转numpy
    dstImg = dstImg.detach().cpu().numpy()
    refImg = refImg.detach().cpu().numpy()

    # index[2], index[3] ---> ref (x, y)
    # index[0], index[1] ---> dst (x, y)
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]

    # 计算直方图
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)

    # 进行颜色的迁移
    # 将ref上的颜色迁移到dst的对应位置上
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    mid = copy.deepcopy(dst_align)
    for i in range(0, 3):
        for k in range(0, len(index[0])):
            # int(mid[i][k]): dst_align位置k上的像素值
            # 将dst_align位置k上的像素值替换成tables[i][int(mid[i][k])]
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = dst_align[i]

    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg

def api(mask1, mask2, face1, face2):
    # test_image1 = r"test_images\face1.png"
    # test_image2 = r"test_images\face2.png"
    #
    # face1 = cv2.cvtColor(cv2.imread(test_image1), cv2.COLOR_BGR2RGB)
    # face2 = cv2.cvtColor(cv2.imread(test_image2), cv2.COLOR_BGR2RGB)
    #
    # mask1_path = r"test_images\mask1.png"
    # mask2_path = r"test_images\mask2.png"
    #
    # mask1 = cv2.imread(mask1_path)
    # mask2 = cv2.imread(mask2_path)
    #
    # # 可视化
    # plt.subplot(121), plt.imshow(face1)
    # plt.subplot(122), plt.imshow(face2)
    # plt.show()
    # face1 = np.array(face1 * mask1[:, :, None], dtype=np.uint8)
    # face2 = np.array(face2 * mask2[:, :, None], dtype=np.uint8)

    # plt.subplot(221), plt.imshow(mask1)
    # plt.subplot(222), plt.imshow(face1)
    # plt.subplot(223), plt.imshow(mask2)
    # plt.subplot(224), plt.imshow(face2)
    # plt.show()

    indexes = []
    index_1 = mask1.nonzero()
    index_2 = mask2.nonzero()

    indexes.append(index_1[0])
    indexes.append(index_1[1])
    indexes.append(index_2[0])
    indexes.append(index_2[1])

    input_face1 = torch.Tensor(face1).permute(2, 0, 1)
    input_face2 = torch.Tensor(face2).permute(2, 0, 1)

    dstImg = histogram_matching(input_face1, input_face2, indexes)

    dstImg = np.array(dstImg.detach().cpu().numpy().transpose(1, 2, 0), np.uint8)

    # plt.subplot(131), plt.imshow(face1)
    # plt.subplot(132), plt.imshow(face2)
    # plt.subplot(133), plt.imshow(dstImg)
    #
    # plt.imshow(dstImg)
    # plt.show()

    return dstImg






