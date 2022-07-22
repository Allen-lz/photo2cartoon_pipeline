import sys
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import cv2
def calc_saturation(diff, slope, limit):
    """
    r(x)
    slope是斜率, 可能是表示增强的强度
    """
    ret = diff * slope
    if ret > limit:
        ret = limit
    elif ret < -limit:
        ret = -limit
    return ret

def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500):
    """
    自动色彩均衡: 好像没有什么用
    https://www.cnblogs.com/wangyong/p/9119394.html
    """
    nimg = nimg.transpose(2, 0, 1)
    # Convert input to an ndarray with column-major memory order(仅仅是地址连续，内容和结构不变)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)

    width = nimg.shape[2]
    height = nimg.shape[1]

    cary = []

    # 随机产生索引(随机检索500个像素点进行计算)
    for i in range(0, samples):

        # 得到随机出来的点的位置并记录在cary的dict中
        _x = random.randint(0, width) % width
        _y = random.randint(0, height) % height

        dict={"x": _x, "y": _y}
        cary.append(dict)


    mat=np.zeros((3,height,width),float)


    # 再获得三个通道的最大值\最小职
    r_max = sys.float_info.min
    r_min = sys.float_info.max

    g_max = sys.float_info.min
    g_min = sys.float_info.max

    b_max = sys.float_info.min
    b_min = sys.float_info.max

    # 遍历图像中的每个位置
    for i in range(height):
          for j in range(width):
             r = nimg[0, i, j]
             g = nimg[1, i, j]
             b = nimg[2, i, j]

             r_rscore_sum = 0.0
             g_rscore_sum = 0.0
             b_rscore_sum = 0.0
             denominator = 0.0

             # 每个位置上的像素都要和sample中的像素进行计算
             for _dict in cary:
                  _x=_dict["x"]  # width
                  _y=_dict["y"]  # height

                  # 计算像素之间的欧氏距离(这个dist应该会作为一个权重)
                  dist = np.sqrt(np.square(_x-j)+np.square(_y-i))
                  # 如果两个像素隔得太远就没意义了
                  if dist < height / 5:
                      continue

                  # 获得当前sample的像素值
                  _sr = nimg[0, _y, _x]
                  _sg = nimg[1, _y, _x]
                  _sb = nimg[2, _y, _x]

                  # 计算三个通道上像素值差值, 然后再做一个分段函数的映射
                  # 这里除以一个dist表示距离越远的差值占的比重就越小
                  r_rscore_sum += calc_saturation(int(r) - int(_sr), slope, limit) / dist
                  g_rscore_sum += calc_saturation(int(g) - int(_sg), slope, limit) / dist
                  b_rscore_sum += calc_saturation(int(b) - int(_sb), slope, limit) / dist

                  # 这是一个归一化系数(两个像素距离越近的话, denominator的值就会越大)
                  # denominator是之后用来进行归一化的, (limit / dist) >= calc_saturation(int(x) - int(_sx), slope, limit) / dist
                  denominator += (limit / dist)

             # 将值映射到(-1, 1)
             r_rscore_sum = r_rscore_sum / denominator
             g_rscore_sum = g_rscore_sum / denominator
             b_rscore_sum = b_rscore_sum / denominator

             mat[0, i, j] = r_rscore_sum
             mat[1, i, j] = g_rscore_sum
             mat[2, i, j] = b_rscore_sum

             # 限制像素值的大小
             if r_max<r_rscore_sum:
                 r_max=r_rscore_sum
             if r_min>r_rscore_sum:
                 r_min=r_rscore_sum

             if g_max<g_rscore_sum:
                 g_max=g_rscore_sum
             if g_min>g_rscore_sum:
                 g_min=g_rscore_sum

             if b_max<b_rscore_sum:
                 b_max=b_rscore_sum
             if b_min>b_rscore_sum:
                 b_min=b_rscore_sum

    for i in range(height):
        for j in range(width):
            nimg[0, i, j] = (mat[0, i, j] - r_min) * 255 / (r_max - r_min)
            nimg[1, i, j] = (mat[1, i, j] - g_min) * 255 / (g_max - g_min)
            nimg[2, i, j] = (mat[2, i, j] - b_min) * 255 / (b_max - b_min)

    return nimg.transpose(1, 2, 0).astype(np.uint8)


if __name__ == "__main__":
    image_dir = r"E:\3D_face_reconstruct\HRFAE\test\input"
    images = os.listdir(image_dir)
    for name in images:
        # image_path = os.path.join(image_dir, name)
        image_path = os.path.join(image_dir, "9962.png")
        img = cv2.imread(image_path)

        img = cv2.resize(img, (256, 256))

        img_balance = automatic_color_equalization(img)

        plt.imshow(cv2.cvtColor(img_balance, cv2.COLOR_BGR2RGB))
        plt.show()