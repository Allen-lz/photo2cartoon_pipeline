import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
def whitebalance(img):
    """
    自动白平衡算法, 但是并没有什么用
    https://www.cnblogs.com/wangyong/p/9119394.html
    """
    rows = img.shape[0]
    cols = img.shape[1]
    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 0])

    for x in range(rows):
        for y in range(cols):
            l, a, b = final[x, y, :]
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * (1 / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (1 / 100.0) * 1.1)

    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    return final

if __name__ == "__main__":
    image_dir = r"E:\3D_face_reconstruct\HRFAE\test\input"
    images = os.listdir(image_dir)
    for name in images:
        # image_path = os.path.join(image_dir, name)
        image_path = os.path.join(image_dir, "9962.png")
        img = cv2.imread(image_path)

        img = cv2.resize(img, (256, 256))

        img_balance = whitebalance(img)

        img_balance = cv2.cvtColor(img_balance, cv2.COLOR_BGR2RGB)

        plt.subplot(121), plt.imshow(img_balance)
        plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()