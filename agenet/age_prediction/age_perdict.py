import cv2
class AGEPredict(object):
    def __init__(self, prototxtPathA, weightsPathA):
        # 年龄检测模型
        self.ageNet = cv2.dnn.readNet(prototxtPathA, weightsPathA)

    def forward(self, face):
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)
        # 预测年龄
        self.ageNet.setInput(faceBlob)
        predictions = self.ageNet.forward()
        i = predictions[0].argmax()
        if i >= 5:
            return True

        return False