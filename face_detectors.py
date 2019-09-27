from abc import ABCMeta, abstractmethod

import dlib

from utils.inference import make_face_coordinates, apply_offsets


class IFaceDetector(metaclass=ABCMeta):
    @abstractmethod
    def detect_faces(self, gray_image):
        pass

    @abstractmethod
    def get_biggest_face(self, gray_image, emotion_offsets):
        pass


class DlibFaceDetector(IFaceDetector):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, gray_image_array):
        # 下面这条狗屎会使得每调用一次函数就构建一个frontal_face_detector模型对象，只执行一次run方法然后释放
        # return dlib.get_frontal_face_detector().run(gray_image_array, 0, 0) <-----拖慢程序速度，造成cpu占用率下降的元凶
        return self.detector.run(gray_image_array, 0, 0)  # 正确的写法，只构建一次模型对象，保存下来重复使用

    def get_biggest_face(self, gray_image, emotion_offsets):
        detected_faces, score, idx = self.detect_faces(gray_image)
        size = 0
        coord = [0, 0, 0, 0]
        # 选出最大的人脸
        for detected_face in detected_faces:
            face_coordinates = make_face_coordinates(detected_face)
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            new_size = abs((x1 - x2) * (y1 - y2))
            if new_size > size:
                coord[0] = x1
                coord[1] = x2
                coord[2] = y1
                coord[3] = y2
                size = new_size

        return gray_image[coord[2]:coord[3], coord[0]:coord[1]], coord
