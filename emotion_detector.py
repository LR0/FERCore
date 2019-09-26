import cv2
import dlib
from keras.models import load_model
import numpy as np
import os
import sys

from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import make_face_coordinates
from utils.preprocessor import preprocess_input

os.chdir(sys.path[0])
emotion_offsets = (20, 40)
emotion_model_path = './trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'


class EmotionDetector:
    def __init__(self, face_detector=dlib.get_frontal_face_detector(),
                 emotion_classifier=load_model(emotion_model_path, compile=False)):
        self.face_detector = face_detector
        self.emotion_classifier = emotion_classifier

    def detect(self, bgr_image):
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detected_faces, score, idx = detect_faces(self.face_detector, gray_image)

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

        if size == 0:
            return None

        gray_face = gray_image[coord[2]:coord[3], coord[0]:coord[1]]
        emotion_target_size = self.emotion_classifier.input_shape[1:3]
        if np.size(gray_face) is 0:
            return None
        gray_face = cv2.resize(gray_face, emotion_target_size)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        # 返回预测值，返回形式为数组，值是emotion_labels中标签对应下标的值
        emotion_prediction = self.emotion_classifier.predict(gray_face)
        return emotion_prediction
