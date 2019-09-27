import cv2
from keras.models import load_model
import numpy as np
import os
import sys

from face_detectors import DlibFaceDetector
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input

os.chdir(sys.path[0])
emotion_offsets = (20, 40)
# 默认模型地址
emotion_model_path = './trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'


class EmotionDetector:
    def __init__(self, labels=get_labels('fer2013'), face_detector=DlibFaceDetector(),
                 emotion_classifier=load_model(emotion_model_path, compile=False)):
        self.labels = labels
        self.face_detector = face_detector
        self.classifier = emotion_classifier

    def detect_biggest(self, bgr_image):
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_face, coord = self.face_detector.get_biggest_face(gray_image, emotion_offsets)
        emotion_target_size = self.classifier.input_shape[1:3]
        if np.size(gray_face) is 0:
            return None, coord
        gray_face = cv2.resize(gray_face, emotion_target_size)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        # 返回预测值，返回形式为数组，值是emotion_labels中标签对应下标的值
        emotion_prediction = self.classifier.predict(gray_face)
        return emotion_prediction, coord


