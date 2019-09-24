import cv2
import dlib
from keras.engine.saving import load_model

from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')
emotion_model_path = './FERProject/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
test_video_path = "./FERProject/test_videos/case2.mp4"


def main():
    face_detector = dlib.get_frontal_face_detector()
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_detector = EmotionDetector(face_detector, emotion_classifier)
    # cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(test_video_path)
    while True:
        bgr_image = video_capture.read()[1]
        try:
            emotion_prediction = emotion_detector.detect(bgr_image)
        except Exception as ex:
            print(ex)
            continue

        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        if emotion_prediction is not None:
            print(str(timestamp) + " " + str(emotion_prediction[0]))


if __name__ == '__main__':
    main()
