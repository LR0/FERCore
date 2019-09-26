import datetime

import cv2
import dlib
from keras.engine.saving import load_model

from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')
# emotion_model_path = './FERProject/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
test_video_path = 'C:/Users/admin/Videos/cases/case8.mp4'
output_text_path = 'C:/Users/admin/Videos/cases/output/case8_emotion_stream.txt'


def main():
    # face_detector = dlib.get_frontal_face_detector()
    # emotion_classifier = load_model(emotion_model_path, compile=False)
    # emotion_detector = EmotionDetector(face_detector, emotion_classifier)
    emotion_detector = EmotionDetector()
    video_capture = cv2.VideoCapture(test_video_path)
    file = open(output_text_path, "w")
    total_frame_number = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    while total_frame_number > 0:
        total_frame_number -= 1
        bgr_image = video_capture.read()[1]
        if(bgr_image is None) or ():
            break
        try:
            emotion_prediction = emotion_detector.detect(bgr_image)
        except Exception as ex:
            print(ex)
            continue

        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        if emotion_prediction is not None:
            # print(str(timestamp) + " " + str(emotion_prediction[0]))
            file.write(str(timestamp) + " " + str(emotion_prediction[0]))
            file.write("\n")

    file.close()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print(end - start)
