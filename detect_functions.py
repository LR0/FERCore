import datetime
import cv2
import numpy as np

from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')


class FrameEmotion:
    def __init__(self, time, prediction):
        self.time = time
        self.prediction = prediction


def get_emotion_stream(video_path, frame_interval_ms, detector):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    emotion_stream = []
    if not video_capture.isOpened():
        video_capture.release()
        return emotion_stream

    frame_no = 1  # 抽取帧的序号
    interval_frame_num = int(frame_interval_ms / 1000 * fps)  # 间隔帧数
    while frame_no <= frame_count:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        frame = video_capture.read()[1]
        if frame is None or np.size(frame) is 0:
            frame_no += interval_frame_num
            continue
        time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        prediction = detector.detect(frame)
        frame_emotion = FrameEmotion(time, prediction)
        emotion_stream.append(frame_emotion)
        frame_no += interval_frame_num

    return emotion_stream
