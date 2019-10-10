import cv2
import json
import numpy as np

from emotion_detector import EmotionDetector
import detect_functions

emotionDetector = EmotionDetector()


# 根据固定时长抽帧并分析
def get_emotion_stream_json(video_path, frame_interval_ms):
    return detect_functions.get_emotion_stream_json(video_path, emotionDetector, frame_interval_ms)

