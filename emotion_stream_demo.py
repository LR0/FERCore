import datetime

from detect_functions import get_emotion_stream
from detect_functions import get_emotion_stream_json
from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')
test_video_path = '../cases/case1.mp4'
output_text_path = '../cases/output/case1_emotion_stream.txt'


def main():
    case1()


def case1():
    emotion_stream = get_emotion_stream(test_video_path, EmotionDetector(), 1000)
    f = open(output_text_path, 'w')
    for frame_emotion in emotion_stream:
        f.write(str(frame_emotion.time / 1000) + 's' + '\n')
        f.write(str(frame_emotion.prediction) + '\n')
    f.close()


def case2():
    emotion_stream_json = get_emotion_stream_json(test_video_path, EmotionDetector(), 1000)
    f = open(output_text_path, 'w')
    f.write(emotion_stream_json)
    f.close()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print(end - start)
