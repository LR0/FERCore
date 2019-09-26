import datetime

from detect_functions import get_emotion_stream
from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')
test_video_path = 'C:/Users/admin/Videos/cases/case4.mp4'
output_text_path = 'C:/Users/admin/Videos/cases/output/case4_emotion_stream.txt'


def main():
    f = open(output_text_path, 'w')
    emotion_stream = get_emotion_stream(test_video_path, 1000, EmotionDetector())
    for frame_emotion in emotion_stream:
        print(str(frame_emotion.time / 1000) + 's')
        print(str(frame_emotion.prediction))
        f.write(str(frame_emotion.time / 1000) + 's' + '\n')
        f.write(str(frame_emotion.prediction) + '\n')
    f.close()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print(end - start)
