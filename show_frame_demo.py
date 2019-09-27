import datetime

from detect_functions import save_emotion_images
from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')
test_video_path = 'C:/Users/admin/Videos/cases/case7.mp4'
save_image_path = 'C:/Users/admin/Videos/cases/output/case7_images'


start = datetime.datetime.now()
save_emotion_images(test_video_path, 1000, EmotionDetector(), save_image_path)
end = datetime.datetime.now()
print(end - start)




