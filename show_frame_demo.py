import datetime

from detect_functions import save_biggest_emotion_images, save_biggest_emotion_images_cut
from emotion_detector import EmotionDetector
from utils.datasets import get_labels

emotion_labels = get_labels('fer2013')
test_video_path = 'C:/Users/admin/Videos/cases/case1.mp4'
save_image_path = 'C:/Users/admin/Videos/cases/output/case_images_1010'


start = datetime.datetime.now()
save_biggest_emotion_images(test_video_path, save_image_path, EmotionDetector(), 1000)
end = datetime.datetime.now()
print(end - start)




