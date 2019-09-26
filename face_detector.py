import dlib

from utils.inference import detect_faces, make_face_coordinates, apply_offsets


def get_biggest_face(gray_image, emotion_offsets, face_detector=dlib.get_frontal_face_detector()):
    detected_faces, score, idx = detect_faces(face_detector, gray_image)
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

    return gray_image[coord[2]:coord[3], coord[0]:coord[1]], coord
