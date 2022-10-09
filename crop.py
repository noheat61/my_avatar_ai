import cv2
from PIL import Image
import os


def get_len(center, long, short):
    center = max(center, short / 2)
    center = min(center, long - short / 2)
    return int(center - short / 2)


def main(filename):
    print("시작")

    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        print("file end withs jpg/png..")
        return None

    image1 = Image.open(filename)
    width, height = image1.size
    if min(width, height) < 256:
        print(width, height)
        print("photo size more than 256..")
        return None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("cannot identify face..")
        return None

    face = faces[0]
    x_coor = face[0] + face[2] * 0.5
    y_coor = face[1] + face[3] * 0.4

    # 좌표 계산
    x = get_len(x_coor, width, height) if height < width else 0
    y = get_len(y_coor, height, width) if height > width else 0
    w = min(width, height)
    h = min(width, height)

    cropped_img = image1.crop((x, y, x + w, y + h))
    # cropped_img.show()
    img_name, extension = os.path.splitext(filename)
    cropped_path = img_name + "-crop" + extension

    cropped_img.save(cropped_path)
    return cropped_path
