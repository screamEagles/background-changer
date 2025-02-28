import cv2
import cvzone  # cvzone version: 1.3.5
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fps_reader = cvzone.FPS()

list_img = os.listdir("images")

img_list = []
for img_path in list_img:
    img = cv2.imread(f"images/{img_path}")
    img_list.append(img)

index_img = 0

while True:
    success, img = cap.read()
    img_out = segmentor.removeBG(img, img_list[index_img], threshold=0.8)

    img_stacked = cvzone.stackImages([img, img_out], 2, 1)
    _, img_stacked = fps_reader.update(img_stacked, color=(255, 255, 0))

    cv2.imshow("Image", img_stacked)
    key = cv2.waitKey(1)
    if key == ord("a"):
        if index_img > 0:
            index_img -= 1
    if key == ord("d"):
        if index_img < len(img_list) - 1:
            index_img += 1
    if key == ord("q"):
        break
