import cv2
import time
import os
from Hand_Tracking import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1600)
cap.set(4, 920)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
# print(myList)
overlay_list = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlay_list.append(image)

detector = htm.handDetector(detecConfi=0.7)
totalFingers = 0
pTime = 0
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        fingers = []
        # thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            # 4 fingers
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)

    height, width, channel = overlay_list[totalFingers-1].shape
    img[125:125+height, 50:50+width] = overlay_list[totalFingers-1]

    cTime = time.time()
    fps = 1 // (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{fps}', (10, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
    cv2.imshow("Video", img)
    cv2.waitKey(1)