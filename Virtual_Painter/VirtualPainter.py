import cv2
import numpy as np
from Hand_Tracking import HandTrackingModule as htm
import os

folderPath = 'Header'
brush_thickness = 15
eraser_thickness = 50
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
xp, yp = 0, 0
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
color = (255, 0, 0)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
detector = htm.handDetector(detecConfi=0.85)
while True:
    # Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # Find Hand Landmarks
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # print(lmlist)
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]


    # Check the number of fingers are up
        fingers = detector.fingersUp()
    # If in selection mode, then there are two corresponding fingers in the up position
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1,y1-15), (x2,y2+15), (255, 0, 255), cv2.FILLED)
            # checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[3]
                    color = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[2]
                    color = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlayList[0]
                    color = (255, 0, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[1]
                    color = (0, 0, 0)
    # If drawing mode, Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), color, eraser_thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), color, brush_thickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, brush_thickness)
            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)