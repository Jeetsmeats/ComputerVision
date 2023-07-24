import math
import cv2
import numpy as np
from Hand_Tracking import HandTrackingModule as htm
import time
import pyautogui

height, width = 480, 640
screen_width, screen_height = pyautogui.size()
smoother = 20

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
detector = htm.handDetector()
pTime = 0
frame_reduc = 100 # frame reduction

while True:
    # 1 Find the hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)

    # 2 Get the tip of the index and middle fingers
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[12][2]
    # 3 Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frame_reduc, frame_reduc),
                      (width - frame_reduc, height - frame_reduc), (255, 0, 255), 2)
    # 4 Check if in moving mode
        if fingers[1] and not fingers[2]:

    # 5 Convert Coordinates to screen resolution
            x3 = np.interp(x1, (frame_reduc, width-frame_reduc), (0, screen_width))
            y3 = np.interp(y1, (frame_reduc, height-frame_reduc), (0, screen_height))

    # 6 Smoothen values
            curr_x = prev_x + (x3 - prev_x) / smoother
            curr_y = prev_y + (y3 - prev_y) / smoother
    # 7 Move mouse
            pyautogui.moveTo(abs(screen_width-x3),y3)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y
    # 8 Check if we are in clicking mode (index and middle fingers are up)
        if fingers[1] and fingers[2]:
            # 9  Find distance between fingers
            length = math.hypot( x1 - x2, y1 - y2)
            if length < 40:
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (mid_x, mid_y), 15, (0, 255, 0), cv2.FILLED)
                # 10 Click mouse is distance is short
                pyautogui.click()


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)