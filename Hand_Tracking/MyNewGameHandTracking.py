import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

capture = cv2.VideoCapture(0)
pTime = 0
cTime = 0
detector = htm.handDetector()

while True:
    success, img = capture.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, (50, 100, 30), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)