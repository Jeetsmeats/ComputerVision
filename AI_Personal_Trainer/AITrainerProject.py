import cv2
import numpy as np
import time
from Pose_Estimation import PoseEstimationModule as pem

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("WorkoutVideos/bicep_curl.mp4")

detector = pem.PoseEstimation()

count = 0
dir = 0
pTime = 0
while True:
    success, img = cap.read()

    img = detector.applyPose(img, False)
    lmlist = detector.findPositions(img, False)

    if len(lmlist) != 0:
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        # right arm
        # detector.findAngle(img, 12, 14, 16)
        # percentage of curl
        per = np.interp(angle, (0, 180), (0, 100))
    #   percentage in bar
    #     check for the dumbbell curls
        if per < 100 and per > 80:
            if dir == 0:
                count += 0.5
                dir = 1
        if per < 80:
            if dir == 1:
                count += 0.5
                dir = 0
        cv2.rectangle(img, (70, 250), (325, 720), (0, 255, 0), cv2.FILTER_SCHARR)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 0, 0), 5)
        cv2.putText(img, f'{count}', (50, 100), cv2.FONT_HERSHEY_COMPLEX,
                    2, (255, 0, 0), 5)
        cTime = time.time()
        fps = 1 // (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(fps), (10, 1500), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)