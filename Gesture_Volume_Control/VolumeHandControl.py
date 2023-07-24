import cv2
import mediapipe as mp
import time
import numpy as np
from Hand_Tracking import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##############################
cam_width, cam_height = 1640, 920
##############################
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

pTime = 0
detector = htm.handDetector(detecConfi=0.7, maxHands=1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()

vol_range = volume.GetVolumeRange()

minVol, maxVol, = vol_range[0], vol_range[1]
first = False
t1 = time.time()
line_length_old = 0
max_length = 0
timer = 0
stop = False
percentage = 0
ih = np.interp(volume.GetMasterVolumeLevel(), [minVol, maxVol], [400, 150])
while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        x3, y3 = lmlist[12][1], lmlist[12][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        p2i_line_length = math.hypot(x3 - x2, y3 - y2)
        line_length = math.hypot(x2 - x1, y2 - y1)
        # Hand range 50 - 300
        # Volume Range -65 0
        if p2i_line_length < 70:
            stop, first = False, False
            timer = 0
            t2, t1 = time.time(), time.time()
            max_length = 0
        if max_length != 0:
            vol = np.interp(line_length, [50, max_length], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
            ih = np.interp(vol, [minVol, maxVol], [400, 150])

        if not stop:
            if first and timer < 2 and line_length_old < line_length:
                line_length_old = line_length
                timer = 0
                t1 = time.time()
            if timer > 2 and first:
                max_length = line_length_old
                stop = True
                print(max_length)
            if timer > 2 and not first:
                timer = 0
                t1 = time.time()
            if line_length < 50 and not first:
                cv2.circle(img, (cx, cy), 20, (0, 255, 255), cv2.FILLED)
                first = True
            elif line_length < 50 and first:
                cv2.circle(img, (cx, cy), 20, (0, 255, 255), cv2.FILLED)

            t2 = time.time()
            timer = t2 - t1
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 10)
    cv2.rectangle(img, (50, int(ih)), (85, 400), (255, 0, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    percentage = np.interp(volume.GetMasterVolumeLevel(), [minVol, maxVol], [0, 100])
    cv2.putText(img, f'FPS: {int(fps)}', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.putText(img, f'{int(percentage)}%', (40, 450), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)