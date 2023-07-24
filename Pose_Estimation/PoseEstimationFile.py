import PoseEstimationModule as peMod
import cv2
import time

detector = peMod.PoseEstimation()
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
option = input("Type in what you want to do... ")

if option.isdigit():
    capture = cv2.VideoCapture(int(option))
else:
    capture = cv2.VideoCapture(option)
pTime = 0

while True:

    success, img = capture.read()
    if not success:
        break

    markers = detector.findPositions(img)

    if len(markers) != 0:
        print(markers)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (50, 100, 30), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)