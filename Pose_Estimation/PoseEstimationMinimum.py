import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture("PoseVideos/WorkoutPose.mp4")
pTime = 0
lmlist = []
while True:

    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN,
                12, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)