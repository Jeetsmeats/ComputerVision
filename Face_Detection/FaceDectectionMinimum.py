import cv2
import mediapipe as mp
import time

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("FaceVideos/TrainWoman.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.7)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):

            bboxC = detection.location_data.relative_bounding_box
            iheight, iwidth, ichannel = img.shape
            bbox = int(bboxC.xmin * iwidth), int(bboxC.ymin * iheight),\
                   int(bboxC.width * iwidth), int(bboxC.height * iheight)
            cv2.rectangle(img, bbox, (0,255,0), 2)
            print(detection.score[0])
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv2.imshow("Video", img)
    cv2.waitKey(1)