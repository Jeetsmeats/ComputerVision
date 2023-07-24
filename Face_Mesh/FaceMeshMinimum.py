import cv2
import mediapipe as mp
import time

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFace = mp.solutions.face_mesh
face_mesh = mpFace.FaceMesh(max_num_faces=3)
draw_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=2,
                               color=(0,255,255))

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLm, mpFace.FACEMESH_FACE_OVAL,
                                  draw_spec, draw_spec)
            for id, lm in enumerate(faceLm.landmark):
                iheight, iwidth, ichannel = img.shape
                x, y = int(lm.x * iwidth), int(lm.y * iheight)
                print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 3)
    cv2.imshow("Video", img)
    cv2.waitKey(1)