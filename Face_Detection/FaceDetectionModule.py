import cv2
import mediapipe as mp
import time
class FaceDetection(object):

    def __init__(self, minDetectionCon=0.5, minTrackCon=0.5):

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon, minTrackCon)

    def findFace(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                iheight, iwidth, ichannel = img.shape
                bbox = int(bboxC.xmin * iwidth), int(bboxC.ymin * iheight), \
                       int(bboxC.width * iwidth), int(bboxC.height * iheight)
                bboxs.append([bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l = 30, thick=5, rthick=1):
        x, y, width, height = bbox
        x1, y1 = (x+width), (y+height)
        # top left
        cv2.rectangle(img, bbox, (0, 255, 0), rthick)
        cv2.line(img, (x, y), (x+l,y), (0, 255, 255), thick)
        cv2.line(img, (x, y), (x,y+l), (0, 255, 255), thick)
        # top right
        cv2.line(img, (x1, y), (x1-l, y), (0, 255, 255), thick)
        cv2.line(img, (x1, y), (x1, y+l), (0, 255, 255), thick)
        # bottom left
        cv2.line(img, (x, y1), (x+l, y1), (0, 255, 255), thick)
        cv2.line(img, (x, y1), (x, y1-l), (0, 255, 255), thick)
        # bottom right
        cv2.line(img, (x1, y1), (x1-l, y1), (0, 255, 255), thick)
        cv2.line(img, (x1, y1), (x1, y1-l), (0, 255, 255), thick)

        return img

def main(file):

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(file)
    pTime = 0
    detector = FaceDetection()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFace(img, 1)
        print(bboxs)
        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime
        cv2.putText(img, f'FPS {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    file = 'FaceVideos/Multiple Faces.mp4'
    main(file)