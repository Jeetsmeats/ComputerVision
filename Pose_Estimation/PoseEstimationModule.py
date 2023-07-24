import cv2
import mediapipe as mp
import math

class PoseEstimation(object):

    def __init__(self, imgMode=False, detecConfi=0.5, trackConfi=0.5, smlms=True, smsmeg=True,
                 complexity=1, segment=False):

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(imgMode, complexity, smlms, segment, smsmeg,
                                     detecConfi, trackConfi)
        self.mpDraw = mp.solutions.drawing_utils

    def findPositions(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):

                height, width, channel = img.shape

                cx, cy = int( width * lm.x ), int( height * lm.y )
                self.lmlist.append((id, cx, cy))
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return self.lmlist
    def applyPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.pose.process(imgRGB)
        if draw:
            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def findAngle(self, img, p1, p2, p3, draw=True):
        # get the landmarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
        # calculate the angle
        angle = math.degrees( math.atan2 (y1 - y2, x1 - x2) -
                              math.atan2 (y3 - y2, x3 - x2)
                               )
        if angle < 0:
            angle += 360
        # draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 20, (255, 0, 0), 3)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 20, (255, 0, 0), 3)
            cv2.circle(img, (x3, x3), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, x3), 20, (255, 0, 0), 3)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2)
        return angle