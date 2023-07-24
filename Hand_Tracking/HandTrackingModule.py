import cv2
import mediapipe as mp
import time

class handDetector():

    def __init__(self, mode=False, maxHands=2, detecConfi=0.5, trackConfi=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detecConfi = detecConfi
        self.trackConfi = trackConfi

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detecConfi, self.trackConfi)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):

        self.lmlist = []
        xList = []
        yList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (30, 240, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmlist, bbox

    def fingersUp(self):
        fingers = []
        # thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            # 4 fingers
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
def main():
    capture = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = handDetector()

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

if __name__ == "__main__":
    main()