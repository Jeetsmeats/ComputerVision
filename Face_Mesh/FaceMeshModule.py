import cv2
import mediapipe as mp
import time
class FaceMeshDetection(object):

    def __init__(self, mode=False, max_num_faces=3, detec_conf=0.5, track_conf=0.5,
                 thickness=1, circle_radius=2, color=(0,255,255), refine_lm=True):

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFace = mp.solutions.face_mesh
        self.mpFace.FaceMesh()
        self.face_mesh = self.mpFace.FaceMesh(mode, max_num_faces, refine_lm, detec_conf, track_conf)
        self.draw_spec = self.mpDraw.DrawingSpec(thickness=thickness, circle_radius=circle_radius,
                                       color=color)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for self.faceLm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.faceLm, self.mpFace.FACEMESH_FACE_OVAL,
                                        self.draw_spec, self.draw_spec)
                faces.append(self.findLandmarks(img))
        return img, faces

    def findLandmarks(self, img):

        face = []
        for id, lm in enumerate(self.faceLm.landmark):
            iheight, iwidth, ichannel = img.shape
            x, y = int(lm.x * iwidth), int(lm.y * iheight)
            face.append([x, y])
            return face

def main():
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture('Face Meshing Videos/Multiple Faces.mp4')
    pTime = 0
    detector = FaceMeshDetection()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 3)
        cv2.imshow("Video", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()