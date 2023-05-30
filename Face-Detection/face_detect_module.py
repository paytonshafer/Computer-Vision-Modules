import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, minDetConf=0.5):
        self.minDetConf = minDetConf

        self.mpFace = mp.solutions.face_detection #have to do always
        self.face = self.mpFace.FaceDetection(self.minDetConf) #params from 
        self.mpDraw = mp.solutions.drawing_utils #drawing utilities
    
    #this function finds the pose given and image and draws the landmarks and connectors
    def getFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #have to take the img and turn it to RGB
        self.results = self.face.process(imgRGB) #find the hands(if there are any)

        if self.results.detections: #if there is a pose
            if draw:
                for detection in self.results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = img.shape #get size of img
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    cv2.rectangle(img, bbox, (255, 0, 255), 2) #draw using bouding box points
                    #mpDraw.draw_detection(img, detection) #draws face using utility
                    cv2.putText(img, f'{int (detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2) #add score
        return img, bbox #return the drawn on image
    
