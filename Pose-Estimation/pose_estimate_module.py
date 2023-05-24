import cv2
import mediapipe as mp
import time

class PoseEstimator:
    def __init__(self, mode=False, modelComplexity=1, upperBodyOnly=False, smoothLandmarks=True, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.upperBodyOnly = upperBodyOnly
        self.smoothLandmarks = smoothLandmarks
        self.detectionConf = detectionConf
        self.trackConf = trackConf 

        self.mpPose = mp.solutions.pose #needed to use pose
        #the below sets up the Pose estimator usiing the parms given
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, self.upperBodyOnly, self.smoothLandmarks, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils #drawing utilities
    
    #this function finds the pose given and image and draws the landmarks and connectors
    def getPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #have to take the img and turn it to RGB
        self.results = self.pose.process(imgRGB) #find the hands(if there are any)
        
        if self.results.pose_landmarks: #if there is a pose
            if draw:
                #mpDraw.draw_landmarks(img, hand) #draws only the landmarks
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS) #draw lms and connections
        return img #return the drawn on image
    
    #get positions of landmarks on the pose
    def getPos(self, img, draw=True):
        lmList = [] #empty landmark list

        if self.results.pose_landmarks: #if there is a pose
            #for each landmark on the pose
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #print(id, lm) #gives id and positions
                h, w, c = img.shape #get size of img
                cx, cy = int(lm.x*w), int(lm.y*h) #turn x and y into pixel values
                #print(id, cx, cy) #print id and x and y values in pixels
                lmList.append([id, cx, cy]) #add the pixel values with id to list
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED) #adding an extra circles

        return lmList #return list
