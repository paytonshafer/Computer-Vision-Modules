import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1,detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf 

        self.mpHands = mp.solutions.hands #have to do always
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConf, self.trackConf) #no paramers bc we use default
        self.mpDraw = mp.solutions.drawing_utils
    
    def getHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #have to take the img and turn it to RGB
        self.results = self.hands.process(imgRGB) #find the hands(if there are any)
        
        if self.results.multi_hand_landmarks: #if there are hands
            for hand in self.results.multi_hand_landmarks: #for each hand in the frame
                if draw:
                    #mpDraw.draw_landmarks(img, hand) #draws only the landmarks
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS) #draws landmarks and connector lines
        return img
    
    def getPos(self, img, handNum=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks: #if there are hands
            myHand = self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(myHand.landmark):
                #print(id, lm) #gives id and positions
                h, w, c = img.shape #get size of img
                cx, cy = int(lm.x*w), int(lm.y*h) #turn x and y into pixel values
                #print(id, cx, cy) #print id and x and y values in pixels
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED) #adding an extra circle for id 0

        return lmList
