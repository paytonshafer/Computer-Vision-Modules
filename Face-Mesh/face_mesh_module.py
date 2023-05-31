import cv2
import mediapipe as mp

class FaceMesh:
    def __init__(self, mode=False, maxFaces=2, redefineLM=False,detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.redefineLM = redefineLM
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpMesh = mp.solutions.face_mesh #have to do always
        self.mesh = self.mpMesh.FaceMesh(self.mode, self.maxFaces, self.redefineLM, self.detectionConf, self.trackConf) #no paramers bc we use default
        self.mpDraw = mp.solutions.drawing_utils 
    
    #The below function gets the hands and if draw it will draw the landmarks
    def getFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #have to take the img and turn it to RGB
        self.results = self.mesh.process(imgRGB) #find the hands(if there are any)
        
        if self.results.multi_face_landmarks: #if there are hands
            if draw: 
                for face in self.results.multi_face_landmarks: #for each hand in the frame
                        
                    #mpDraw.draw_landmarks(img, face) #draws only the landmarks
                    self.mpDraw.draw_landmarks(img, face, self.mpMesh.FACEMESH_CONTOURS) #draws landmarks and connector lines
                    #add own specs to drawing first one is for landmarks second is for connections
                    #mpDraw.draw_landmarks(img, face, mpMesh.FACEMESH_CONTOURS, drawSpec, drawSpec) #draws landmarks and connector lines

        return img
    
    #this creates a list of all landmark positions on the hand
    def getPos(self, img, faceNum=0, draw=True):
        lmList = [] #create the empty list

        if self.results.multi_face_landmarks: #if there are faces
            face = self.results.multi_face_landmarks[faceNum]
            
            #for each lm on the hand
            for id, lm in enumerate(face.landmark):
                #print(id, lm) #gives id and positions
                h, w, c = img.shape #get size of img
                cx, cy = int(lm.x*w), int(lm.y*h) #turn x and y into pixel values
                #print(id, cx, cy) #print id and x and y values in pixels
                lmList.append([id, cx, cy]) #add positions to list with lm id
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

        #return the list of landmarks
        return lmList
