import cv2
import time
from face_mesh_module import FaceMesh

#time set up for fps
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0) #set the camera
detector = FaceMesh() #create the face detector, can add params

while 1: #infiite loop
    success, img = cap.read() #capture a frame
    img = cv2.flip(img,1) #this line makes the camera act as a mirror

    img = detector.getFaces(img, draw=False) #function to get hands from an img

    #generate the list of landmarks
    lmList = detector.getPos(img, draw=True)
    if len(lmList) != 0:
        print(lmList[4]) #get pos of tip of thumb

    #getting fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #show fps on screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img) #show the frame
    cv2.waitKey(1) #wait 1 ms