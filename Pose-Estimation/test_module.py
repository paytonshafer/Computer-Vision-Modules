import cv2
import time
from pose_estimate_module import PoseEstimator

pTime = 0
cTime = 0

cap = cv2.VideoCapture('PoseVideos/2.mp4') #set the camera
detector = PoseEstimator()

while 1: #infiite loop
    success, img = cap.read() #capture a frame

    if not success:
        break

    img = cv2.flip(img,1) #this line makes the camera act as a mirror

    img = detector.getPose(img)

    lmList = detector.getPos(img, draw=False)
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