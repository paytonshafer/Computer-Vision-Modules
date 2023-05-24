import cv2
import time
from pose_estimate_module import PoseEstimator

pTime = 0
cTime = 0

cap = cv2.VideoCapture('PoseVideos/2.mp4') #set the camera
detector = PoseEstimator()

while 1: #infiite loop
    success, img = cap.read() #capture a frame

    if not success: #if the video is done then break
        break

    img = cv2.flip(img,1) #this line makes the camera act as a mirror

    img = detector.getPose(img) #get the pose

    lmList = detector.getPos(img, draw=False) #get the positions
    if len(lmList) != 0:
        #draw blue circle on right elblow
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (255, 0, 0), cv2.FILLED)

    #getting fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #show fps on screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img) #show the frame
    cv2.waitKey(1) #wait 1 ms