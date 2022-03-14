import cv2
from cv2 import imshow
import numpy as np
import time

def centroid(image, final,keyb):
    moment = cv2. moments(image)
    x_ = int(moment['m10']/moment['m00'])
    y_ = int(moment['m01']*0.9/moment['m00'])
    cv2.circle(final, (x_, y_), 10, (0,0,0), -1)
    cv2.circle(keyb, (x_, y_), 10, (0,0,0), -1)
    return [x_, y_]

def keys():   
        keyboard = cv2.imread("new_kb.png",-1)
        
        cv2.rectangle(keyboard,(100,100),(225,300),(0,200,0), 20, cv2.FONT_HERSHEY_COMPLEX)
        cv2.putText(keyboard, "hi", (110,200),cv2.FONT_HERSHEY_COMPLEX, 2, (100, 255, 0),2)
        #cv2.imshow("keyboard", keyboard)
        
        return keyboard
        #cv2.waitKey(0)

def detection():
    capture = cv2.VideoCapture(0)
    while True:
        
        ret, frame = capture.read()
        frame = face_censoring(frame) #censoring face to reduce noise
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #lower and upper range
        lower_blue = np.array([0,15,0])
        upper_blue = np.array([17,170,255])

        #masking
        mask = cv2.inRange(hsv, lower_blue, upper_blue )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        #contorus and hull
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(frame, [contours], -1, (255,255,0), 2)
        hull = cv2.convexHull(contours)
        cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
        
        framekey = keys()
        #show center of contour
        m, n = centroid(hull, frame,framekey)

        #displaying
        cv2.imshow("masked",mask)
        cv2.imshow("frame", frame)
        cv2.imshow("keyboard",framekey)
        if cv2.waitKey(1)==ord('q'):
            break
def face_detection():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), -1)            
        cv2.imshow("face", frame)
        cv2.waitKey(1)
def face_censoring(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)            
    cv2.imshow("face", frame)
    censored_frame = frame

    return censored_frame
print("hello world")
detection()