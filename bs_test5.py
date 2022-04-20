import cv2
import sys
from cv2 import imshow
import numpy as np
import dlib
import pyautogui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

cap = cv2.VideoCapture(2)

fgbg = cv2.createBackgroundSubtractorKNN(250) #500
detector = dlib.get_frontal_face_detector()
res = 0
f = 0
def face_censoring(frame):
    faces = detector(frame)
    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(250,0,0),2)
def bs(frame):
    fgmask = fgbg.apply(frame)
    return fgmask
def hsv(frame):
    f=0
    fgmask = fgbg.apply(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)               
    #lower and upper range
    lower_blue = np.array([0,15,0])
    upper_blue = np.array([17,170,255])
    #masking
    mask = cv2.inRange(hsv, lower_blue, upper_blue ) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    res = cv2.bitwise_and(fgmask,src2=mask)

    if np.sum(mask) == 0:
        print()

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(frame, [contours], -1, (255,255,0), 2)
    hull = cv2.convexHull(contours)
    (x,y,w,h) = cv2.boundingRect(contours)
    rec_frame = frame
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
    #draw rectangle instead of convexhull
    cv2.rectangle(rec_frame,(x,y),(x+w,y+h),(0,0,0),2)
     #
    moment = cv2.moments(hull)
     #fixing no contour bug
    m00 = moment['m00']
    if moment['m00'] == 0:
          print("no contours")
          m00 = 1
    x_ = int(moment['m10']/m00)
    y_ = int(moment['m01']*0.9/m00)
    cv2.circle(frame, (x_, y_), 10, (0,0,0), -1)
     #fixing pyautogui mouse to corner bug
    #pyautogui.FAILSAFE = False
     #pyautogui.moveTo(x_*3, y_*2.25)
     #roi save for test
    cv2.rectangle(frame,(400,100),(600,400),color=(0,0,0),thickness=2)
    #ratio to calculate fist
    (a,b) = np.shape(res[y:y+h,x:x+w])
    if np.sum(res[y:y+h,x:x+w])/(a*b*255) >0.6 and np.sum(res[y:y+h,x:x+w])/(a*b*255) < 0.75 and f >= 30:
        f = 0
        print("fist"+str(np.sum(res[y:y+h,x:x+w])/(a*b*255)))
        #pyautogui.click()
        
    else:
        print("hand"+str(np.sum(res[y:y+h,x:x+w])/(a*b*255)))
        f+=1
    return frame

def window():
   app = QApplication(sys.argv)
   widget = QWidget()
   
   button1 = QPushButton(widget)
   button1.setText("Button1")
   button1.setGeometry(0,0,100,100)
   button1.move(400,400)
   button1.clicked.connect(button1_clicked)

   button2 = QPushButton(widget)
   button2.setText("Button2")
   button2.setGeometry(0,0,100,100)

   button2.move(64,400)
   button2.clicked.connect(button2_clicked)

   widget.setGeometry(50,50,640,480)
   widget.setWindowTitle("PyQt5 Button Click Example")
   widget.show()
   sys.exit(app.exec_())


def button1_clicked():
   print("Button 1 clicked")

def button2_clicked():
   print("Button 2 clicked") 

app = QApplication(sys.argv)
widget = QWidget()

while (True):
   
    button1 = QPushButton(widget)
    button1.setText("Button1")
    button1.setGeometry(0,0,100,100)
    button1.move(400,200)
    button1.clicked.connect(button1_clicked)

    button2 = QPushButton(widget)
    button2.setText("Button2")
    button2.setGeometry(0,0,100,100)

    button2.move(64,64)
    button2.clicked.connect(button2_clicked)

    widget.setGeometry(50,50,640,480)
    widget.setWindowTitle("PyQt5 Button Click Example")
    widget.show()
    _,frame = cap.read()
    _,user_frame = cap.read()
    fgmask = fgbg.apply(frame)
    faces = detector(frame)
    for face in faces:
        
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(250,0,0),-1)
    #cv2.imshow("",frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)               
    #lower and upper range
    lower_blue = np.array([0,15,0])
    upper_blue = np.array([17,170,255])
    #masking
    mask = cv2.inRange(hsv, lower_blue, upper_blue ) 
    cv2.namedWindow("hsv")
    cv2.moveWindow("hsv",1300,480)
    cv2.imshow("hsv", fgmask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    res = cv2.bitwise_and(fgmask,src2=mask)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    __, res = cv2.threshold(res, 0,255, cv2.THRESH_BINARY)

    if np.sum(res) == 0:
        continue

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(frame, [contours], -1, (255,255,0), 2)
    hull = cv2.convexHull(contours)
    (x,y,w,h) = cv2.boundingRect(contours)
    rec_frame = frame
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
    cv2.drawContours(user_frame, [hull], -1, (0, 255, 255), 2)
    #draw rectangle instead of convexhull
    cv2.rectangle(rec_frame,(x,y),(x+w,y+h),(0,0,0),2)
    cv2.rectangle(user_frame,(x,y),(x+w,y+h),(0,0,0),2)
     #
    moment = cv2.moments(hull)
     #fixing no contour bug
    m00 = moment['m00']
    if moment['m00'] == 0:
          print("no contours")
          m00 = 1
    x_ = int(moment['m10']/m00)
    y_ = int(moment['m01']*0.9/m00)
    pyautogui.FAILSAFE = False

    pyautogui.moveTo(x_+50, y_+50)
    cv2.circle(frame, (x_, y_), 10, (0,0,0), -1)
    cv2.circle(user_frame, (x_, y_), 10, (0,0,0), -1)
     #fixing pyautogui mouse to corner bug
    #pyautogui.FAILSAFE = False
     #pyautogui.moveTo(x_*3, y_*2.25)
     #roi save for test
    #ratio to calculate fist
    (a,b) = np.shape(res[y:y+h,x:x+w])
    if np.sum(res[y:y+h,x:x+w])/(a*b*255) >0.6 and np.sum(res[y:y+h,x:x+w])/(a*b*255) < 0.75 and f >= 30 and a*b >1000:
        f = 0
        print("fist"+str(a*b))
        pyautogui.click()
        
    else:
        print(str(np.sum(res[y:y+h,x:x+w])/(a*b*255)))
        f+=1

    #cv2.imshow("frame",frame)
    cv2.namedWindow("user frame")
    cv2.moveWindow("user frame",1300,0)
    cv2.imshow("user frame",user_frame)
    cv2.imshow("binary", res)


    if cv2.waitKey(1) == ord("q"):
        break

print("hello")

