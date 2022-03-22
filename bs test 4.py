# Python code for Background subtraction using OpenCV
from asyncio.windows_events import NULL
from cv2 import imshow
import numpy as np
import cv2
import pyautogui


cap = cv2.VideoCapture(2)
fgbg = cv2.createBackgroundSubtractorKNN(90)
res =0
i = 0
while(True):
   
   ret, frame = cap.read()
   
   i+=1
   if i > 0:
     fgmask = fgbg.apply(frame)
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


     #lower and upper range
     lower_blue = np.array([0,15,0])
     upper_blue = np.array([17,170,255])

          #masking
     mask = cv2.inRange(hsv, lower_blue, upper_blue ) 
     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
     #compining backsub mask with hsv mask
     res = cv2.bitwise_and(fgmask,src2=mask)
     #res = cv2.bitwise_and(frame, frame,mask=fgmask)
     #res = cv2.bitwise_and(res, res,mask=mask)
     #contour
     contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     contours = max(contours, key=lambda x: cv2.contourArea(x))
     cv2.drawContours(frame, [contours], -1, (255,255,0), 2)

     hull = cv2.convexHull(contours)
     cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
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
     pyautogui.FAILSAFE = False
     #pyautogui.moveTo(x_*3, y_*2.25)
     #
     cv2.imshow('hsv mask', mask)
     cv2.imshow('frame',frame )
     cv2.imshow("res",res)


     if cv2.waitKey(1)==ord('q'):
               break
	

cap.release()
cv2.destroyAllWindows()
