import cv2
import numpy as np
import tensorflow as tf
#image1=cv2.imread("aa.png",1)
#lower and upper range
lower_blue = np.array([0,15,0])
upper_blue = np.array([17,170,255])

image2 = cv2.imread("data/hand-stress-anger-fear.jpg",1)

image3 = cv2.imread("data/open-hand.jpg",1)
image4 = cv2.imread('data/three_hand_sign.jpg',1)
#image1 = cv2.resize(image1,(128,128))
image2 = cv2.resize(image2,(128,128))

image3 = cv2.resize(image3,(128,128))

hsv = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)
mask3 = cv2.inRange(hsv, lower_blue, upper_blue )
image3 = cv2.bitwise_and(image3,image3, mask=mask3)

image4 = cv2.resize(image4,(128,128))

hsv = cv2.cvtColor(image4, cv2.COLOR_BGR2HSV)
mask4 = cv2.inRange(hsv, lower_blue, upper_blue )
image4 = cv2.bitwise_and(image4,image4, mask=mask4)

imgaes= [image2,image3,image4]
imgaes = np.array(imgaes)
resnet_model = tf.keras.models.load_model("resnet_50")

predicting_new = resnet_model.predict(imgaes)
predicting_new = np.argmax(np.round(predicting_new),axis=1)
predicting_new[0]
i = 0

for x in predicting_new:
 # itemindex = np.where(x==np.argmax(y_test,axis = 1))
  #print(old_ytest_list[itemindex][0])
  #cv2.imshow(imgaes[i])
  i+=1
#itemindex = np.where(predicting_new[0]==np.argmax(y_test,axis = 1))
print(predicting_new)
