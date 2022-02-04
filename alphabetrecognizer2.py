import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#loading images and files
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

#making an array of alphabets
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#training the algorithm
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=26,train_size=7500,test_size=2500)

#scaling the features 
X_train_scaled = X_train/255.0 
X_test_scaled = X_test/255.0

#classifier
clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,y_train)
y_pred = clf.predict(X_test_scaled)

#accuracy
accuracy = accuracy_score(y_test,y_pred)
 #print('Accuracy: ',accuracy)

#opening the camera
cap = cv2.VideoCapture(0)

#running a loop
while(True):
    #capturing frame 
    try:
        ret,frame = cap.read()

        #operating on frames
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #drawing a box in the center
        height,width = gray.shape
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

        #region of intrest
        roi = gray[ upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        #converting to imagge pil
        img_pil = Image.fromarray(roi)

        #converting to gray scale
        img_bw = img_pil.convert('L')
        img_bw_resize = img_bw.resize((28,28),Image.ANTIALIAS)

        #inverting the image
        img_bw_resize_invert = PIL.ImageOps.invert(img_bw_resize)
        pixel_filter = 20

        #changing the scaler quality
        min_pixel = np.percentile(img_bw_resize_invert,pixel_filter)

        #limiting the values between 0,255
        img_bw_resize_invert_scaled = np.clip(img_bw_resize_invert-min_pixelm,0,255)
        max_pixel = np.max(img_bw_resize_invert)

        #converting to an array
        img_bw_resize_invert_scaled = np.asarray(img_bw_resize_invert_scaled)/max_pixel

        #reshaping and pridicting
        test_sample = np.array(img_bw_resize_invert_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        
        #printing the result
        print("Predicted class: ",test_pred)

        #showng the screen
        cv2.imshow('frame',gray)

        #to come out of while loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    except Exception as e:
      pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()