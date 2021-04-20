#import important things
import cv2
import os
import requests


#download the pre-trained data from github
resp = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
model_xml = resp.text
with open("haarcascade_frontalface_default.xml", "w+") as f:
  f.write(model_xml)

#load pre-trained data on face frontals 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 


#choose an image to detect faces in
img = cv2.imread('aj.jpg')



#make sure you convert the image to grayscale
grayscaled_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

#detect the faces
face_coordinates = trained_face_data.detectMultiScale( grayscaled_img)
#define the parameters for the rectangle
thickness = 2 # the thickness of the rectangle around the face
color = (0, 255, 0)#dont forget BGR, in our case I'm using green
if len( face_coordinates ) > 0:
    for x,y,w,h in face_coordinates:
        #superimpose a rectangle on the image
        cv2.rectangle( img, (x,y), (x+w, y+h),  (0, 255, 0), thickness )
else:
    print("No faces detected")
#show img
cv2.imshow("Kayode's Face Detector", img)
cv2.waitKey()


print("Code Completed.")