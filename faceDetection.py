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

if input(" Do you want to use your webcam? [y / n ] : ").lower() == "y":
    vidUrl = 0
else:
    vidUrl = input("Enter the absolute path of the video: ")
#choose an image to detect faces in
#img = cv2.imread(imageURI) i comment it out to switch from single image to video
webcam = cv2.VideoCapture(vidUrl) # if you use zero, it will use your webcam. or you can use the file name.

try: 

    #Loop until we close the webcam or press q
    while True:
        print("press q to quit")
        #read current frame
        successful_frame_read, frame = webcam.read()
        img = frame
        #make sure you convert the image to grayscale
        grayscaled_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

        #detect the faces
        face_coordinates = trained_face_data.detectMultiScale( grayscaled_img)
        #define the parameters for the rectangle
        thickness = 2 # the thickness of the rectangle around the face

        if len( face_coordinates ) > 0:
            for x,y,w,h in face_coordinates:
                color = (0,0,255 )#dont forget BGR is the color format, in our case I'm using red
                #superimpose a rectangle on the image
                cv2.rectangle( img, (x,y), (x+w, y+h),  color, thickness )
        else:
            print("No faces detected")
        #show img
        cv2.imshow("Kayode's Face Detector", img)
        key = cv2.waitKey(1) #wait key will wait for 1 milliseconds before going to the next frame
        if key==81 or key == 113: #capital or small q is pressed
            break
except Exception as e:
    print("Error occurred: ",e)
finally:
    #cleanup
    webcam.release()


print("Code Completed.")