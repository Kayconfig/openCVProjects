#import important things
import cv2
import requests

def download_model(url, filename):
    #download the pre-trained data from github
    resp = requests.get(url)
    model_xml = resp.text
    with open(filename, "w+") as f:
        f.write(model_xml)



#download smile detector model
download_model( "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml", "haarcascade_smile.xml" )
#download face detector model
download_model("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", "haarcascade_frontalface_default.xml" )

#load pre-trained data on face frontals 
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml') 

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

        if not successful_frame_read:
            break
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
                #get the sub frame 
                the_face = img[y:y+h, x:x+w]
                #change to grayscale
                face_grayscale = cv2.cvtColor( the_face, cv2.COLOR_BGR2GRAY)
                #get the smile
                smiles = smile_detector.detectMultiScale( face_grayscale, scaleFactor = 1.7, minNeighbors = 20) 
                #draw a rectangle around smile
                # for ( x_, y_, w_, h_) in smiles:
                #     color = (0, 255, 0)
                #     cv2.rectangle( the_face, (x_,y_), (x_+w_, y_+h_), color, thickness)
                #if someone is smiling
                if len(smiles) >= 1:
                    cv2.putText( img, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color= ( 255, 255, 255))
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
    cv2.destroyAllWindows()


print("Code Completed.")