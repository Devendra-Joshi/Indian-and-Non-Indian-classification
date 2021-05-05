import os
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize

def crop_images(img):
    
    output = 1
    label = 0
    values = [] 
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the cascade(create your own path as  in every pc its different)
    face_cascade = cv2.CascadeClassifier(r'C:\Users\thete\Desktop\ML-major project\haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, minNeighbors = 15, minSize=(100, 100))
    
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:

        faces = img[y:y + h, x:x + w]
        
        if len(faces)>=1:
            output = 0
            new_path = 'cropped/'+str(label)+'.jpg'
            cv2.imwrite(new_path, faces)
            values.append([x,y,w,h])

    return output,values


def addon(img):

    flat_data = []
    model = pickle.load(open('img_model.p','rb'))
    
    img_resized = resize(img,(40,40,3))
    flat_data.append(img_resized.flatten())

    flat_data = np.array(flat_data)
    targets = model.predict(flat_data)
    return targets



def skin_tone():

    img_path = 'cropped/0.jpg'
    img = cv2.imread(img_path)

    skin_data = img.mean()

    if skin_data< 94:
        tone = 1

    elif skin_data>94 and skin_data< 112:
        tone = 2

    elif skin_data>112:
        tone =3
    
    return tone

def write_img(query1,query2,img,faces):

    for (x, y, w, h) in faces:
  
        if query1==0:
            cv2.putText(img,'Indian',(x+W+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            if query2==1:
                cv2.putText(img,'Dark skin tone',(x+w+6,y+h-36), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) 
            elif query2==2:
                cv2.putText(img,'Mild skin tone',(x+w+6,y+h-36), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) 
            elif query3==3:
                cv2.putText(img,'Fair skin tone',(x+w+6,y+h-36), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)  
        elif query1==1:
            cv2.putText(img,'Non Indian',(x+w+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)  

    cv2.imshow('Webcam', img)     

def main():

    values=[]
    vid = cv2.VideoCapture(0)

    while(True):

        ret, frame = vid.read()
        target = 0
        tone = 0
        output,values = crop_images(frame)
            
        if output == 0:

            target = addon(frame)            

            if target == 0:
                tone = skin_tone()
        write_img(target,tone,frame,values)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
   main()
