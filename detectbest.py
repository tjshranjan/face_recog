import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('haarcascade.xml')
cap=cv2.VideoCapture(0)
face_data=[]
data_setpath='./'
file_name=input('enter the name : ')
skip=0
while True:
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    # faces=sorted(faces,key=lambda f:f[2]*f[3])
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        skip+=1
        offset=10
        face_section=frame[y-offset:y+offset+h,x-offset:x+w+offset]
        cv2.imshow('framesss',face_section)
        cv2.imshow("face_section",face_section)
        if skip%10==0:
            face_section=cv2.resize(face_section,(100,100))
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('video frame',frame)
    

    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
face_data=np.array(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)
np.save(data_setpath+file_name+'.npy',face_data)

cap.release()
cv2.destroyAllWindows()