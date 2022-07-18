import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
#  for prediction , KNN code -------------------------------
def dis(x,y):
    return np.sqrt(sum((x-y)**2))

def knn(train,test,k=15):
    dist=[]
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=dis(test,ix)
        dist.append([d,iy])
    dk = sorted(dist, key=lambda d:d[0])[:k]
    labels=np.array(dk)[:,-1]
    output=np.unique(labels,return_counts=True)
    index=output[1].argmax()
    return output[0][index]
# ---------------------------------------------------------

face_cascade=cv2.CascadeClassifier('haarcascade.xml')
cap=cv2.VideoCapture(0)
face_data=[]
labels =[]
class_id=0
names={} #mapping of names to id
data_setpath='./'
#---------------------

for fx in os.listdir(data_setpath):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        data_item=np.load(data_setpath+fx)
        face_data.append(data_item)
        target=class_id*np.ones((data_item.shape[0],1))
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0)
# print(face_dataset.shape)
# print(face_labels.shape)
trainset=np.concatenate((face_dataset,face_labels),axis=1)
# print(trainset.shape)


# testing by video stream---------------

while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for (x,y,w,h) in faces:
        offset=10
        face_section=frame[y-offset:y+offset+h,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        face_section=face_section.flatten()
        face_section=np.array(face_section)
        out = knn(trainset,face_section)
        pred_name=names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("face_detector",frame)
    if cv2.waitKey(1)   &  0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()