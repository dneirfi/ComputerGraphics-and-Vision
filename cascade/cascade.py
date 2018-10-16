# -*- coding: utf-8 -*-
import cv2
import numpy as np
#import urllib.request



font=cv2.FONT_HERSHEY_SIMPLEX
#cap1=cv2.VideoCapture('video4.avi')
cam=cv2.VideoCapture(0)
cam.set(3,720)
cam.set(4,720)

while(True):
    
    #ret1, frame1 = cap1.read()
    ret_val, img = cam.read()
    
    gray1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    face_cascade=cv2.CascadeClassifier('haarcascade_frontface.xml')
    hand_cascade=cv2.CascadeClassifier('cascade3.xml')
    
    faces1=face_cascade.detectMultiScale(gray1,1.3,5)
    
    hands1=hand_cascade.detectMultiScale(gray1,1.3,5)
    
    
    
    for (x,y,w,h) in faces1:
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
   
    for (hx,hy,hw,hh) in hands1:
         cv2.rectangle(img,(hx,hy),(hx+hw,hy+hh),(0,255,0),2)
         
    if(len(hands1)>0):
        faces1 = list(faces1)
        for i in range(len(faces1)):
            pos = i
            for j in range(i,len(faces1)):
                if(faces1[j][0]<faces1[pos][0]):
                    pos = j
            faces1[i],faces1[pos] = faces1[pos],faces1[i]
        
        for (fx,fy,fw,fh) in faces1:
            if(hands1[0][0] < fx):
                hx,hy,hw,hh = hands1[0]
                x,y = min([fx,hx]), min([fy,hy])
                w,h = max([fx+fw,hx+hw])-x, max([fy+fh,hy+hh]) - y
                
                dst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
                rect = np.array(((x,y),(x+w,y),(x+w,y+h),(x,y+h)),dtype="float32")
                perspectiveMat = cv2.getPerspectiveTransform(rect,dst)
                warpMat = cv2.warpPerspective(img, perspectiveMat, (w,h))
                warpMat = cv2.resize(warpMat,(640,480),interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Cam Viewer",warpMat)
                if cv2.waitKey(1)&0xFF == ord('q'):
                    break
                break
                
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    
    cv2.imshow('Cam Viewer',img)
    

#cap1.release()
cam.release()
cv2.destroyAllWindows()
