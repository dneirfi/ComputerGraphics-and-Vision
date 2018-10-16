import cv2
import numpy as np

#bbox=cv2.selectROI(frame,False,True)

def featureMatching(bbox_ori,frame):
    MIN_MATCH_COUNT = 10
    img1=cv2.cvtColor(bbox_ori,cv2.COLOR_BGR2GRAY)
    img2=frame
    surf = cv2.xfeatures2d.SURF_create(300)
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    #///////////////////////////
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h = img1.shape[0]
        w= img1.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
    cv2.imshow('matches',img3)


fps=0

cap=cv2.VideoCapture(0)
#cap=cv2.VideoCapture('C:\\Users\\dneir\\Desktop\\KAU\\ComputerVision\\test.avi')

status_tracker,frame=cap.read()
bbox = (287, 23, 86, 320)
bbox = cv2.selectROI(frame, False)
bbox_img=frame[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]


 #값이 작아지면 검출하는 키포인트개수 많아짐

tracker = cv2.TrackerKCF_create()
status_tracker=tracker.init(frame,bbox)

while(True):
   status_tracker,frame=cap.read()
   if not status_tracker:
       break
   
   timer=cv2.getTickCount()
   status_tracker, bbox=tracker.update(frame)
   timer=cv2.getTickCount()-timer
   fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)
   

   if status_tracker:
       x,y,w,h=[int(i) for i in bbox]
       cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
       
       
   else:
       cv2.putText(frame,"Tracking failure detected",(100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
       
       
       #값이 작아지면 검출하는 키포인트개수 많아짐
       
       featureMatching(bbox_img,frame)
       
   
       
   cv2.imshow("MedianFlow tracker",frame)
   cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)     
   cv2.imshow('bbox',bbox_img)
   if cv2.waitKey(1)&0xFF == ord('q'):
       break
        
        
cap.release()        
cv2.destroyAllWindows()
