import cv2
import numpy as np
import sys
import os
import datetime
import time
import threading

null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
save_fds = [os.dup(1), os.dup(2)]


#Initialize variables
delta=10
down_res=(1024,704)
points=[]
first_frame_cnt=0
hist_size=3        #History of frame points
min_pnt_hist=2     #Min frames for point to appear
point_history=[ [] for i in range(hist_size) ]
shoot_points=[]
min_dist=10
points_old=[]
#Color setting for circle lines and axe
axeMin = (0,0,0)
axeMax = (40, 40,40)
backGroundRedMin = (150,100,105)
backGroundRedMax = (185, 185, 145)
backGroundBlueMin = (90,65,90)
backGroundBlueMax = (125, 120, 135)
backGroundGreenMin = (30,60,70)
backGroundGreenMax = (70, 120, 120)

#Crop Image to focus on target area Added 08-02-22
cropXmin = 38
cropYmin = 4
cropXmax = 792
cropYmax = 715

prjectorImage = cv2.imread('Target.jpg')
image_buffer=[]
def RTSP_streamer(rtsp_local, threadID):
   vcap=cv2.VideoCapture(rtsp_local)
   if(vcap.isOpened()==False):
        print('Error opening file')
   else:
      while(True):
         global image_buffer
         ret,image_buffer=vcap.read()
   

#Read Blob Detector Parameters
f=open('Target4x4.bin')
line=f.readline()
param=dict({})
if(line=="Shoot detector config1\n"):
    print('Initializing Blob Detector')
    for line in f:
        ptr=int(line.find('='))
        key=line[:ptr]
        value=line[ptr+1:]
        if "True" in value:
            value=True
        elif "False" in value:
            value=False
        elif "." in value:
            value=float(value)
        else:
            value=int(value)
        param[key]=value
    print(param)
else:
    print('bad file!')
    #exit()



image_path_main = './OutputVideos/'

#Stdout pointers
'''null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
save_fds = [os.dup(1), os.dup(2)]'''
dt = datetime.datetime.now()

#Video read & write objects
rtsp_url = "rtsp://admin:Zcompany4@192.168.1.108:554"

vid = cv2.VideoCapture(rtsp_url)
if(vid.isOpened()==False):
    print('Error opening video')
    exit()
    
#Start thread to read RTSP stream
tt=threading.Thread(target=RTSP_streamer,args=(rtsp_url,0),name="RTSP_streamer")
tt.daemon=True
tt.start()

#Wait for RTSP thread to start
while(image_buffer==[]):
    pass

fps = int(vid.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(image_path_main+'output_'+str((dt.strftime("%H%M_%d%B%y")))+'.avi',
                      cv2.VideoWriter_fourcc('M','J','P','G'), fps*0.75, (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                   int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
n=0

points = []
first_frame=True
first_shot = True

firstFramesForCircle = 0
circleBlueFirst = None
circleRedFirst = None
circleGreenFirst = None
    
shoot_index=0

while(vid.isOpened()):
    start=time.time()
    
    src=image_buffer
    prjectorImage = cv2.imread('Target.jpg')
    
    shoot_points=[]
    img=src.copy()
    #Image Segmentation
    if(first_frame):
        print('Image shape',src.shape)
        first_frame=False

        #Get Segmented Image
        #[cX,cY]=[709,336]#Center point of target
        [cX,cY]=[414,339]#Center point of target
        
       
        
        print([cX,cY])

    #Image Processor
    blr_src = cv2.GaussianBlur(src, (3,3), 0)
    # Switch image from BGR colorspace to HSV
    frame=blr_src
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Sets pixels to white if in range, else will be set to black
    maskAxe = cv2.inRange(hsv, axeMin, axeMax)
    maskBackGroundRed = cv2.inRange(hsv, backGroundRedMin, backGroundRedMax)
    maskBackGroundBlue = cv2.inRange(hsv, backGroundBlueMin, backGroundBlueMax)
    maskBackGroundGreen = cv2.inRange(hsv, backGroundGreenMin, backGroundGreenMax)
    circlesBlue = cv2.HoughCircles(maskBackGroundBlue, cv2.HOUGH_GRADIENT, cv2.HOUGH_GRADIENT,200,param1=160,param2=240,minRadius=200,maxRadius=470)
    circlesRed = cv2.HoughCircles(maskBackGroundRed, cv2.HOUGH_GRADIENT, cv2.HOUGH_GRADIENT,200,param1=160,param2=240,minRadius=0,maxRadius=470)
    circlesGreen = cv2.HoughCircles(maskBackGroundGreen, cv2.HOUGH_GRADIENT, cv2.HOUGH_GRADIENT,45,param1=160,param2=150,minRadius=0,maxRadius=470)
    
    
    # Bitwise-AND of mask and purple only image - only used for display
    mask = maskAxe
    res = cv2.bitwise_and(frame, frame, mask= mask)
    mask = cv2.erode(mask, np.ones((1,1), np.uint8) , iterations=1)
    # commented out erode call, detection more accurate without it
    # dilate makes the in range areas larger
    tmp = cv2.dilate(mask, None, iterations=2)
    
    
    #Crop Image to focus on target area Added 08-02-22
    tmp[0:cropYmin,:]=0
    tmp[:,0:cropXmin]=0
    tmp[cropYmax:tmp.shape[0],:]=0
    tmp[:,cropXmax:tmp.shape[1]]=0
    #Pause detection if Circle not detected
    #if(circlesBlue is None):
    #    tmp[:,:]=0

    
    contours, hir = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    keypoint=[]
    for i in range(len(contours)):
       M = cv2.moments(contours[i])
       if(M['m00'] == 0.0):
         continue
       cnt_area=cv2.contourArea(contours[i])
       
       if(cnt_area>param['maxArea'])or(cnt_area<param['minArea']):
            continue
       #estimate shape of contour (circle, rectangle, square) to remove noise
       peri = cv2.arcLength(contours[i], True)
       approx = cv2.approxPolyDP(contours[i], 0.04 * peri, True)
       (x, y, w, h) = cv2.boundingRect(approx)
       ar = w / float(h)
       # a square will have an aspect ratio that is approximately
       # equal to one, otherwise, the shape is a rectangle
       shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
       if(len(approx)<=3 or ar<0.5 or ar>1.8 or cnt_area>param['maxArea'])or(cnt_area<param['minArea']): #if vertices less than 4, noise
         continue
       
       print('Vertices: ',len(approx),'ar',ar,' Area',cnt_area)
       x, y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
       
       src = cv2.drawContours(src, contours, i, (139,69,19), 3)
       cv2.circle(src, (x, y), 1, (165,42,42), -1)
       keypoint.append([x,y])
    print(len(contours))
    
    # Plot Background circles
    print("Circles detected",(circlesBlue),(circlesRed),(circlesGreen))
    if((firstFramesForCircle <=12) and (circlesBlue is not None) and (circlesRed is not None) and (circlesGreen is not None)):
        firstFramesForCircle+=1
        circleBlueFirst=circlesBlue
        circleRedFirst=circlesRed
        circleGreenFirst=circlesGreen
        print('First circle detected')
    '''circles = circleBlueFirst
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(src, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(src, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
    circles = circleRedFirst
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(src, (x, y), r, (255, 0, 0), 2)
            cv2.rectangle(src, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)
    
    circles = circleGreenFirst
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(src, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(src, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)'''
    
    
    #History filter
    if first_frame_cnt<hist_size:
        points=[]
        kp=[]
        for i in range(0,len(keypoint)):
            points.append(keypoint[i])#[ int(keypoint[i].pt[0]) , int(keypoint[i].pt[1]) ])
        point_history[first_frame_cnt]=points.copy()
        first_frame_cnt+=1
        continue
    else:
        points=[]
        kp=[]
        #extract x,y from keypoints
        for i in range(0,len(keypoint)):
            points.append(keypoint[i])#[ int(keypoint[i].pt[0]) , int(keypoint[i].pt[1]) ])
        
        #add new points to 2D history
        point_history.append(points)
        point_history.pop(0)

        #turn 2D history points to 1D history
        history_1d=[]
        history_pnts=[]
        for t in point_history:
            for pnt in t:
                history_1d.append(pnt)
                if (pnt not in history_pnts): #remove repeating points from 1D history
                    new_pnt=True
                    for i in history_pnts:
                        dist=abs(np.linalg.norm(np.array(pnt[0:2])-np.array(i[0:2])))
                        if(dist<delta):
                            new_pnt=False
                            break
                    if(new_pnt):
                        history_pnts.append(pnt)


        #Digitally clean target
        if(first_frame_cnt==hist_size):
           first_frame_cnt+=1
           points_first_frame=history_pnts.copy()

        #count points in 1D history
        history_pnts_cnt=np.zeros(len(history_pnts)) #create array for point counts
        for pnt in history_pnts:                     #Generate count for each point
            for chk_pnt in history_1d:
                dist = abs(np.linalg.norm(np.array(pnt[0:2])-np.array(chk_pnt[0:2])))
                if(dist<delta):
                    history_pnts_cnt[history_pnts.index(pnt)]+=1

        #insert selected points to final array
        live_pnts=[]
        for i in range(0,len(history_pnts_cnt)):    #Insert points in final array
            if history_pnts_cnt[i]>=min_pnt_hist:
                new_pnt=True
                if(shoot_points is None):           #Always insert first point
                    shoot_points.append(history_pnts[i])
                    break
                for k in shoot_points:
                    dist=abs(np.linalg.norm(np.array(history_pnts[i][0:2])-np.array(k[0:2])))
                    if(dist<delta):
                        new_pnt=False
                        break
                

                new_live_pnt=True
                for pnt in points_first_frame:
                   dist=abs(np.linalg.norm(np.array(history_pnts[i][0:2])-np.array(pnt[0:2])))
                   if(dist<delta):
                        new_live_pnt=False
                        break
                if(not new_live_pnt):
                   continue

                if(new_pnt):
                    shoot_points.append(history_pnts[i])
                live_pnts.append(history_pnts[i])
                cv2.circle(src,tuple(history_pnts[i]),20,[255,0,0],2)

                x=history_pnts[i][0]
                y=history_pnts[i][1]
                dist_center=np.linalg.norm(np.array([x,y])-np.array([cX,cY]))
                
                circle_num=0
                circle_score=0
                dist_center=np.linalg.norm(np.array([cX,cY])-np.array([x,y]))
                if(circleGreenFirst is not None and circleRedFirst is not None and circleBlueFirst is not None):
                    if (dist_center<circleGreenFirst[0][0][2]):
                        circle_num=1
                        circle_score=5
                    elif (dist_center<circleRedFirst[0][0][2]):
                        circle_num=2
                        circle_score=3
                    elif (dist_center<circleBlueFirst[0][0][2]):
                        circle_num=3
                        circle_score=1
                    else:
                        circle_num=4
                        circle_score=0
                    print('Distance Center',dist_center,'circleGreenFirst',circleGreenFirst[0][0][2],'circleRedFirst',circleRedFirst[0][0][2],'circleBlueFirst',circleBlueFirst[0][0][2])
                    cv2.putText(src,str('Score'+str(circle_score)),tuple(history_pnts[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.7,[0,255,255], 1)
                    cv2.putText(prjectorImage,str('Score'+str(circle_score)),(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2,[0,0,0], 1)

        
        #Point filter
        if first_frame_cnt==hist_size:
          first_frame_cnt+=1
          points_old=shoot_points.copy()
          shoot_points=[]
        else:
          for cnt in range(0,len(points_old)):
              for shoot in shoot_points:
                  dist=abs(np.linalg.norm(np.array(points_old[cnt][0:2])-np.array(shoot[0:2])))
                  if(dist<delta):
                      shoot_points.remove(shoot)
                      points_old[cnt]=shoot

        #Show output
        cv2.circle(prjectorImage,(512,384),5,[0,230,0],-1)
        #cv2.circle(src,(512,384),5,[0,230,0],-1)
        
        #cv2.imshow('Source',cv2.resize(src,down_res))
        cv2.imshow('Blr',cv2.resize(src,down_res))
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('frame',-1024,0)
        cv2.imshow('frame',prjectorImage)
        #Write output
        out.write(src)
        

        while (time.time()-start)<1:
           if(first_frame_cnt<hist_size):
              break
        print('Thread took ',time.time()-start,' secs')
    #else:
    #   points_new=shoot_points.copy()
    #   break
    #Quit if user presses 'q'
    if cv2.waitKey(30)&0xFF == ord('q'):
        points_new=shoot_points.copy()
        break

#Write last image
cv2.imwrite('Last_Frame_0.jpg',img)
#Plot detected points (Live points)
if(points_new is not None):
    for t_count in range(0,len(points_new)):
        x=points_new[t_count][0]
        y=points_new[t_count][1]
        label = str(t_count)
        #red circles
        #cv2.circle(src,(x,y),15,[0,0,255])
        if(True):
            shoot_index+=1
            dist_center=np.linalg.norm(np.array([x,y])-np.array([cX,cY]))
            angle_center = np.rad2deg(np.arctan2(y - cY, x - cX))+90
            angle_true=(angle_center-90)*-1
            if(angle_center<0):
                angle_center+=360
            elif(angle_center>360):
                angle_center-=360
            if(angle_true<0):
                angle_true+=360
            elif(angle_true>360):
                angle_true-=360
            clock_arm=int(round(angle_center/30))
            if(clock_arm==0):
                clock_arm=12

#Release memory
out.release()
vid.release()
cv2.destroyAllWindows()




#####################################################################################################################