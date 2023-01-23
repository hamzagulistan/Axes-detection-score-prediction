from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout,QPushButton,QSlider,QCheckBox,QComboBox)
import cv2
import numpy as np
import threading
import time


path = 'DatasetLight08-02-2022/Long3.mp4'
camera_index=0
#path= camerapositions[0][1]
vid = cv2.VideoCapture(path)
ret,src = vid.read()
light_background = False
illumination=43
flag=True
lock=False
tt=[]
image_buffer=cv2.imread('Last_Frame_0.jpg')



#def RTSP_streamer(path):
def RTSP_streamer(path):

   global flag,lock
   lock=True
   vcap=cv2.VideoCapture(path)
   if(vcap.isOpened()==False):
        print('Error opening file')
   else:
       #print('lock')
       while(flag):
         #print('Reading')
         global image_buffer
         #ret,image_buffer=vcap.read()
   lock=False
   print("thread ended!")


tt=threading.Thread(target=RTSP_streamer,args=(path,),name="RTSP_streamer")
tt.daemon=True
tt.start()


def getSegmentedImage(img,illumination):
    #Gaussian Blur for removing Noise
    img = cv2.GaussianBlur(img, (3,3), 0)
    #RGB to Gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #illumination normalization
    gray = cv2.equalizeHist(gray)
    
    ##Segmentation
    rett,thresh = np.array(cv2.threshold(gray, illumination, 255, 0))
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    if(light_background):
        thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 4)
    else:
        thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 4)
    result=thresh.astype('uint8')
    #shrinking gray input
    if(light_background):
        result=cv2.dilate(result,np.ones((9,9),np.uint8))
    else:
        result=cv2.erode(result,np.ones((9,9),np.uint8))
    im2, contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area=0
    sec_max=[]
    plate_cnt=[]
    mask = np.zeros_like(img)
    for cnt in contours:
        #Select the contour with a big area
        #print('length ',cv2.arcLength(cnt,True))
        if ((cv2.contourArea(cnt)>max_area)|(cv2.arcLength(cnt,True)>1000)):
            #print('selected length ',cv2.arcLength(cnt,True))
            max_area=cv2.contourArea(cnt)
            #print('area',cv2.contourArea(cnt))
            plate_cnt=cnt
            cv2.drawContours(mask, [plate_cnt],0, [255,255,255], -1)
    #Shrink mask
    if(light_background):
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
    else:
        mask=cv2.erode(mask,np.ones((5,5),np.uint8))
    #crop the target
    crop = np.zeros_like(img)
    if(light_background):
        crop[mask == 0] = img[mask == 0]
    else:
        crop[mask == 255] = img[mask == 255]
    return crop




# Initialize application
app = QApplication([])

#Create ComboBox
drop=QComboBox()
def setPath(cam_no):
    global camera_index
    global path
    global vid
    global tt,flag
    flag=False #stop prev thread
    camera_index=cam_no
    path=drop.currentText()
    #print(path)
    vid.release()
    vid = cv2.VideoCapture(path)
    flag=False
    print('Waiting fot thread to end')
    time.sleep(0.1)
    #while(lock):
    #    pass
    flag=True
    tt=threading.Thread(target=RTSP_streamer,args=(path,),name="RTSP_streamer")
    tt.daemon=True
    tt.start()
#for i in range(0,len(camerapositions)):
#    drop.addItem(camerapositions[i][1])
drop.currentIndexChanged.connect(setPath)

# Create label
label = QLabel()
label.setPixmap(QPixmap('Last_Frame_0.jpg'))
def say_hello(event):
    label.setText('Done!')
def updateAll(event):
    label.setText('All updated!')
    
#Create checkbox
check=QCheckBox()
def updateFlag():
    global light_background
    if(check.isChecked()):
        light_background=True
    else:
        light_background=False
check.stateChanged.connect(updateFlag)

# Create buttons
button = QPushButton('Confirm')
button.clicked.connect(say_hello)

button_all = QPushButton('Update All')
button_all.clicked.connect(updateAll)

def printloc(QMouseEvent):
    global light_background,orign,x1,x2,y2
    x=QMouseEvent.pos().x()
    y=QMouseEvent.pos().y()  #offset 30 for correction in clicks
    x_scale=int(x)
    y_scale=int(y)
    blr_src = cv2.GaussianBlur(image_buffer.copy(), (3,3), 0)
    '''lab = cv2.cvtColor(blr_src, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    norm_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)'''

    # Switch image from BGR colorspace to HSV
    frame=blr_src
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print('x',x_scale,' y',y_scale)
    print('image_buffer:',hsv[y,x,:])

label.mousePressEvent=printloc

# Create slider
slider = QSlider(QtCore.Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(255)
def changeThreshold(event):
    global illumination
    global image_buffer
    illumination = slider.value()
    #ret,src = vid.read()
    src=cv2.resize(image_buffer,(705,500))
    src = getSegmentedImage(src,slider.value())

    img = QImage(src.data,src.shape[1],src.shape[0], src.strides[0], QImage.Format_RGB888)
    img = img.rgbSwapped()
    label.setPixmap(QPixmap.fromImage(img))
    label.setAlignment(QtCore.Qt.AlignCenter)
    
slider.valueChanged.connect(changeThreshold)

# Create layout and add widgets
layout = QVBoxLayout()
layout.addWidget(drop)
layout.addWidget(label)
layout.addWidget(slider)
layout.addWidget(check)
layout.addWidget(button)
layout.addWidget(button_all)

# Apply layout to widget
widget = QWidget()
widget.setLayout(layout)
widget.show()
app.exec_()
