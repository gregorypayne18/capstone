import cv2
import Jetson.GPIO as GPIO

#img = cv2.imread('lena.png')
cap = cv2.VideoCapture(1)
#cap.set(3, 640) #640
#cap.set(4, 480) #480

#read names from file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#connect to neural net
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)

#establish parameters
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def zoom(sign, read):
    #scale change
    scale = 1000
    if sign == "+":
        scale = scale - 50
    if sign == "-":
        scale = scale + 50

    #get the webcam dimensions
    height, width, channels = read.shape

    #crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(scale*height/100),int(scale*width/100)
    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = read[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))

    return resized_cropped

def adjust_motor(x, y):
    pass

while True:
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, 0.5)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            text = float(classNames[classId-1] + " " + confidence) * 100
            #change color
            if (classNames[classId-1]) == "plate":
                color = red
            else:
                color = green
            #adjust to find object
            if (classNames[classId-1]) == "plate":
                zoom("+", img)
                cv2.rectangle(img, box, color=color, thickness=2)
                cv2.putText(img, str(text).upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                #adjust_motor()
            else:
                zoom("-", img)
                cv2.rectangle(img, box, color=color, thickness=2)
                cv2.putText(img, str(text).upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                #adjust_motor()
    cv2.imshow('my webcam', img)
    cv2.waitKey(0)
    #cv2.imshow('output', img)
