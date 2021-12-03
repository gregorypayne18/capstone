import cv2

#img = cv2.imread('lena.png')
cap = cv2.VideoCapture(1)
cap.set(3, 640) #640
cap.set(4, 480) #480

#read into names file
classNames = []
classFile = 'testnames'
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

#display video
while True:
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, 0.5)

    #if object detected:
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            text = str(classNames[classId-1]) + " " + str(round(confidence*100)) + "%"
            #looking for plates
            if (classNames[classId-1]) == "plate":
                color = red
                cv2.rectangle(img, box, color=color, thickness=2)
                cv2.putText(img, text.upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            else:
                color = green
                cv2.rectangle(img, box, color=color, thickness=2)
                cv2.putText(img, text.upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


    cv2.imshow('output', img)
    cv2.waitKey(1)
