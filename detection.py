import cv2
import numpy as np

url = '/home/ubuntu/app/yolo/'

#food file name
def foodDetect(foodname):
    net = cv2.dnn.readNet(url+"yolov2-food100.weights", url+"yolov2-food100.cfg")
    classes = []
    with open(url+"obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.imread(foodname)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb=cv2.split(img)
    Y = cv2.equalizeHist(Y)
    img = cv2.merge([Y,Cr,Cb])
    img = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    labels = []
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:          
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
         
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                labels.append(classes[class_id])
    
    #box draw
    '''indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            labels.append(label)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    cv2.imwrite("detect.png",img)'''
    return labels
