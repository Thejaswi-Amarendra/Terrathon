import cv2
import numpy as np
import time

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
 
    distance = (real_face_width * Focal_Length)/face_width_in_frame
 
    # return the distance
    return distance

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
avg_person_width = 42
classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

frame = cv2.VideoCapture(1)

while True:
    _, img = frame.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.7:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    

    if len(boxes)>0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255, size=(len(boxes), 3))

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            if label=='person':
                distance = Distance_finder(709.883, 42, w)
                print(distance)
                cv2.putText(img, label + ' ' +confidence+' '+ str(round(distance, 2))+'cm', (x,y+20), font, 2, (255, 255, 255), 2)
            cv2.putText(img, label + ' ' +confidence, (x,y+20), font, 2, (255, 255, 255), 2)
            print(label)

            # if label=="person":
                
            #     print("Waits for 3 seconds")
            #     time.sleep(3)



    cv2.imshow('img', img)
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

frame.release()