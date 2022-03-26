import cv2
import numpy as np

def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image* measured_distance)/ real_width
    return focal_length

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

f=0
for op in range(0, 5):
    img = cv2.imread("reference_images/ref"+str(op+1)+".jpg")
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
            image_width = w
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + ' ' +confidence, (x,y+20), font, 2, (255, 255, 255), 2)
            # print(label)

            # if label=="person":
                    
            #     print("Waits for 3 seconds")
            #     time.sleep(3)



    cv2.imshow('img', img)

    

    known_width = 8.6
    known_distance = [54, 69, 84, 99, 114]
    f+=FocalLength(known_distance[op], 8.6, image_width)
    print(FocalLength(known_distance[op], 8.6, image_width))

print(f/5)
cv2.waitKey()
cv2.destroyAllWindows()