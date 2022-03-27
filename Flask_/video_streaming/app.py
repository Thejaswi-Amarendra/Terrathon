from flask import Flask,render_template,Response
import requests
import cv2
import numpy as np
import time
import imutils
import pyttsx3

app=Flask(__name__)



def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
 
    distance = (real_face_width * Focal_Length)/face_width_in_frame
 
    # return the distance
    return distance

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


obj_width_dict ={'person':[42,150], 'car':[175,500], 'motorbike': [80,300], 'bus': [250,500], 'train': [350,1000], 'traffic light': [29,600], 'stop sign': [61,400], 'dog':[30, 200]}

classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

url = "http://192.168.255.219:8080/shot.jpg"


def generate_frames():
    while True:
            
        danger = False
        objects = []

        img_ = requests.get(url)
        img_arr = np.array(bytearray(img_.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800) 
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
                if label not in objects:
                    objects.append(label)
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                
                if label in obj_width_dict.keys():
                    
                    distance = Distance_finder(709.883, obj_width_dict[label][0], w)
                    if distance < obj_width_dict[label][1]:
                        danger = True
                        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
                        cv2.putText(img, label + ' ' +confidence+' '+ str(round(distance, 2))+'cm', (x,y+20), font, 2, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                        cv2.putText(img, label + ' ' +confidence+' '+ str(round(distance, 2))+'cm', (x,y+20), font, 2, (0, 255, 0), 2)
                # cv2.putText(img, label + ' ' +confidence, (x,y+20), font, 2, (255, 255, 255), 2)
                

                # if label=="person":
                    
                #     print("Waits for 3 seconds")
                #     time.sleep(3)
            if danger==True:
                print(" ".join(objects))
                
                pyobj.say(" ".join(objects))

                time.sleep(3)
        
        
        ret,buffer=cv2.imencode('.jpg',img)
        fra=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fra + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)