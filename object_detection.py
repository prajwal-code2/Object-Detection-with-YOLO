import cv2
import os
import numpy as np
def detect_objects(image):
    # Load Yolo weights and configuration file
    net = cv2.dnn.readNet("yolo\\yolov3.weights", "yolo\\yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load COCO names (classes)
    with open("yolo\\coco.names", "r") as f:
        classes = [line.strip() for line in f]
    
    img=cv2.imread(image)
    os.remove(image)
    height, width, channels = img.shape
    
    # Preprocess image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Get class IDs, confidences and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confi=str(round(confidences[i],2))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(img, label, (x, y+h+20), font, 1, (0,0,0), 2)
            cv2.putText(img, confi, (x, y), font, 0.5, (0,0,0), 2)

    ret, jpeg = cv2.imencode('.jpg', img)

    frame = jpeg.tobytes()

    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')