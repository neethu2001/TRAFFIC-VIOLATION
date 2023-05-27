import cv2
import numpy as np
from anpr_video import anpr_detect
import is_inside_centroid

def detect_tri():
    net = cv2.dnn.readNet("yolov4-custom_10000.weights", "yolov4-custom.cfg")
    print("done")

    def resize_image(image, new_width):
        height, width, _ = image.shape
        aspect_ratio = float(height) / float(width)
        new_height = int(new_width * aspect_ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Define the text font and size


    classes = []
    with open("classes.names", "r") as f:
        read = f.readlines()
    for i in range(len(read)):
        classes.append(read[i].strip("\n"))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread('img.jpg')

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
        

                if helmet_count > 2:
                    triple_violations.append(i)

 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

 


    for i in triple_violations:
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_bike = img[y:y+h, x:x+w]
        cv2.imwrite(f"violation_bike_{i}.png", cropped_bike)
        anpr_detect(f"violation_bike_{i}.png")
        cv2.putText(img, "Triple Violation", (x + 20, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
         with open('report.csv','w'):
                csv.writer(place,count,violation,date,time)

    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pr.jpg", img)




