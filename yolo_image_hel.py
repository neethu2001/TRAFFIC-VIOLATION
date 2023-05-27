import cv2
import numpy as np
import is_centroid_inside

def detect_hel():
    net = cv2.dnn.read("yolov4-custom_10000.weights", "yolov4-custom.cfg")
    print("done")

    def resize_image(image, new_width):
        height, width, _ = image.shape
        aspect_ratio = float(height) / float(width)
        new_height = int(new_width * aspect_ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)




    classes = []
    with open("classes.names", "r") as f:
        read = f.readlines()
    for i in range(len(read)):
        classes.append(read[i].strip("\n"))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    

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
    
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            if label == 'bike':
                helmet_count = 0
                for j in range(len(boxes)):
                    if j in indexes and i != j:
                        x2, y2, w2, h2 = boxes[j]
                        label2 = classes[class_ids[j]]
                        if label2 in ['hel', 'nohel'] and is_centroid_inside(x, y, w, h, x2, y2, w2, h2):
                            helmet_count += 1
                            if label2 == 'nohel':
                                violations.append((i, j))

                if helmet_count > 2:
                    triple_violations.append(i)

           
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            with open('report.csv','w'):
                csv.writer(place,count,violation,date,time)




    
    for (i, j) in violations:
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_bike = img[y:y+h, x:x+w]
        cv2.imwrite(f"violation_bike_{i}.png", cropped_bike)
        
        cv2.putText(img, "Helmet Violation", (x-10,y-100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
       

    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    





