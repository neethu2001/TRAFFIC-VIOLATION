import cv2
import numpy as np

def detect_hel():
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
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    violations = []
    triple_violations = []
    for i in range(len(boxes)):
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

            # Draw all detected objects
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)




      # Highlight helmet violations
    for (i, j) in violations:
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_bike = img[y:y+h, x:x+w]
        cv2.imwrite(f"violation_bike_{i}.png", cropped_bike)
        
        cv2.putText(img, "Helmet Violation", (x-10,y-100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
       

    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def is_centroid_inside(x1, y1, w1, h1, x2, y2, w2, h2):
    centroid_x2 = x2 + w2 // 2
    centroid_y2 = y2 + h2 // 2
    
    x_in_range = x1 <= centroid_x2 and centroid_x2 <= x1 + w1
    y_in_range = y1 <= centroid_y2 and centroid_y2 <= y1 + h1
    
    return x_in_range and y_in_range


