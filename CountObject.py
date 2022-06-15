#!/usr/bin/env python
# coding: utf-8

# In[6]:
import os

import cv2
import numpy as np

# In[7]:
class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]


def count_object(file_path, obj_to_count, file_name, upload_folder):
    img_to_detect = cv2.imread(file_path)

    # In[8]:

    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)

    # In[9]:

    # In[10]:

    class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors, (16, 1))

    # In[11]:

    yolo_model = cv2.dnn.readNetFromDarknet('./YoloConfig/yolov3.cfg', './YoloConfig/yolov3.weights')
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    # In[12]:

    yolo_model.setInput(img_blob)
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    # In[13]:

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    # In[14]:

    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
        # loop over the detections
        for object_detection in object_detection_layer:

            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            # take only predictions with confidence more than 50%
            if prediction_confidence > 0.25:
                # obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))

                ############## NMS Change 2 ###############
                # save class id, start x, y, width & height, confidences in a list for nms processing
                # make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                ############## NMS Change 2 END ###########

    # In[15]:

    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    # In[16]:

    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    num_object_found = 0
    for max_valueid in max_value_ids:
        max_class_id = max_valueid
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]

        # get the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
        ############## NMS Change 3 END ###########

        # obtain the bounding box end co-oridnates
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        # get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]

        # convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]

        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        num_object_found += count_object_found(obj_to_count, predicted_class_label)

        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    box_color, 1)

    os.remove(file_path)
    save_path = os.path.join(upload_folder, f'{file_name}')
    cv2.imwrite(save_path, img_to_detect)
    return num_object_found


def count_object_found(obj_to_count, predicted_labels):
    obj_to_count = str.lower(obj_to_count)
    predicted_labels = str.lower(predicted_labels)
    return predicted_labels.count(obj_to_count)
