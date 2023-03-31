# my_tf_object_detection.
# Version:20230308 Erling Lindholm
#
# First attemt to use object detection with a pretrained model from tensorflow
#
# Usage: python .\my_tf_object_detection.py
#
# when presented with a remote control or a mobile phone the program will stop
# Output:
# person
# person
# cell phone
# remote
# cell phone
#
# Program stopped
#
import os
import cv2
import numpy as np
import urllib
import matplotlib.pyplot as plt
import sys

modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"
with open(classFile) as fp:
    labels = fp.read().split("\n")
print(labels)
# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
# For ach file in the directory
def detect_objects(net, im):
    dim = 300

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0,0,0), swapRB=True, crop=False)

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects
def display_text(im, text, x, y):

    # Get text size
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv2.rectangle(im, (x,y-dim[1] - baseline), (x + dim[0], y + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle
    cv2.putText(im, text, (x, y-5 ), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

def display_objects(im, objects, threshold = 0.25):

    rows = im.shape[0]; cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30,10)); plt.imshow(mp_img); plt.show();
def name_obj(wanted_name, objects, threshold = 0.25):
    detected = False
    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])


        # Check if the detection is of good quality
        if score > threshold:
            print("{}".format(labels[classId]))
            if "{}".format(labels[classId]) == wanted_name:
                detected = True
    return detected
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
    # cv2.imshow(win_name, frame)
    im = frame
    objects = detect_objects(net, im)
    # if name_objects('cell phone', objects) == 1
    # display_objects(im, objects)
    if name_obj('remote', objects):
        break
    # cv2.waitKey(8000)

source.release()
cv2.destroyWindow(win_name)
