#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import numpy as np
import cv2


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]
CONFIDANCE = 0.2
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

current_script_directory = os.path.dirname(os.path.abspath(__file__))
proto = current_script_directory + '/mobileNet/MobileNetSSD_deploy.prototxt.txt'
model = current_script_directory + '/mobileNet/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto, model)

def detect(net, frame, CONFIDANCE, COLORS, CLASSES):
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDANCE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY)))

    return image, results
