# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/7 14:10
@File    : gmm_show.py
@Author  : zj
@Description: 
"""

import cv2

import numpy as np

if __name__ == '__main__':
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    # subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    capture = cv2.VideoCapture("../assets/vtest.avi")
    if not capture.isOpened():
        print("Error opening video stream or file")
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fgmask = subtractor.apply(frame)

        cv2.imshow("fgmask", fgmask)
        cv2.waitKey(10)
