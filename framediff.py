# -*- coding: utf-8 -*-

"""
@date: 2023/6/17 下午9:35
@file: framediff.py
@author: zj
@description: 
"""

import cv2


def inter_frame_diff(frame, last_frame):
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(blur1, blur2)
    ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    element_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element_rect)

    cnts, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # if cv2.contourArea(cnt) < 1000:
        if w * h < 1000:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def three_frame_diff(frame, last_frame, penultimate_frame):
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(penultimate_frame, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    blur3 = cv2.GaussianBlur(gray3, (5, 5), 0)

    diff1 = cv2.absdiff(blur1, blur2)
    diff2 = cv2.absdiff(blur2, blur3)
    # diff = diff1 | diff2
    diff = cv2.bitwise_and(diff1, diff2)

    ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    element_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element_rect, iterations=4)

    cnts, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # if cv2.contourArea(cnt) < 1000:
        if w * h < 1000:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame
