# -*- coding: utf-8 -*-

"""
@date: 2023/6/17 下午8:41
@file: demo.py
@author: zj
@description:
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
"""
import copy

import cv2


def inter_frame_diff(frame, last_frame):
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(blur1, blur2)
    ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 100:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def main():
    video_path = "./assets/vtest.avi"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot read video.")
        exit(-1)

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if last_frame is None:
            last_frame = copy.deepcopy(frame)
            continue

        dst_frame = inter_frame_diff(copy.deepcopy(frame), last_frame)
        last_frame = copy.deepcopy(frame)

        # Display the resulting frame
        cv2.imshow('frame', dst_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
