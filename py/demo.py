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
from framediff import inter_frame_diff


def main():
    video_path = "../assets/vtest.avi"
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
