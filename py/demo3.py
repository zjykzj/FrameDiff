# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 10:49
@File    : subtraction.py
@Author  : zj
@Description:

1. [How to Use Background Subtraction Methods](https://docs.opencv.org/4.8.0/d1/dc5/tutorial_background_subtraction.html)
2. [samples/cpp/segment_objects.cpp](https://docs.opencv.org/4.x/d5/de8/samples_2cpp_2segment_objects_8cpp-example.html#a23)
3. [计算机视觉——基于背景差分的运动物体检测和追踪](https://zhuanlan.zhihu.com/p/512267368?utm_id=0)

"""

import cv2
import copy

from framediff import BackgroundSubtractor


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

    subtractor = BackgroundSubtractor()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = subtractor.run(copy.deepcopy(frame))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
