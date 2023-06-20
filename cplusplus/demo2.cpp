#include <iostream>

#include "framediff.h"
#include "opencv2/opencv.hpp"

int main() {
    ThreeFrameDiff frame_diff;

    const char* video_path = "../../assets/vtest.avi";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cout << "Cannot open video." << std::endl;
        exit(-1);
    }

    int fps = capture.get(cv::CAP_PROP_FPS);
    int fw = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int fh = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);

    while (true) {
        cv::Mat frame;
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "Can't receive frame (stream end?). Exiting ..." << std::endl;
            break;
        }

        cv::Mat dst_frame;
        frame_diff.Run(frame, dst_frame);

        assert(!dst_frame.empty());
        cv::imshow("frame", dst_frame);
        if (cv::waitKey(1) == (int)'q') {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
