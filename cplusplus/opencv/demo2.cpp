#include <iostream>

#include "framediff.h"
#include "opencv2/opencv.hpp"

int main() {
    ThreeFrameDiff frame_diff;

    const char* video_path = "../../../assets/vtest.avi";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cout << "Cannot open video." << std::endl;
        exit(-1);
    }

    int fps = capture.get(cv::CAP_PROP_FPS);
    int fw = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int fh = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);

    double total_t = 0;
    int num = 0;
    while (true) {
        cv::Mat frame;
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "Can't receive frame (stream end?). Exiting ..." << std::endl;
            break;
        }

        cv::Mat dst_frame;
        auto t1 = std::chrono::high_resolution_clock::now();
        frame_diff.Run(frame, dst_frame);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        total_t += fp_ms.count();
        num++;

        assert(!dst_frame.empty());
        cv::imshow("frame", dst_frame);
        if (cv::waitKey(1) == (int)'q') {
            break;
        }
    }
    capture.release();
    cv::destroyAllWindows();

    std::cout << "One Process need: " << (total_t / num) << "ms" << std::endl;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
