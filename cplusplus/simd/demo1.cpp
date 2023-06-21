//
// Created by zj on 23-6-21.
//

#include <iostream>

#include "framediff.h"
#include "opencv2/opencv.hpp"

/*
 * https://docs.opencv.org/3.4/d3/d9c/samples_2cpp_2tutorial_code_2videoio_2video-write_2video-write_8cpp-example.html
 */
int main() {
    InterFrameDiff frame_diff;

    const char* video_path = "../../../assets/vtest.avi";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cout << "Cannot open video." << std::endl;
        return -1;
    }

    int ex = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));       // Get Codec Type- Int form
    cv::Size S = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH),  // Acquire input size
                          (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const char* save_path = "../../../assets/framediff.avi";
    cv::VideoWriter writer;
    writer.open(save_path, ex, capture.get(cv::CAP_PROP_FPS), S, true);
    if (!writer.isOpened()) {
        std::cout << "Could not open the output video for write: " << save_path << std::endl;
        return -1;
    }

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8), (char)((ex & 0XFF0000) >> 16),
                  (char)((ex & 0XFF000000) >> 24), 0};
    std::cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
              << " of nr#: " << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    std::cout << "Input codec type: " << EXT << std::endl;

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

        writer.write(dst_frame);

        cv::imshow("frame", dst_frame);
        if (cv::waitKey(1) == (int)'q') {
            break;
        }
    }

    capture.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
