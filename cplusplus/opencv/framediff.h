//
// Created by zj on 23-6-20.
//

#ifndef CPLUSPLUS__FRAMEDIFF_H_
#define CPLUSPLUS__FRAMEDIFF_H_

#include <opencv2/opencv.hpp>

class InterFrameDiff {
   public:
    InterFrameDiff() = default;
    ~InterFrameDiff() = default;

    int Run(const cv::Mat& frame, cv::Mat& dst_frame);

   private:
    cv::Mat last_frame;
};

class ThreeFrameDiff {
   public:
    ThreeFrameDiff() = default;
    ~ThreeFrameDiff() = default;

    int Run(const cv::Mat& frame, cv::Mat& dst_frame);

   private:
    cv::Mat last_frame;
    cv::Mat penultimate_frame;
};

#endif  // CPLUSPLUS__FRAMEDIFF_H_
