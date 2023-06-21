//
// Created by zj on 23-6-21.
//

#ifndef FRAMEDIFF__FRAMEDIFF_H_
#define FRAMEDIFF__FRAMEDIFF_H_

#include "opencv2/opencv.hpp"
#ifndef SIMD_OPENCV_ENABLE
#define SIMD_OPENCV_ENABLE
#endif
#include "Simd/SimdLib.hpp"

typedef Simd::View<Simd::Allocator> View;

class InterFrameDiff {
   public:
    InterFrameDiff() = default;
    ~InterFrameDiff() = default;

    int Run(const cv::Mat& frame, cv::Mat& dst_frame);

   private:
    View* last_frame = nullptr;
};

class ThreeFrameDiff {
   public:
    ThreeFrameDiff() = default;
    ~ThreeFrameDiff() = default;

    int Run(const cv::Mat& frame, cv::Mat& dst_frame);

   private:
    View* last_frame = nullptr;
    View* penultimate_frame = nullptr;
};

#endif  // FRAMEDIFF__FRAMEDIFF_H_
