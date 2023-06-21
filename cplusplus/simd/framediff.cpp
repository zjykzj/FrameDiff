//
// Created by zj on 23-6-21.
//

#include "framediff.h"

int InterFrameDiff::Run(const cv::Mat& frame, cv::Mat& dst_frame) {
    int ret = 0;

    dst_frame = frame.clone();
    View view_frame = frame;

    if (last_frame == nullptr) {
        last_frame = view_frame.Clone();
        ret = -1;
    } else {
        View view_gray1(view_frame.width, view_frame.height, View::Gray8);
        BgrToGray(view_frame, view_gray1);
        View view_gray2(view_frame.width, view_frame.height, View::Gray8);
        BgrToGray(*last_frame, view_gray2);

        View view_blur1(view_frame.width, view_frame.height, View::Gray8);
        GaussianBlur3x3(view_gray1, view_blur1);
        View view_blur2(view_frame.width, view_frame.height, View::Gray8);
        GaussianBlur3x3(view_gray2, view_blur2);

        View view_diff(view_frame.width, view_frame.height, View::Gray8);
        AbsDifference(view_blur1, view_blur2, view_diff);

        View view_thresh(view_frame.width, view_frame.height, View::Gray8);
        Binarization(view_diff, 25, 255, 0, view_thresh, SimdCompareType::SimdCompareGreater);

        cv::Mat thresh = view_thresh;

        cv::Mat close;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(thresh, close, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(close, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL,
                         cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            cv::Rect rect = cv::boundingRect(cnt);

            int area = rect.width * rect.height;
            if (area < 1000) continue;

            cv::rectangle(dst_frame, rect, cv::Scalar(0, 255, 0), 2);
        }

        delete last_frame;
        last_frame = view_frame.Clone();
    }

    return ret;
}

int ThreeFrameDiff::Run(const cv::Mat& frame, cv::Mat& dst_frame) {
    int ret = 0;

    dst_frame = frame.clone();
    View view_frame = frame;

    if (last_frame == nullptr) {
        last_frame = view_frame.Clone();
        ret = -1;
    } else if (penultimate_frame == nullptr) {
        penultimate_frame = last_frame->Clone();
        delete last_frame;
        last_frame = view_frame.Clone();
        ret = -2;
    } else {
        View view_gray1(view_frame.width, view_frame.height, View::Gray8);
        BgrToGray(view_frame, view_gray1);
        View view_gray2(view_frame.width, view_frame.height, View::Gray8);
        BgrToGray(*last_frame, view_gray2);
        View view_gray3(view_frame.width, view_frame.height, View::Gray8);
        BgrToGray(*penultimate_frame, view_gray3);

        View view_blur1(view_frame.width, view_frame.height, View::Gray8);
        GaussianBlur3x3(view_gray1, view_blur1);
        View view_blur2(view_frame.width, view_frame.height, View::Gray8);
        GaussianBlur3x3(view_gray2, view_blur2);
        View view_blur3(view_frame.width, view_frame.height, View::Gray8);
        GaussianBlur3x3(view_gray3, view_blur3);

        View view_diff1(view_frame.width, view_frame.height, View::Gray8);
        AbsDifference(view_blur1, view_blur2, view_diff1);
        View view_diff2(view_frame.width, view_frame.height, View::Gray8);
        AbsDifference(view_blur2, view_blur3, view_diff2);
        View view_diff(view_frame.width, view_frame.height, View::Gray8);
        OperationBinary8u(view_diff1, view_diff2, view_diff, SimdOperationBinary8uType::SimdOperationBinary8uAnd);

        View view_thresh(view_frame.width, view_frame.height, View::Gray8);
        Binarization(view_diff, 25, 255, 0, view_thresh, SimdCompareType::SimdCompareGreater);

        cv::Mat thresh = view_thresh;

        cv::Mat close;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(thresh, close, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 4);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(close, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL,
                         cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            cv::Rect rect = cv::boundingRect(cnt);

            int area = rect.width * rect.height;
            if (area < 1000) continue;

            cv::rectangle(dst_frame, rect, cv::Scalar(0, 255, 0), 2);
        }

        delete penultimate_frame;
        penultimate_frame = last_frame->Clone();
        delete last_frame;
        last_frame = view_frame.Clone();
    }

    return ret;
}
