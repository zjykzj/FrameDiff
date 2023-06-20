//
// Created by zj on 23-6-20.
//

#include "framediff.h"

int InterFrameDiff::Run(const cv::Mat &frame, cv::Mat &dst_frame) {
    int ret = 0;

    frame.copyTo(dst_frame);

    if (last_frame.empty()) {
        ret = -1;
    } else {
        cv::Mat gray1;
        cv::cvtColor(frame, gray1, cv::COLOR_BGR2GRAY);
        cv::Mat gray2;
        cv::cvtColor(last_frame, gray2, cv::COLOR_BGR2GRAY);

        cv::Mat blur1;
        cv::GaussianBlur(gray1, blur1, cv::Size(5, 5), 0);
        cv::Mat blur2;
        cv::GaussianBlur(gray2, blur2, cv::Size(5, 5), 0);

        cv::Mat diff;
        cv::absdiff(blur1, blur2, diff);

        cv::Mat thresh;
        cv::threshold(diff, thresh, 25, 255, cv::THRESH_BINARY);

        cv::Mat close;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(thresh, close, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(close, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL,
                         cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

        for (const auto &cnt : contours) {
            cv::Rect rect = cv::boundingRect(cnt);

            int area = rect.width * rect.height;
            if (area < 1000) continue;

            cv::rectangle(dst_frame, rect, cv::Scalar(0, 255, 0), 2);
        }
    }

    last_frame = frame.clone();
    return ret;
}

int ThreeFrameDiff::Run(const cv::Mat &frame, cv::Mat &dst_frame) {
    int ret = 0;

    frame.copyTo(dst_frame);

    if (last_frame.empty()) {
        last_frame = frame.clone();
        ret = -1;
    } else if (penultimate_frame.empty()) {
        penultimate_frame = last_frame.clone();
        last_frame = frame.clone();
        ret = -2;
    } else {
        cv::Mat gray1;
        cv::cvtColor(frame, gray1, cv::COLOR_BGR2GRAY);
        cv::Mat gray2;
        cv::cvtColor(last_frame, gray2, cv::COLOR_BGR2GRAY);
        cv::Mat gray3;
        cv::cvtColor(penultimate_frame, gray3, cv::COLOR_BGR2GRAY);

        cv::Mat blur1;
        cv::GaussianBlur(gray1, blur1, cv::Size(5, 5), 0);
        cv::Mat blur2;
        cv::GaussianBlur(gray2, blur2, cv::Size(5, 5), 0);
        cv::Mat blur3;
        cv::GaussianBlur(gray3, blur3, cv::Size(5, 5), 0);

        cv::Mat diff1;
        cv::absdiff(blur1, blur2, diff1);
        cv::Mat diff2;
        cv::absdiff(blur2, blur3, diff2);
        cv::Mat diff;
        cv::bitwise_and(diff1, diff2, diff);

        cv::Mat thresh;
        cv::threshold(diff, thresh, 25, 255, cv::THRESH_BINARY);

        cv::Mat close;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(thresh, close, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 4);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(close, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL,
                         cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

        for (const auto &cnt : contours) {
            cv::Rect rect = cv::boundingRect(cnt);

            int area = rect.width * rect.height;
            if (area < 1000) continue;

            cv::rectangle(dst_frame, rect, cv::Scalar(0, 255, 0), 2);
        }

        penultimate_frame = last_frame.clone();
        last_frame = frame.clone();
    }

    return ret;
}
