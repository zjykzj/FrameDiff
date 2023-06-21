//
// Created by zj on 23-6-21.
//

#include "opencv2/opencv.hpp"
#ifndef SIMD_OPENCV_ENABLE
#define SIMD_OPENCV_ENABLE
#endif
#include "Simd/SimdContour.hpp"
#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdLib.hpp"

typedef Simd::View<Simd::Allocator> View;
typedef Simd::ContourDetector<Simd::Allocator> ContourDetector;

void Bgr2Gray() {
    cv::Mat bgr;
    bgr = cv::imread("../../test/lena.jpg", cv::IMREAD_COLOR);
    assert(!bgr.empty());

    View view_gray(bgr.cols, bgr.rows, View::Gray8);
    BgrToGray((View)bgr, view_gray);

    cv::Mat gray = view_gray;
    assert(!gray.empty());

    cv::imshow("bgr", bgr);
    cv::imshow("gray", gray);
    cv::waitKey(0);
}

void GaussianBlur() {
    cv::Mat bgr;
    bgr = cv::imread("../../test/lena.jpg", cv::IMREAD_COLOR);
    assert(!bgr.empty());

    View view_gray(bgr.cols, bgr.rows, View::Gray8);
    BgrToGray((View)bgr, view_gray);

    View view_blur(bgr.cols, bgr.rows, View::Gray8);
    GaussianBlur3x3(view_gray, view_blur);

    cv::Mat gray = view_gray;
    assert(!gray.empty());
    cv::Mat blur = view_blur;
    assert(!blur.empty());

    cv::imshow("bgr", bgr);
    cv::imshow("gray", gray);
    cv::imshow("blur", blur);
    cv::waitKey(0);
}

void AbsDifference() {
    cv::Mat bgr;
    bgr = cv::imread("../../test/lena.jpg", cv::IMREAD_COLOR);
    assert(!bgr.empty());

    View view_gray(bgr.cols, bgr.rows, View::Gray8);
    BgrToGray((View)bgr, view_gray);

    View view_blur(bgr.cols, bgr.rows, View::Gray8);
    GaussianBlur3x3(view_gray, view_blur);

    cv::Mat gray = view_gray;
    assert(!gray.empty());
    cv::Mat blur = view_blur;
    assert(!blur.empty());

    View view_diff(bgr.cols, bgr.rows, View::Gray8);
    AbsDifference(view_gray, view_blur, view_diff);

    cv::Mat diff = view_diff;
    assert(!diff.empty());

    cv::imshow("bgr", bgr);
    cv::imshow("gray", gray);
    cv::imshow("blur", blur);
    cv::imshow("diff", diff);
    cv::waitKey(0);
}

void AbsDifference2() {
    cv::Mat bgr1;
    bgr1 = cv::imread("../../test/first.jpg", cv::IMREAD_COLOR);
    assert(!bgr1.empty());
    cv::Mat bgr2;
    bgr2 = cv::imread("../../test/second.jpg", cv::IMREAD_COLOR);
    assert(!bgr2.empty());

    View view_gray1(bgr1.cols, bgr1.rows, View::Gray8);
    BgrToGray((View)bgr1, view_gray1);
    View view_gray2(bgr2.cols, bgr2.rows, View::Gray8);
    BgrToGray((View)bgr2, view_gray2);

    View view_diff(bgr1.cols, bgr1.rows, View::Gray8);
    AbsDifference(view_gray1, view_gray2, view_diff);

    cv::Mat diff = view_diff;
    assert(!diff.empty());

    cv::imshow("bgr1", bgr1);
    cv::imshow("bgr2", bgr2);
    cv::imshow("diff", diff);
    cv::waitKey(0);
}

void BinaryThresh() {
    cv::Mat bgr;
    bgr = cv::imread("../../test/lena.jpg", cv::IMREAD_COLOR);
    assert(!bgr.empty());

    View view_gray(bgr.cols, bgr.rows, View::Gray8);
    BgrToGray((View)bgr, view_gray);

    View view_blur(bgr.cols, bgr.rows, View::Gray8);
    GaussianBlur3x3(view_gray, view_blur);

    cv::Mat gray = view_gray;
    assert(!gray.empty());
    cv::Mat blur = view_blur;
    assert(!blur.empty());

    View view_diff(bgr.cols, bgr.rows, View::Gray8);
    AbsDifference(view_gray, view_blur, view_diff);

    cv::Mat diff = view_diff;
    assert(!diff.empty());

    View view_thresh(bgr.cols, bgr.rows, View::Gray8);
    Binarization(view_diff, 25, 255, 0, view_thresh, SimdCompareType::SimdCompareGreater);

    cv::Mat thresh = view_thresh;
    assert(!thresh.empty());

    cv::imshow("bgr", bgr);
    cv::imshow("gray", gray);
    cv::imshow("blur", blur);
    cv::imshow("diff", diff);
    cv::imshow("thresh", thresh);
    cv::waitKey(0);
}

void BinaryThresh2() {
    cv::Mat bgr1;
    bgr1 = cv::imread("../../test/first.jpg", cv::IMREAD_COLOR);
    assert(!bgr1.empty());
    cv::Mat bgr2;
    bgr2 = cv::imread("../../test/second.jpg", cv::IMREAD_COLOR);
    assert(!bgr2.empty());

    View view_gray1(bgr1.cols, bgr1.rows, View::Gray8);
    BgrToGray((View)bgr1, view_gray1);
    View view_gray2(bgr2.cols, bgr2.rows, View::Gray8);
    BgrToGray((View)bgr2, view_gray2);

    View view_diff(bgr1.cols, bgr1.rows, View::Gray8);
    AbsDifference(view_gray1, view_gray2, view_diff);

    cv::Mat diff = view_diff;
    assert(!diff.empty());

    View view_thresh(bgr1.cols, bgr1.rows, View::Gray8);
    Binarization(view_diff, 25, 255, 0, view_thresh, SimdCompareType::SimdCompareGreater);

    cv::Mat thresh = view_thresh;
    assert(!thresh.empty());

    cv::imwrite("thresh.jpg", thresh);

    cv::imshow("bgr1", bgr1);
    cv::imshow("bgr2", bgr2);
    cv::imshow("diff", diff);
    cv::imshow("thresh", thresh);
    cv::waitKey(0);
}

void Contours() {
    cv::Mat bgr1;
    bgr1 = cv::imread("../../test/first.jpg", cv::IMREAD_COLOR);
    assert(!bgr1.empty());
    cv::Mat bgr2;
    bgr2 = cv::imread("../../test/second.jpg", cv::IMREAD_COLOR);
    assert(!bgr2.empty());

    View view_gray1(bgr1.cols, bgr1.rows, View::Gray8);
    BgrToGray((View)bgr1, view_gray1);
    View view_gray2(bgr2.cols, bgr2.rows, View::Gray8);
    BgrToGray((View)bgr2, view_gray2);

    View view_diff(bgr1.cols, bgr1.rows, View::Gray8);
    AbsDifference(view_gray1, view_gray2, view_diff);

    cv::Mat diff = view_diff;
    assert(!diff.empty());

    View view_thresh(bgr1.cols, bgr1.rows, View::Gray8);
    Binarization(view_diff, 25, 255, 0, view_thresh, SimdCompareType::SimdCompareGreater);

    //    cv::Mat thresh = view_thresh;
    //    assert(!thresh.empty());

    ContourDetector contourDetector;
    contourDetector.Init(view_gray1.Size());

    ContourDetector::Contours contours;
    contourDetector.Detect(view_thresh, contours);

    for (size_t i = 0; i < contours.size(); ++i) {
        for (size_t j = 1; j < contours[i].size(); ++j) {
            Simd::DrawLine(view_gray2, contours[i][j - 1], contours[i][j], uint8_t(255));
            //            Simd::DrawRectangle(view_gray2, )
        }
    }

    cv::Mat thresh = view_thresh;
    assert(!thresh.empty());

    cv::Mat gray2 = view_gray2;

    cv::imshow("bgr1", bgr1);
    cv::imshow("bgr2", bgr2);
    cv::imshow("diff", diff);
    cv::imshow("thresh", thresh);
    cv::imshow("gray2", gray2);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    //    Bgr2Gray();
    //    GaussianBlur();
    //    AbsDifference();
    //    AbsDifference2();
    //    BinaryThresh();
    BinaryThresh2();
    //    Contours();

    return 0;
}