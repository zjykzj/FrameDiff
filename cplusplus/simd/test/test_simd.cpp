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

/*
 * http://ermig1979.github.io/Simd/help/group__conversion.html
 * http://ermig1979.github.io/Simd/help/group__bgr__conversion.html
 */
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

/*
 * http://ermig1979.github.io/Simd/help/group__gaussian__filter.html
 */
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

/*
 * http://ermig1979.github.io/Simd/help/group__correlation.html
 */
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

/*
 * http://ermig1979.github.io/Simd/help/group__correlation.html
 */
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

/*
 * http://ermig1979.github.io/Simd/help/group__binarization.html
 */
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

/*
 * http://ermig1979.github.io/Simd/help/group__binarization.html
 */
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

/*
 * http://ermig1979.github.io/Simd/help/group__cpp__contour.html
 * http://ermig1979.github.io/Simd/help/struct_simd_1_1_contour_detector.html
 */
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

    for (auto& contour : contours) {
        for (size_t j = 1; j < contour.size(); ++j) {
            Simd::DrawLine(view_gray2, contour[j - 1], contour[j], uint8_t(255));
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

/*
 * http://ermig1979.github.io/Simd/help/group__operation.html#ga6154e5dfa9b9ad0f59f3dc75c2322392
 */
void BitwiseAnd() {
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

    View view_thresh1(bgr1.cols, bgr1.rows, View::Gray8);
    Binarization(view_gray1, 25, 255, 0, view_thresh1, SimdCompareType::SimdCompareGreater);

    View view_thresh2(bgr1.cols, bgr1.rows, View::Gray8);
    Binarization(view_gray2, 25, 255, 0, view_thresh2, SimdCompareType::SimdCompareGreater);

    View view_bitwise_and(bgr1.cols, bgr1.rows, View::Gray8);
    OperationBinary8u(view_thresh1, view_thresh2, view_bitwise_and,
                      SimdOperationBinary8uType::SimdOperationBinary8uAnd);

    cv::Mat thresh1 = view_thresh1;
    cv::Mat thresh2 = view_thresh2;
    cv::Mat bitwise_and = view_bitwise_and;

    cv::imshow("bgr1", bgr1);
    cv::imshow("bgr2", bgr2);
    cv::imshow("thresh1", thresh1);
    cv::imshow("thresh2", thresh2);
    cv::imshow("bitwise_and", bitwise_and);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    //    Bgr2Gray();
    //    GaussianBlur();
    //    AbsDifference();
    //    AbsDifference2();
    //    BinaryThresh();
    //    BinaryThresh2();
    //    Contours();
    BitwiseAnd();

    return 0;
}