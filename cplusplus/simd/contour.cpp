//
// Created by zj on 23-6-21.
//

#include "Simd/SimdContour.hpp"
#include "Simd/SimdDrawing.hpp"

int main() {
    typedef Simd::ContourDetector<Simd::Allocator> ContourDetector;

    ContourDetector::View image;
    //    image.Load("../../data/image/face/lena.pgm");
    image.Load("/home/zj/pp/Simd/data/image/face/lena.pgm");

    ContourDetector contourDetector;

    contourDetector.Init(image.Size());

    ContourDetector::Contours contours;
    contourDetector.Detect(image, contours);

    for (size_t i = 0; i < contours.size(); ++i) {
        for (size_t j = 1; j < contours[i].size(); ++j)
            Simd::DrawLine(image, contours[i][j - 1], contours[i][j], uint8_t(255));
    }
    image.Save("result.pgm");

    return 0;
}