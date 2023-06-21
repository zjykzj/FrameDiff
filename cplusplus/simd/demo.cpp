//
// Created by zj on 23-6-21.
//

#include <iostream>
#include <list>
#include <string>

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/opencv.hpp"
#ifndef SIMD_OPENCV_ENABLE
#define SIMD_OPENCV_ENABLE
#endif
#include "Simd/SimdMotion.hpp"

using namespace Simd::Motion;
typedef std::list<Event> EventList;
typedef Simd::Pixel::Bgr24 Color;

const Color Red(0, 0, 255), Yellow(0, 255, 255), White(0, 255, 255);

void Annotate(const Metadata& metadata, const Simd::Font& font, EventList& events, View& image) {
    for (size_t i = 0; i < metadata.objects.size(); i++) {
        const Object& object = metadata.objects[i];
        bool alarmed = false;
        for (size_t j = 0; j < metadata.events.size(); ++j) {
            const Event& event = metadata.events[j];
            if (event.objectId == object.id) {
                alarmed = true;
                break;
            }
        }
        Color color = alarmed ? Red : Yellow;
        int width = alarmed ? 2 : 1;
        Simd::DrawRectangle(image, object.rect, color, width);
        font.Draw(image, ToString(object.id), Point(object.rect.left, object.rect.top - font.Height()), color);
        for (size_t j = 1; j < object.trajectory.size(); ++j)
            Simd::DrawLine(image, object.trajectory[j - 1].point, object.trajectory[j].point, color, width);
    }
    for (size_t i = 0; i < metadata.events.size(); ++i) {
        events.push_front(metadata.events[i]);
        if (events.size() * font.Height() > image.height) events.pop_back();
    }
    Point location;
    for (EventList::const_iterator it = events.begin(); it != events.end(); ++it) {
        std::stringstream ss;
        Color color = White;
        switch (it->type) {
            case Event::ObjectIn:
                ss << "in " << it->objectId;
                break;
            case Event::ObjectOut:
                ss << "out " << it->objectId;
                break;
            case Event::SabotageOn:
                ss << "SABOTAGE ON";
                color = Red;
                break;
            case Event::SabotageOff:
                ss << "SABOTAGE OFF";
                color = Red;
                break;
        };
        font.Draw(image, ss.str(), location, color);
        location.y += font.Height();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "You have to set video source! It can be 0 for camera or video file name." << std::endl;
        return 1;
    }
    std::string source = argv[1], output = argc > 2 ? argv[2] : "";

    cv::VideoCapture capture;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (source == "0")
        capture.open(0);
    else
        capture.open(source);
    if (!capture.isOpened()) {
        std::cout << "Can't capture '" << source << "' !" << std::endl;
        return 1;
    }

    cv::VideoWriter writer;
    if (output.size()) {
        writer.open(output, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), capture.get(cv::CAP_PROP_FPS),
                    cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
        if (!writer.isOpened()) {
            std::cout << "Can't open output file '" << output << "' !" << std::endl;
            return 1;
        }
    }

    EventList events;
    Detector detector;
    Simd::Font font((int)capture.get(cv::CAP_PROP_FRAME_HEIGHT) / 32);

#if 1
    // There is an example of change of parameters to detect shooting star in the night sky:
    Model model;
    model.size = FSize(0.01, 0.01);
    detector.SetModel(model);

    Options options;
    options.TrackingAdditionalLinking = 5;
    options.ClassificationShiftMin = 0.01;
    options.ClassificationTimeMin = 0.01;
    options.DifferenceDxFeatureWeight = 0;
    options.DifferenceDyFeatureWeight = 0;
    detector.SetOptions(options);
#endif

    const char* WINDOW_NAME = "MotionDetector";
    cv::namedWindow(WINDOW_NAME, 1);
    for (;;) {
        cv::Mat frame;
        if (!capture.read(frame)) break;
        View image = frame;
        Frame input(image, false, capture.get(cv::CAP_PROP_POS_MSEC) * 0.001);
        Metadata metadata;

        detector.NextFrame(input, metadata);

        Annotate(metadata, font, events, image);

        cv::imshow(WINDOW_NAME, frame);
        if (writer.isOpened()) writer.write(frame);
        if (cv::waitKey(1) == 27)  // "press 'Esc' to break video";
            break;
    }
    return 0;
}
