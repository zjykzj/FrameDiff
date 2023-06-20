cmake_minimum_required(VERSION 3.16)
project(FrameDiff)

# set opencv
set(OpenCV_DIR /home/zj/software/opencv/install_4_5_2/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

set(CMAKE_CXX_STANDARD 17)

add_executable(demo1 demo1.cpp framediff.cpp)
target_link_libraries(demo1 ${OpenCV_LIBS})

add_executable(demo2 demo2.cpp framediff.cpp)
target_link_libraries(demo2 ${OpenCV_LIBS})