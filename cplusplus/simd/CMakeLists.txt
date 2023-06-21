cmake_minimum_required(VERSION 3.24)
project(FrameDiff)

# set opencv
set(OpenCV_DIR /home/zj/software/opencv/install_4_5_2/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

set(CMAKE_CXX_STANDARD 17)

include_directories(/home/zj/pp/Simd/src)
set(Simd_LIB /home/zj/pp/Simd/build/libSimd.a)

set(EXT_LIB pthread)
# --------------------------------------------------------

add_executable(demo1 demo1.cpp framediff.cpp)
target_link_libraries(demo1 ${OpenCV_LIBS} ${Simd_LIB} ${EXT_LIB})

# ---------------------------------------------------------

add_subdirectory(${CMAKE_SOURCE_DIR}/test)