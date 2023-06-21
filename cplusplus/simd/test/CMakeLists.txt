cmake_minimum_required(VERSION 3.24)

add_executable(test_simd test_simd.cpp)
target_link_libraries(test_simd ${OpenCV_LIBS} ${Simd_LIB} ${EXT_LIB})

add_executable(contour contour.cpp)
target_link_libraries(contour ${OpenCV_LIBS} ${Simd_LIB} ${EXT_LIB})