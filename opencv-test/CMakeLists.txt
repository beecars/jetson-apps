cmake_minimum_required (VERSION 2.8)

project(opencv-test)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv-test opencv-test.cpp)

target_link_libraries(opencv-test ${OpenCV_LIBS})
