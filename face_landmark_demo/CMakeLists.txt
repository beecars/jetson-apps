cmake_minimum_required (VERSION 2.8)

project(face-landmark-detect)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(face-landmark-detect face-landmark-detect.cpp)

target_link_libraries(face-landmark-detect ${OpenCV_LIBS})
