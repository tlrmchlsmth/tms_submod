cmake_minimum_required(VERSION 3.10.2)

project(tms_submodular)

#    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -ipp -qopenmp -O3")
endif()

include_directories( ${MY_SOURCE_DIR} )
include_directories( ${MY_SOURCE_DIR}/perf )

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
    min_norm_point.cpp
    perf/perf.cpp
    #
    test/validate/validate.cpp
    test/validate/validate.h
)

add_executable(tms_submod ${SOURCE_FILES})

