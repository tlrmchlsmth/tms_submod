cmake_minimum_required(VERSION 2.8.12.2)

project(tms_submodular)

#    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mkl=parallel -ipp -qopenmp -O3")
endif()

if (VALIDATE_WITH_LEMON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DVALIDATE_WITH_LEMON=\"1\"")
    target_link_libraries(tms_submod emon)
endif()

include_directories( ${MY_SOURCE_DIR} )
include_directories( ${MY_SOURCE_DIR}/perf )

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
    main.cpp
    perf/perf.cpp
    util.cpp
    #
    test/validate.cpp
    test/bench.cpp
)

add_executable(tms_submod ${SOURCE_FILES})

