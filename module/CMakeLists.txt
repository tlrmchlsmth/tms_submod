cmake_minimum_required(VERSION 3.14)

project(pymnp)

#    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wunused -qopenmp -mkl=parallel -ipp")
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wunused -qopenmp -mkl=parallel -ipp")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

include_directories( ${MY_SOURCE_DIR} )
include_directories( ${MY_SOURCE_DIR}/perf )

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
    ../perf/perf.cpp
    ../util.cpp
    ../la/vector.cpp
    ../la/matrix.cpp
    libpymnp.cpp
)

#add_executable(native_hello ${SOURCE_FILES})
add_library(pymnp SHARED ${SOURCE_FILES})
