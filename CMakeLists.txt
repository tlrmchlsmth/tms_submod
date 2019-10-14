cmake_minimum_required(VERSION 3.13)

project(tms_submod)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wunused -qopenmp -mkl=parallel -ipp -DINTEL_IPP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-inline-max-size")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_package(LAPACK REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wunused")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()


if (VALIDATE_WITH_LEMON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DVALIDATE_WITH_LEMON=\"1\"")
    target_link_libraries(tms_submod emon)
endif()

#include_directories( ${MY_SOURCE_DIR} )
#include_directories( ${MY_SOURCE_DIR}/perf )


set(SOURCE_FILES
    perf/perf.cpp
    util.cpp
    #
    la/vector.cpp
    la/matrix.cpp
    #
    test/validate.cpp
    test/bench.cpp
    #
    main.cpp
)

add_executable(tms_submod ${SOURCE_FILES})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(tms_submod ${LAPACK_LIBRARIES})
endif()
