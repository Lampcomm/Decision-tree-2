cmake_minimum_required(VERSION 3.27)
project(Decision_tree_2)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -D PrintTrainingTime")

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize=undefined")

#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64")
endif ()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
set(PROJECT_FILES ${SOURCES} ${HEADERS})

add_executable(Decision_tree_2 ${PROJECT_FILES})

find_package(OpenMP REQUIRED)
target_link_libraries(Decision_tree_2 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(Decision_tree_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET Decision_tree_2 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)