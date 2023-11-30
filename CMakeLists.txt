cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project(FilterGraph)

add_compile_options(-O3 -lrt -std=c++17 -march=native -fPIC -w -fopenmp)

add_executable(build_index build_index.cpp)
target_link_options(build_index PUBLIC -O3 -fopenmp)

add_executable(txt2spmat txt2spmat.cpp)

add_executable(format_convert format_convert.cpp)

add_executable(graph_profile graph_profile.cpp)

add_executable(search_index search_index.cpp)
target_link_options(search_index PUBLIC -O3 -fopenmp)