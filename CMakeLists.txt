cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project(FilterGraph)

add_compile_options(-O3 -lrt -std=c++17 -march=native -fPIC -w -fopenmp)

add_executable(build_index build_index.cpp)
target_link_options(build_index PUBLIC -O3 -fopenmp)

add_executable(build_index_v3 build_index_v3.cpp)
target_link_options(build_index_v3 PUBLIC -O3 -fopenmp)

add_executable(build_heuristic build_heuristic.cpp)
target_link_options(build_heuristic PUBLIC -O3 -fopenmp)

add_executable(search_index_v3 search_index_v3.cpp)
target_link_options(search_index_v3 PUBLIC -O3 -fopenmp)

add_executable(search_index search_index.cpp)
target_link_options(search_index PUBLIC -O3 -fopenmp)