cmake_minimum_required(VERSION 3.12)
project(TrieGPUProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Include src and bench directories
add_subdirectory(src)
add_subdirectory(bench)