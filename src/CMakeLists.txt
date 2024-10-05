# src/CMakeLists.txt
set(SRC_FILES TrieGPU.cu)

add_library(TrieGPU STATIC ${SRC_FILES})

target_include_directories(TrieGPU PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})