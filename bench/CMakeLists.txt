# bench/CMakeLists.txt
set(SRC_FILES TrieExample.cu)

add_executable(TrieExample ${SRC_FILES})

target_link_libraries(TrieExample PRIVATE TrieGPU)

target_include_directories(TrieExample PRIVATE ${CMAKE_SOURCE_DIR}/src)