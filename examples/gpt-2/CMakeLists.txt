#
# gpt-2-server

set(TEST_TARGET gpt-2-server)
add_executable(${TEST_TARGET} server.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
find_package(Threads REQUIRED)
target_link_libraries(${TEST_TARGET} PRIVATE Threads::Threads)
if (WIN32)
    target_link_libraries(${TEST_TARGET} PRIVATE ws2_32)
endif()
