FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(cosyvoice-tests
    test_config.cpp
    test_graph_routing.cpp
    test_conv1d_contiguity.cpp
    # test_stft_roundtrip.cpp  # TODO: depends on internal fft/stft symbols hidden by CXX_VISIBILITY_PRESET; re-enable after symbol export or static-link workaround
    test_dit_contiguity.cpp
    test_im2col_chunking.cpp
)

target_link_libraries(cosyvoice-tests PRIVATE cosyvoice ggml Catch2::Catch2WithMain)
target_include_directories(cosyvoice-tests PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/vendor/miniaudio"
    "${CMAKE_SOURCE_DIR}/common"
    "${GGML_SOURCE_DIR}/include"
)

include(CTest)
include(Catch)
catch_discover_tests(cosyvoice-tests)
