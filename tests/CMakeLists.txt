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
    test_dit_contiguity.cpp
    test_im2col_chunking.cpp
)

# test_stft_roundtrip.cpp exercises internal fft/stft symbols (ggml_stft, ggml_istft,
# create_fft_context, create_istft_context) defined in src/fft.cpp. Under the default
# shared build these are hidden by CXX_VISIBILITY_PRESET, so the test only links against
# a static cosyvoice. Configure with -DBUILD_SHARED_LIBS=OFF to enable it.
if(NOT BUILD_SHARED_LIBS)
    target_sources(cosyvoice-tests PRIVATE test_stft_roundtrip.cpp)
else()
    message(STATUS "tests: test_stft_roundtrip.cpp skipped (requires -DBUILD_SHARED_LIBS=OFF)")
endif()

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
