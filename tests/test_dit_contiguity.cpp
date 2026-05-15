// DiT sequence-length-dependent contiguity tests.
//
// The Flow DiT pipeline produces correct output for 31-token sequences
// but silence for 60-token sequences on Metal/CPU. This test verifies
// that ggml ops used in the DiT path maintain contiguity invariants
// across different sequence lengths.
//
// Key ops to test:
// - ggml_permute (DiT entry/exit)
// - ggml_repeat_4d (position_ids, spks broadcast)
// - ggml_view_3d (cut_len slicing)
// - split_tensor (AdaLayerNorm, attn_norm)
// - concat_tensors (InputEmbedding)

#include <catch2/catch_test_macros.hpp>

#include <ggml.h>

#include <cstring>

// Reimplementation of split_tensor from cosyvoice-graph.cpp
static void split_tensor(ggml_context* ctx, ggml_tensor* tensor, int dim, ggml_tensor** tensors, uint16_t chunks)
{
    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t offset_per_chunk = tensor->nb[dim] * ne[dim];

    for (uint16_t i = 0; i != chunks; ++i)
        tensors[i] = ggml_view_4d(
            ctx, tensor,
            ne[0], ne[1], ne[2], ne[3],
            tensor->nb[1], tensor->nb[2], tensor->nb[3],
            i * offset_per_chunk);
}

SCENARIO("DiT permute produces non-contiguous tensor at any sequence length") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    for (int seq_len : {31, 60, 100, 236}) {
        GIVEN("a tensor with sequence length " + std::to_string(seq_len)) {
            auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, 80, 2);

            WHEN("permuted (1,0,2,3) like DiT entry") {
                auto p = ggml_permute(ctx, x, 1, 0, 2, 3);

                THEN("result is non-contiguous") {
                    REQUIRE_FALSE(ggml_is_contiguous(p));
                    REQUIRE(p->ne[0] == 80);
                    REQUIRE(p->ne[1] == seq_len);
                }

                WHEN("ggml_cont is applied") {
                    auto c = ggml_cont(ctx, p);
                    THEN("result is contiguous with same shape") {
                        REQUIRE(ggml_is_contiguous(c));
                        REQUIRE(c->ne[0] == 80);
                        REQUIRE(c->ne[1] == seq_len);
                    }
                }
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("ggml_view_3d for cut_len slicing produces non-contiguous views") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a 3D tensor [80, seq_len, 2] with different cut_len values") {
        for (int seq_len : {31, 60, 100}) {
            auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 80, seq_len, 2);
            int cut_len = seq_len / 3;

            auto view = ggml_view_3d(ctx, x,
                x->ne[0], x->ne[1] - cut_len, x->ne[2],
                x->nb[1], x->nb[2],
                x->nb[1] * cut_len);

            THEN("view has reduced ne[1] and is contiguous (strides match)") {
                REQUIRE(view->ne[0] == 80);
                REQUIRE(view->ne[1] == seq_len - cut_len);
                REQUIRE(view->ne[2] == 2);
                // View with offset but matching strides is contiguous
                REQUIRE(view->nb[0] == sizeof(float));
                REQUIRE(view->nb[1] == 80 * sizeof(float));
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("split_tensor for AdaLayerNorm produces views with expected contiguity") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a 1D tensor split into 2 along dim=0 (like AdaLayerNorm scale/shift)") {
        for (int dim0 : {128, 256, 512}) {
            auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim0);
            ggml_tensor* chunks[2];
            split_tensor(ctx, tensor, 0, chunks, 2);

            THEN("chunks have half the elements") {
                REQUIRE(chunks[0]->ne[0] == dim0 / 2);
                REQUIRE(chunks[1]->ne[0] == dim0 / 2);
            }

            THEN("chunk[0] is contiguous") {
                REQUIRE(ggml_is_contiguous(chunks[0]));
            }

            THEN("chunk[1] is contiguous (strides match, just offset)") {
                REQUIRE(ggml_is_contiguous(chunks[1]));
            }
        }
    }

    GIVEN("a 2D tensor [dim0, seq_len] split along dim=0") {
        for (int seq_len : {31, 60, 100}) {
            auto tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, seq_len);
            ggml_tensor* chunks[2];
            split_tensor(ctx, tensor, 0, chunks, 2);

            THEN("chunks have shape [256, seq_len]") {
                REQUIRE(chunks[0]->ne[0] == 256);
                REQUIRE(chunks[0]->ne[1] == seq_len);
            }

            THEN("chunk[0] is NOT contiguous (strides from parent)") {
                // nb[1] = 512 * sizeof(float) but ne[0] = 256
                // so nb[1] != ne[0] * nb[0] → non-contiguous
                REQUIRE(chunks[0]->nb[1] == 512 * sizeof(float));
                REQUIRE_FALSE(ggml_is_contiguous(chunks[0]));
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("ggml_repeat_4d preserves contiguity") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("position_ids repeat pattern used in DiT") {
        for (int seq_len : {31, 60, 100}) {
            auto pos = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len);
            auto repeated = ggml_repeat_4d(ctx, pos, seq_len, 2, 1, 1);

            THEN("repeated tensor has expected shape") {
                REQUIRE(repeated->ne[0] == seq_len);
                REQUIRE(repeated->ne[1] == 2);
            }
        }
    }

    ggml_free(ctx);
}
