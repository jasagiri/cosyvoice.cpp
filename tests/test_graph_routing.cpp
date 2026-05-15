// Graph routing tests for set_graph_backend (HiFT CUSTOM-op routing)
//
// The routing function is static in cosyvoice-tts.cpp, so we test a
// functionally equivalent reimplementation here. Any algorithm change
// in cosyvoice-tts.cpp must be reflected in this test.

#include <catch2/catch_test_macros.hpp>

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#include <vector>

// Reimplementation of set_graph_backend from cosyvoice-tts.cpp for testing.
// Routes CUSTOM ops (and their trailing VIEW/PERMUTE) to cpu_backend,
// everything else to main backend.
static void set_graph_backend(
    ggml_cgraph* gf, ggml_backend_sched_t sched,
    ggml_backend_t backend, ggml_backend_t cpu_backend, int nodes = -1)
{
    if (nodes < 0)
        nodes = ggml_graph_n_nodes(gf);

    for (int i = 0; i != nodes; ++i) {
        auto node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_CUSTOM)
        {
            do {
                ggml_backend_sched_set_tensor_backend(sched, node, cpu_backend);
                if (++i == nodes) return;
                node = ggml_graph_node(gf, i);
            } while ((node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE));

            goto set_main_backend;
        }
        else
        {
        set_main_backend:
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
        }
    }
}

namespace {

struct ggml_test_fixture {
    ggml_backend_t backend;
    ggml_backend_sched_t sched;
    ggml_context* ctx;

    ggml_test_fixture() {
        backend = ggml_backend_cpu_init();
        REQUIRE(backend != nullptr);

        sched = ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        REQUIRE(sched != nullptr);

        ggml_init_params params{};
        params.mem_size = 64 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
    }

    ~ggml_test_fixture() {
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
    }

    // Build a graph with the given op sequence.
    // Supports: GGML_OP_MUL (as a regular op), GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_CUSTOM
    ggml_cgraph* build_graph(const std::vector<ggml_op>& ops) {
        auto gf = ggml_new_graph(ctx);

        // Base tensor that ops will reference
        auto base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);

        for (auto op : ops) {
            ggml_tensor* node = nullptr;
            switch (op) {
                case GGML_OP_VIEW:
                    node = ggml_view_1d(ctx, base, 64, 0);
                    break;
                case GGML_OP_PERMUTE:
                {
                    auto t4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 4, 1);
                    node = ggml_permute(ctx, t4d, 1, 0, 2, 3);
                    break;
                }
                case GGML_OP_RESHAPE:
                    node = ggml_reshape_1d(ctx, base, 64);
                    break;
                case GGML_OP_MUL:
                    node = ggml_mul(ctx, base, base);
                    break;
                default:
                    // For CUSTOM and other ops, use mul as placeholder
                    // (we'll override the op field below)
                    node = ggml_mul(ctx, base, base);
                    node->op = op;
                    break;
            }
            ggml_build_forward_expand(gf, node);
        }

        return gf;
    }
};

} // anonymous namespace

SCENARIO("HiFT routing assigns CUSTOM ops to CPU backend") {
    ggml_test_fixture f;

    GIVEN("a graph with [MUL, CUSTOM, VIEW, MUL]") {
        auto gf = f.build_graph({GGML_OP_MUL, GGML_OP_CUSTOM, GGML_OP_VIEW, GGML_OP_MUL});

        WHEN("set_graph_backend routes the graph") {
            set_graph_backend(gf, f.sched, f.backend, f.backend);

            THEN("all nodes are assigned to backend (CPU-only case is no-op)") {
                for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                    auto assigned = ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, i));
                    REQUIRE(assigned == f.backend);
                }
            }
        }
    }
}

SCENARIO("HiFT routing with dual backends") {
    ggml_test_fixture f;
    // Create a second "backend" using another CPU instance to simulate GPU vs CPU
    auto cpu2 = ggml_backend_cpu_init();
    REQUIRE(cpu2 != nullptr);

    // Rebuild sched with 2 backends
    ggml_backend_sched_free(f.sched);
    ggml_backend_t backends[] = { f.backend, cpu2 };
    f.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    REQUIRE(f.sched != nullptr);

    GIVEN("a graph with [MUL, CUSTOM, VIEW, MUL]") {
        auto gf = f.build_graph({GGML_OP_MUL, GGML_OP_CUSTOM, GGML_OP_VIEW, GGML_OP_MUL});

        WHEN("set_graph_backend routes CUSTOM to cpu2") {
            set_graph_backend(gf, f.sched, f.backend, cpu2);

            THEN("MUL[0] goes to main backend") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 0)) == f.backend);
            }
            THEN("CUSTOM goes to CPU") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 1)) == cpu2);
            }
            THEN("VIEW after CUSTOM follows CPU") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 2)) == cpu2);
            }
            THEN("MUL[3] after VIEW goes back to main backend") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 3)) == f.backend);
            }
        }
    }

    GIVEN("a graph with [CUSTOM, PERMUTE, CUSTOM, MUL]") {
        // The routing algorithm only follows VIEW/PERMUTE after a CUSTOM.
        // When the next non-virtual op is another CUSTOM, the goto jumps to
        // set_main_backend — so the second CUSTOM goes to the main backend.
        // This matches HiFT's graph structure where consecutive CUSTOMs don't occur.
        auto gf = f.build_graph({GGML_OP_CUSTOM, GGML_OP_PERMUTE, GGML_OP_CUSTOM, GGML_OP_MUL});

        WHEN("set_graph_backend routes the graph") {
            set_graph_backend(gf, f.sched, f.backend, cpu2);

            THEN("first CUSTOM goes to CPU") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 0)) == cpu2);
            }
            THEN("PERMUTE after first CUSTOM follows CPU") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 1)) == cpu2);
            }
            THEN("second CUSTOM after non-virtual break goes to main (goto set_main_backend)") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 2)) == f.backend);
            }
            THEN("MUL goes to main") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 3)) == f.backend);
            }
        }
    }

    GIVEN("a graph with no CUSTOM ops [MUL, MUL, MUL]") {
        auto gf = f.build_graph({GGML_OP_MUL, GGML_OP_MUL, GGML_OP_MUL});

        WHEN("set_graph_backend routes the graph") {
            set_graph_backend(gf, f.sched, f.backend, cpu2);

            THEN("all go to main backend") {
                for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
                    REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, i)) == f.backend);
            }
        }
    }

    GIVEN("a graph ending with CUSTOM + VIEW (no trailing non-virtual op)") {
        auto gf = f.build_graph({GGML_OP_MUL, GGML_OP_CUSTOM, GGML_OP_VIEW});

        WHEN("set_graph_backend routes the graph") {
            set_graph_backend(gf, f.sched, f.backend, cpu2);

            THEN("MUL goes to main") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 0)) == f.backend);
            }
            THEN("CUSTOM goes to CPU") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 1)) == cpu2);
            }
            THEN("trailing VIEW goes to CPU") {
                REQUIRE(ggml_backend_sched_get_tensor_backend(f.sched, ggml_graph_node(gf, 2)) == cpu2);
            }
        }
    }

    ggml_backend_free(cpu2);
}
