#include <catch2/catch_test_macros.hpp>

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"

SCENARIO("generation config validation accepts valid configurations") {
    GIVEN("a valid generation config") {
        cosyvoice_generation_config_t config{};
        config.temperature = 1.0f;
        config.sampling.top_k = 25;
        config.sampling.top_p = 0.9f;
        config.sampling.win_size = 50;
        config.sampling.tau_r = 0.1f;
        config.min_token_text_ratio = 2.0f;
        config.max_token_text_ratio = 50.0f;

        // set_generation_config requires a model context, so we test the
        // validation logic indirectly via the public API once a context exists.
        // For now, we verify the struct layout and field access.
        THEN("fields are accessible and correctly laid out") {
            REQUIRE(config.temperature == 1.0f);
            REQUIRE(config.sampling.top_k == 25);
            REQUIRE(config.sampling.top_p == 0.9f);
            REQUIRE(config.sampling.win_size == 50);
            REQUIRE(config.sampling.tau_r == 0.1f);
            REQUIRE(config.min_token_text_ratio == 2.0f);
            REQUIRE(config.max_token_text_ratio == 50.0f);
        }
    }
}

SCENARIO("default context params are initialized with sane values") {
    GIVEN("default context params") {
        cosyvoice_context_params_t params{};
        cosyvoice_init_default_context_params(&params);

        THEN("flash attention is enabled by default") {
            REQUIRE(params.flow_use_flash_attn == true);
            REQUIRE(params.llm_use_flash_attn == true);
        }

        THEN("KV cache defaults to Q8_0 with fallback") {
            REQUIRE(params.llm_kv_cache_type == COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0);
            REQUIRE(params.llm_allow_kv_cache_fallback == true);
        }

        THEN("inference buffer policy defaults to balanced") {
            REQUIRE(params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED);
        }

        THEN("batch and sequence limits are reasonable") {
            REQUIRE(params.n_batch == 256);
            REQUIRE(params.n_max_seq == 2048);
        }

        THEN("seed is non-zero (randomized)") {
            REQUIRE(params.seed != 0);
        }

        THEN("sampler RNG policy defaults to reset per session") {
            REQUIRE(params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION);
        }

        THEN("no custom sampler by default") {
            REQUIRE(params.sampler == nullptr);
            REQUIRE(params.sampler_ctx == nullptr);
        }
    }
}

SCENARIO("context params enum values are distinct") {
    THEN("KV cache types have distinct values") {
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_F32 != COSYVOICE_LLM_KV_CACHE_TYPE_F16);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_F16 != COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0 != COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1 != COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0 != COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1 != COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0);
    }

    THEN("inference buffer policies have distinct values") {
        REQUIRE(COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED != COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED);
        REQUIRE(COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED);
    }

    THEN("inference modes have distinct values") {
        REQUIRE(COSYVOICE_INFERENCE_MODE_ZERO_SHOT != COSYVOICE_INFERENCE_MODE_INSTRUCT);
        REQUIRE(COSYVOICE_INFERENCE_MODE_INSTRUCT != COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL);
    }

    THEN("sampler RNG policies have distinct values") {
        REQUIRE(COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION !=
                COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS);
    }
}
