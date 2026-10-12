#include "ninfer/engine.h"
#include "artifact/reader.h"
#include "product/prompt_input/prompt_input.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "MODEL MANIFEST.json RESULTS.json [mtp|plain] [sink_tokens] [verify]\n"; return 2; }
    try {
        using Json = nlohmann::json;
        using namespace ninfer;
        EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = 262144;
        options.kv_capacity = KvCapacityPolicy::automatic(512ULL << 20);
        options.prefill_chunk = 2048;
        options.kv_cache = KvCacheStorage::Int8Group64;
        options.kvmem.selected_tokens = 24576;
        options.kvmem.reserve_tokens = 8192;
        options.kvmem.sink_tokens = argc > 5 ? std::stoul(argv[5]) : 5120;
        options.kvmem.verify_transfers = argc > 6 && std::string_view(argv[6]) == "verify";
        options.kvmem.recent_tokens = 64;
        options.kvmem.host_bytes = 12288ULL << 20;
        options.kvmem.retained_sessions = 4;
        options.enable_vision = false;
        options.context_cache.enabled = false;
        options.context_cache.host_state_slots = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
        if (argc < 5 || std::string_view(argv[4]) != "plain") {
            options.speculative.backend = SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = 3;
            options.speculative.proposal_head = ProposalHead::Optimized;
        }
        artifact::Reader reader(argv[1]);
        const auto resource = [&](const char* name) {
            auto object = reader.find(std::string("frontend/") + name);
            if (!object) throw std::runtime_error("missing tokenizer resource");
            auto bytes = reader.payload(*object).data;
            return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        const auto tj = resource("tokenizer.json"), tc = resource("tokenizer_config.json"),
                   gc = resource("generation_config.json");
        targets::qwen3_6::frontend_internal::Tokenizer tokenizer({tj, tc, gc});
        Json cases;
        std::ifstream(argv[2]) >> cases;
        Engine engine(std::move(options));
        Json report = {{"model", argv[1]}, {"speculative", argc < 5 || std::string_view(argv[4]) != "plain"},
                       {"cases", Json::array()}};
        for (const auto& item : cases) {
            const bool thinking = item.value("thinking", true);
            auto prompt_input = product::prompt_from_messages(item.at("file").get<std::string>(), thinking, false);
            prompt_input.options.preserve_thinking = true;
            const auto effort = item.value("effort", std::string("high"));
            if (thinking) prompt_input.options.reasoning_effort = effort == "low" ? ReasoningEffort::Low : ReasoningEffort::XHigh;
            RequestOptions request;
            request.execution.requested_output_tokens = item.value("max_tokens", 8192U);
            request.execution.sampling.temperature = item.value("temperature", 0.6F);
            request.execution.sampling.top_p = 0.95F;
            request.execution.sampling.top_k = 20;
            request.execution.sampling.min_p = 0.0F;
            request.execution.sampling.presence_penalty = 0.0F;
            request.execution.sampling.frequency_penalty = 0.0F;
            request.execution.sampling.seed = item.value("seed", 42ULL);
            if (item.contains("thinking_budget")) request.execution.thinking.budget = item.at("thinking_budget").get<std::uint32_t>();
            std::uint32_t header_tokens = 0;
            if (item.contains("header_file")) {
                auto header = product::prompt_from_messages(item.at("header_file").get<std::string>(), thinking, false);
                header.options.reasoning_effort = prompt_input.options.reasoning_effort;
                header_tokens = engine.count_tokens(std::move(header));
            }
            auto prepared = engine.prepare(std::move(prompt_input));
            const auto start = std::chrono::steady_clock::now();
            auto result = engine.generate(std::move(prepared), std::move(request));
            // Decode accepted IDs independently. Presentation/tool recovery is not the oracle.
            std::string raw;
            try { raw = tokenizer.decode(result.generated_token_ids); }
            catch (const std::exception& e) { raw = std::string("DECODE ERROR: ") + e.what(); }
            Json row = {{"name", item.at("name")}, {"profile", item},
                        {"prompt_tokens", result.prompt.prompt_tokens}, {"header_probe_tokens", header_tokens},
                        {"output_tokens", result.generated_token_ids.size()},
                        {"cached_tokens", result.reused_prompt_tokens},
                        {"finish_reason", static_cast<int>(result.finish_reason)},
                        {"raw", raw}, {"content", result.content}, {"reasoning", result.reasoning},
                        {"tool_call_count", result.tool_calls.size()},
                        {"fallback_reason", tool_call_parse_fallback_reason_name(result.tool_call_parse.fallback_reason)},
                        {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}};
            report["cases"].push_back(std::move(row));
            std::ofstream(argv[3]) << report.dump(2);
            std::cout << item.at("name") << " prompt=" << result.prompt.prompt_tokens
                      << " output=" << result.generated_token_ids.size()
                      << " tools=" << result.tool_calls.size() << "\n" << std::flush;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
