#include "artifact/reader.h"
#include "artifact/gguf_directory.h"
#include <fstream>
#include <bit>
#include <map>
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace ninfer::artifact;

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: native-gguf-test MODEL.gguf REFERENCE.ninfer [MMPROJ.gguf]\n";
        return 2;
    }
    try {
        Reader source(argv[1], argc == 4 ? argv[3] : ""), reference(argv[2]);
        if (!source.is_native_gguf()) throw std::runtime_error("source is not native GGUF");
        namespace fi        = ninfer::targets::qwen3_6::frontend_internal;
        const auto resource = [](Reader& reader, const char* name) {
            auto data = reader.payload(*reader.find(std::string("frontend/") + name)).data;
            return std::string(reinterpret_cast<const char*>(data.data()), data.size());
        };
        const auto st = resource(source, "tokenizer.json"),
                   sc = resource(source, "tokenizer_config.json"),
                   sg = resource(source, "generation_config.json"),
                   rt = resource(reference, "tokenizer.json"),
                   rc = resource(reference, "tokenizer_config.json"),
                   rg = resource(reference, "generation_config.json");
        fi::Tokenizer native({st, sc, sg}), converted({rt, rc, rg});
        for (int id = 0; id < 248320; ++id) {
            if (native.is_valid_token(id) != converted.is_valid_token(id) ||
                native.is_special_token(id) != converted.is_special_token(id) ||
                (native.is_valid_token(id) && native.decode_token_bytes(id) != converted.decode_token_bytes(id)))
                throw std::runtime_error("token decode/domain mismatch at " + std::to_string(id));
        }
        for (const std::string text :
             {"Hello, world!", "你好，世界！", "e\u0301 = é\n", "<think>test</think>",
              "<tool_call><function=record_result><parameter=value>391</parameter></function></"
              "tool_call>",
              "<|vision_start|><|image_pad|><|vision_end|>", "😀\\n\t12345"}) {
            if (native.encode(text) != converted.encode(text))
                throw std::runtime_error("tokenizer semantic mismatch");
        }
        if (native.default_stop_token_ids() != converted.default_stop_token_ids())
            throw std::runtime_error("stop token mismatch");
        fi::CompiledChatTemplate::resolve(resource(source, "chat_template.jinja"));
        const auto* ids           = source.find("proposal/token_ids");
        const auto* reference_ids = reference.find("proposal/token_ids");
        if (!reference_ids) reference_ids = reference.find("text/draft_head_token_ids");
        if (!reference_ids ||
            source.payload(*ids).data.size() != reference.payload(*reference_ids).data.size() ||
            std::memcmp(source.payload(*ids).data.data(),
                        reference.payload(*reference_ids).data.data(), object_bytes(*ids)))
            throw std::runtime_error("registered shortlist mismatch");
        if (argc == 4) {
            std::ifstream in(argv[3], std::ios::binary | std::ios::ate);
            if (!in) throw std::runtime_error("cannot read mmproj oracle fixture");
            std::vector<std::byte> physical(static_cast<std::size_t>(in.tellg()));
            in.seekg(0);
            in.read(reinterpret_cast<char*>(physical.data()), physical.size());
            const auto directory = GgufDirectoryParser(physical).parse();
            std::map<std::string, const GgufTensor*> names;
            for (const auto& tensor : directory.tensors) names[tensor.name] = &tensor;
            const auto rounded = [](float value) {
                const auto bits = std::bit_cast<std::uint32_t>(value);
                auto high       = std::uint16_t(bits >> 16);
                const auto low  = bits & 65535;
                if (low > 32768 || (low == 32768 && (high & 1))) ++high;
                return high;
            };
            const auto read_value = [&](const GgufTensor& tensor, std::uint64_t index) {
                if (tensor.type == 30) {
                    std::uint16_t value;
                    std::memcpy(&value, physical.data() + tensor.offset + index * 2, 2);
                    return value;
                }
                if (tensor.type != 0)
                    throw std::runtime_error("BF16 oracle fixture requires F32/BF16");
                float value;
                std::memcpy(&value, physical.data() + tensor.offset + index * 4, 4);
                return rounded(value);
            };
            std::map<std::string, std::string> mapping = {
                {"vision/patch_embedding_bias", "v.patch_embd.bias"},
                {"vision/position_embedding", "v.position_embd.weight"},
                {"vision/merger/fc1", "mm.0.weight"},
                {"vision/merger/fc1_bias", "mm.0.bias"},
                {"vision/merger/fc2", "mm.2.weight"},
                {"vision/merger/fc2_bias", "mm.2.bias"},
                {"vision/merger/norm/weight", "v.post_ln.weight"},
                {"vision/merger/norm/bias", "v.post_ln.bias"}};
            const std::pair<const char*, const char*> layer_mapping[] = {
                {"attention/qkv", "attn_qkv.weight"},
                {"attention/qkv_bias", "attn_qkv.bias"},
                {"attention/output", "attn_out.weight"},
                {"attention/output_bias", "attn_out.bias"},
                {"mlp/fc1", "ffn_up.weight"},
                {"mlp/fc1_bias", "ffn_up.bias"},
                {"mlp/fc2", "ffn_down.weight"},
                {"mlp/fc2_bias", "ffn_down.bias"},
                {"norm1/weight", "ln1.weight"},
                {"norm1/bias", "ln1.bias"},
                {"norm2/weight", "ln2.weight"},
                {"norm2/bias", "ln2.bias"}};
            for (unsigned layer = 0; layer < 27; ++layer)
                for (const auto& [dst, src] : layer_mapping)
                    mapping["vision/layers/" + std::to_string(layer) + "/" + dst] =
                        "v.blk." + std::to_string(layer) + "." + src;
            for (const auto& [dst, src] : mapping) {
                const auto* object = source.find(dst);
                if (!object) throw std::runtime_error("vision mapping missing " + dst);
                std::vector<std::byte> bytes(object_bytes(*object));
                source.read_direct(source.absolute_offset(*object), bytes);
                const auto& tensor = *names.at(src);
                for (std::size_t i = 0; i < bytes.size() / 2; ++i) {
                    std::uint16_t actual;
                    std::memcpy(&actual, bytes.data() + i * 2, 2);
                    if (actual != read_value(tensor, i))
                        throw std::runtime_error("vision weight mismatch " + dst);
                }
                // Partial byte reads protect streaming cast boundaries.
                if (bytes.size() > 7) {
                    std::array<std::byte, 7> partial;
                    source.read_direct(source.absolute_offset(*object) + 1, partial);
                    if (std::memcmp(partial.data(), bytes.data() + 1, 7))
                        throw std::runtime_error("unaligned vision read mismatch");
                }
            }
            const auto* patch = source.find("vision/patch_embedding");
            std::vector<std::uint16_t> actual(object_bytes(*patch) / 2);
            source.read_direct(source.absolute_offset(*patch),
                               {reinterpret_cast<std::byte*>(actual.data()), actual.size() * 2});
            // Independent tensor stack oracle: [out,channel,time,y,x].
            for (std::size_t i = 0; i < actual.size(); ++i) {
                auto index   = i;
                const auto x = index % 16;
                index /= 16;
                const auto y = index % 16;
                index /= 16;
                const auto time = index % 2;
                index /= 2;
                const auto channel = index % 3;
                const auto out     = index / 3;
                const auto& tensor =
                    *names.at(time ? "v.patch_embd.weight.1" : "v.patch_embd.weight");
                if (actual[i] != read_value(tensor, ((out * 3 + channel) * 16 + y) * 16 + x))
                    throw std::runtime_error("temporal patch stacking mismatch");
            }
            std::cout << "PASS vision: " << mapping.size() + 1
                      << " tensors, exact BF16 casts/temporal patch stack and unaligned reads\n";
        }
        std::vector<std::byte> buffer(1024 * 1024);
        std::uint64_t checked = 0;
        std::size_t count     = 0;
        for (const auto& object : source.objects()) {
            const auto name = object_name(object);
            if (name.starts_with("frontend/") || name.starts_with("vision/")) {
                ++count;
                continue;
            }
            const auto* expected = reference.find(name);
            if (name == "proposal/head") {
                const auto* full      = reference.find("text/output_head");
                const auto full_bytes = reference.payload(*full).data;
                const auto row_bytes  = full_bytes.size() / 248320;
                const auto ids_bytes  = source.payload(*ids).data;
                std::vector<std::byte> row(row_bytes);
                for (std::size_t i = 0; i < 131072; ++i) {
                    std::uint32_t token;
                    std::memcpy(&token, ids_bytes.data() + i * 4, 4);
                    if (source.read_direct(source.absolute_offset(object) + i * row_bytes, row) !=
                            row_bytes ||
                        std::memcmp(row.data(), full_bytes.data() + token * row_bytes, row_bytes))
                        throw std::runtime_error("shortlist head row mismatch");
                }
                ++count;
                checked += object_bytes(object);
                continue;
            }
            if (name == "proposal/token_ids") {
                ++count;
                checked += object_bytes(object);
                continue;
            }
            if (name == "text/gdn_input_columns")
                expected = reference.find("text/layers/0/gdn/output/input_columns");
            if (!expected) throw std::runtime_error("reference missing " + std::string(name));
            auto bytes = reference.payload(*expected).data;
            std::vector<std::byte> gathered;
            const auto* source_tensor    = std::get_if<TensorDescriptor>(&object);
            const auto* reference_tensor = std::get_if<TensorDescriptor>(expected);
            if (source_tensor && reference_tensor &&
                source_tensor->format != reference_tensor->format)
                throw std::runtime_error("quantization format mismatch " + std::string(name));
            if (name.ends_with("attention/query") && reference_tensor &&
                (reference_tensor->shape[0] == 6144 || reference_tensor->shape[0] == 7168)) {
                const auto gate_name =
                    std::string(name).substr(0, std::string(name).size() - 5) + "gate";
                const auto* gate = reference.find(gate_name);
                if (!gate) throw std::runtime_error("reference missing gate");
                const auto gate_bytes = reference.payload(*gate).data;
                if (reference_tensor->shape[0] == 7168)
                    bytes = bytes.first(bytes.size() / 7168 * 6144);
                gathered.insert(gathered.end(), bytes.begin(), bytes.end());
                const auto* gate_tensor = std::get_if<TensorDescriptor>(gate);
                const auto gate_count   = gate_tensor && gate_tensor->shape[0] == 7168
                                              ? gate_bytes.size() / 7168 * 6144
                                              : gate_bytes.size();
                gathered.insert(gathered.end(), gate_bytes.begin(),
                                gate_bytes.begin() + gate_count);
                bytes = gathered;
            }
            if (reference_tensor && reference_tensor->shape[0] == 7168 &&
                (name.ends_with("attention/key") || name.ends_with("attention/value")))
                bytes = bytes.last(bytes.size() / 7168 * 1024);
            if (reference_tensor && reference_tensor->shape.size() == 2 &&
                reference_tensor->shape[0] == 14336 &&
                std::string(name).find("attention/") != std::string::npos) {
                const auto row_bytes = bytes.size() / 14336;
                if (name.ends_with("attention/query")) {
                    gathered.resize(12288 * row_bytes);
                    std::memcpy(gathered.data(), bytes.data(), 6144 * row_bytes);
                    std::memcpy(gathered.data() + 6144 * row_bytes, bytes.data() + 7168 * row_bytes,
                                6144 * row_bytes);
                    bytes = gathered;
                } else if (name.ends_with("attention/key"))
                    bytes = bytes.subspan(6144 * row_bytes, 1024 * row_bytes);
                else if (name.ends_with("attention/value"))
                    bytes = bytes.subspan(13312 * row_bytes, 1024 * row_bytes);
            }
            if (name.ends_with("gdn/query") && reference_tensor &&
                reference_tensor->shape[0] == 16384)
                bytes = bytes.first(static_cast<std::size_t>(object_bytes(object)));
            else if (name.ends_with("gdn/z") && reference_tensor &&
                     reference_tensor->shape[0] == 16384)
                bytes = bytes.last(static_cast<std::size_t>(object_bytes(object)));
            if (name.ends_with("mlp/gate") && reference_tensor &&
                reference_tensor->shape[0] == 34816)
                bytes = bytes.first(static_cast<std::size_t>(object_bytes(object)));
            else if (name.ends_with("mlp/up") && reference_tensor &&
                     reference_tensor->shape[0] == 34816)
                bytes = bytes.last(static_cast<std::size_t>(object_bytes(object)));
            if (bytes.size() != object_bytes(object))
                throw std::runtime_error(
                    "byte size mismatch " + std::string(name) +
                    " source=" + std::to_string(object_bytes(object)) +
                    " reference=" + std::to_string(bytes.size()) +
                    " rows=" + std::to_string(reference_tensor ? reference_tensor->shape[0] : 0));
            const auto* tensor = std::get_if<TensorDescriptor>(&object);
            const bool fp32    = tensor && tensor->format == NumericFormat::FP32;
            for (std::size_t at = 0; at < bytes.size(); at += buffer.size()) {
                const auto n = std::min(buffer.size(), bytes.size() - at);
                auto span    = std::span<std::byte>(buffer.data(), n);
                if (source.read_direct(source.absolute_offset(object) + at, span) != n)
                    throw std::runtime_error("short virtual read");
                if (fp32) {
                    for (std::size_t j = 0; j < n; j += 4) {
                        float a, b;
                        std::memcpy(&a, span.data() + j, 4);
                        std::memcpy(&b, bytes.data() + at + j, 4);
                        if (!std::isfinite(a) || !std::isfinite(b) ||
                            std::abs(a - b) > 1e-6F * std::max(1.0F, std::abs(b)))
                            throw std::runtime_error("FP32 transform mismatch " +
                                                     std::string(name));
                    }
                } else if (std::memcmp(span.data(), bytes.data() + at, n))
                    throw std::runtime_error("block/BF16/resource bytes mismatch " +
                                             std::string(name) + " at " + std::to_string(at));
                checked += n;
            }
            ++count;
            if (count % 50 == 0)
                std::cout << "CHECKED " << count << " objects " << checked << " bytes\n"
                          << std::flush;
        }
        std::cout << "PASS " << count << " objects " << checked
                  << " bytes; block and BF16 payloads exact, FP32 tolerance 1e-6; tokenizer and "
                     "shortlist semantics verified\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
