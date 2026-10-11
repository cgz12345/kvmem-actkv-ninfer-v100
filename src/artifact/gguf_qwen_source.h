#pragma once
#include "artifact/gguf_directory.h"
#include <cmath>
#include <fstream>
#include <numeric>
#include <map>

namespace ninfer::artifact {
#include "artifact/gguf_qwen_shortlist.inc"

// A virtual byte address space maps native GGUF rows into the existing binder's
// logical objects. Large block matrices remain mmap-backed; only small vectors
// and control matrices are converted. No model-sized repacked file is written.
class GgufQwenSource {
    struct Segment {
        std::uint64_t offset = 0, bytes = 0, row_bytes = 0;
        const std::byte* source = nullptr;
        int scalar_type         = -1;
        std::vector<std::uint32_t> rows;
        std::vector<std::byte> owned;
    };

    std::vector<Segment> segments_;
    std::span<const std::byte> file_;
    GgufDirectory directory_;
    std::map<std::string, const GgufTensor*, std::less<>> tensors_;
    std::uint64_t size_ = 0;

    const GgufTensor& tensor(std::string_view name,
                             std::initializer_list<std::uint64_t> shape) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw ArtifactError("GGUF Qwen: missing tensor " + std::string(name));
        const auto& t = *it->second;
        auto expected = std::vector<std::uint64_t>(shape);
        std::reverse(expected.begin(), expected.end());
        if (t.shape != expected)
            throw ArtifactError("GGUF Qwen: unexpected shape for " + std::string(name));
        return t;
    }

    static NumericFormat format(std::uint32_t type) {
        switch (type) {
        case 8:
            return NumericFormat::GGUF_Q8_0;
        case 10:
            return NumericFormat::GGUF_Q2_K;
        case 11:
            return NumericFormat::GGUF_Q3_K;
        case 12:
            return NumericFormat::GGUF_Q4_K;
        case 13:
            return NumericFormat::GGUF_Q5_K;
        case 14:
            return NumericFormat::GGUF_Q6_K;
        case 16:
            return NumericFormat::GGUF_IQ2_XXS;
        case 17:
            return NumericFormat::GGUF_IQ2_XS;
        case 18:
            return NumericFormat::GGUF_IQ3_XXS;
        case 19:
            return NumericFormat::GGUF_IQ1_S;
        case 20:
            return NumericFormat::GGUF_IQ4_NL;
        case 21:
            return NumericFormat::GGUF_IQ3_S;
        case 22:
            return NumericFormat::GGUF_IQ2_S;
        case 23:
            return NumericFormat::GGUF_IQ4_XS;
        case 29:
            return NumericFormat::GGUF_IQ1_M;
        default:
            throw ArtifactError("GGUF Qwen: projection format is unsupported: " +
                                std::to_string(type));
        }
    }

    static std::uint32_t grouped(std::uint32_t h) { return (h % 3) * 16 + h / 3; }

    std::size_t add_segment(Segment s) {
        s.offset = (size_ + 4095) / 4096 * 4096;
        size_    = s.offset + s.bytes;
        segments_.push_back(std::move(s));
        return segments_.size() - 1;
    }

    void add_object(std::string name, std::vector<std::uint64_t> shape, NumericFormat f,
                    StorageLayout layout, std::size_t segment) {
        const auto& s = segments_[segment];
        objects.push_back(
            TensorDescriptor{std::move(name), std::move(shape), f, layout, s.offset, s.bytes});
    }

    void block(std::string name, std::string_view source, std::uint64_t n, std::uint64_t k,
               std::vector<std::uint32_t> rows = {}) {
        const auto& t = tensor(source, {n, k});
        const auto f  = format(t.type);
        Segment s;
        s.bytes     = t.bytes;
        s.row_bytes = t.bytes / n;
        s.source    = file_.data() + t.offset;
        s.rows      = std::move(rows);
        if (!s.rows.empty()) {
            if (s.rows.size() != n) throw ArtifactError("GGUF Qwen: row map size mismatch");
            for (auto r : s.rows)
                if (r >= n) throw ArtifactError("GGUF Qwen: row map out of range");
        }
        auto id = add_segment(std::move(s));
        add_object(std::move(name), {n, k}, f, StorageLayout::GgufBlocksV1, id);
    }

    static float half(std::uint16_t v) {
        const int exponent = (v >> 10) & 31;
        const int mantissa = v & 1023;
        if (exponent == 31) throw ArtifactError("GGUF Qwen: nonfinite half scale");
        const float x = exponent ? std::ldexp(float(1024 + mantissa), exponent - 25)
                                 : std::ldexp(float(mantissa), -24);
        return v & 32768 ? -x : x;
    }

    float scalar(const GgufTensor& t, std::uint64_t i) const {
        const auto* p = file_.data() + t.offset;
        if (t.type == 0) {
            float f;
            std::memcpy(&f, p + 4 * i, 4);
            return f;
        }
        if (t.type == 30) {
            std::uint16_t h;
            std::memcpy(&h, p + 2 * i, 2);
            return std::bit_cast<float>(std::uint32_t(h) << 16);
        }
        if (t.type == 1) {
            std::uint16_t h;
            std::memcpy(&h, p + 2 * i, 2);
            return half(h);
        }
        if (t.type == 8) {
            p += i / 32 * 34;
            std::uint16_t h;
            std::memcpy(&h, p, 2);
            const auto q = std::bit_cast<std::int8_t>(std::to_integer<std::uint8_t>(p[2 + i % 32]));
            return half(h) * float(q);
        }
        throw ArtifactError("GGUF Qwen: unsupported small tensor format");
    }

    void small(std::string name, std::string_view source,
               std::initializer_list<std::uint64_t> shape, bool fp32 = false, float subtract = 0,
               bool reorder = false, bool logarithm = false, bool convolution = false) {
        const auto& t       = tensor(source, shape);
        std::uint64_t count = 1;
        for (auto d : shape) count *= d;
        if (logarithm && t.type != 0) throw ArtifactError("GGUF Qwen: A parameter must be FP32");
        Segment s;
        s.bytes = count * (fp32 ? 4 : 2);
        s.owned.resize(static_cast<std::size_t>(s.bytes));
        const auto width = shape.size() == 2 ? *(shape.begin() + 1) : 1;
        for (std::uint64_t i = 0; i < count; ++i) {
            std::uint64_t src = i;
            if (reorder) src = std::uint64_t(grouped(std::uint32_t(i / width))) * width + i % width;
            if (convolution) {
                const auto channel = i % 10240, tap = i / 10240;
                const auto mapped =
                    channel < 4096
                        ? channel
                        : 4096 +
                              std::uint64_t(grouped(std::uint32_t((channel - 4096) / 128))) * 128 +
                              (channel - 4096) % 128;
                src = mapped * 4 + tap;
            }
            float x = scalar(t, src) - subtract;
            if (logarithm) {
                if (!(x < 0))
                    throw ArtifactError("GGUF Qwen: invalid negative exponential A parameter");
                x = std::log(-x);
            }
            if (!std::isfinite(x)) throw ArtifactError("GGUF Qwen: nonfinite small tensor");
            if (fp32)
                std::memcpy(s.owned.data() + i * 4, &x, 4);
            else {
                auto bits = std::bit_cast<std::uint32_t>(x);
                auto bf   = std::uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                std::memcpy(s.owned.data() + i * 2, &bf, 2);
            }
        }
        auto id        = add_segment(std::move(s));
        auto out_shape = std::vector<std::uint64_t>(shape);
        if (convolution) out_shape = {4, 10240};
        add_object(std::move(name), std::move(out_shape),
                   fp32 ? NumericFormat::FP32 : NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                   id);
    }
public:
    std::vector<ObjectDescriptor> objects;
    std::map<std::string, std::string, std::less<>> aliases;

    GgufQwenSource(std::span<const std::byte> file, const std::filesystem::path&)
        : file_(file), directory_(GgufDirectoryParser(file).parse()) {
        const auto& m                                          = directory_.metadata;
        const std::pair<const char*, std::uint64_t> geometry[] = {
            {"qwen35.embedding_length", 5120},    {"qwen35.feed_forward_length", 17408},
            {"qwen35.attention.head_count", 24},  {"qwen35.attention.head_count_kv", 4},
            {"qwen35.attention.key_length", 256}, {"qwen35.attention.value_length", 256},
            {"qwen35.ssm.conv_kernel", 4},        {"qwen35.ssm.state_size", 128},
            {"qwen35.ssm.group_count", 16},       {"qwen35.ssm.time_step_rank", 48},
            {"qwen35.ssm.inner_size", 6144},      {"qwen35.full_attention_interval", 4}};
        if (m.value("general.architecture", std::string()) != "qwen35")
            throw ArtifactError(
                "GGUF: native loading currently requires Qwen/Swift dense 27B qwen35 architecture");
        for (const auto& [key, expected] : geometry)
            if (!m.contains(key) || m.at(key) != expected)
                throw ArtifactError(std::string("GGUF Qwen: unsupported geometry at ") + key);
        // Geometry alone does not identify the registered model's mathematics.
        if (m.value("qwen35.rope.dimension_count", 0) != 64 ||
            m.value("qwen35.rope.freq_base", 0.0) != 1.0e7 ||
            std::abs(m.value("qwen35.attention.layer_norm_rms_epsilon", 0.0) - 1.0e-6) > 1.0e-11 ||
            m.value("qwen35.context_length", 0) != 262144 ||
            !m.contains("qwen35.rope.dimension_sections") ||
            m.at("qwen35.rope.dimension_sections") != nlohmann::json::array({11, 11, 10, 0}) ||
            m.value("qwen35.rope.scaling.type", std::string("none")) != "none")
            throw ArtifactError("GGUF Qwen: unsupported RoPE, norm or context configuration");
        auto blocks = m.value("qwen35.block_count", 0);
        if (blocks != 64 && blocks != 65)
            throw ArtifactError("GGUF Qwen: expected 64 text layers with optional MTP layer");
        for (const auto& t : directory_.tensors) tensors_.emplace(t.name, &t);
        const bool has_mtp    = tensors_.contains("blk.64.nextn.eh_proj.weight");
        const auto mtp_layers = m.value("qwen35.nextn_predict_layers", 0);
        if (mtp_layers < 0 || mtp_layers > 1 || (mtp_layers == 1 && !has_mtp) ||
            (blocks == 65 && !has_mtp) || (has_mtp && mtp_layers != 1))
            throw ArtifactError("GGUF Qwen: inconsistent or unsupported embedded MTP layer");
        block("text/token_embedding", "token_embd.weight", 248320, 5120);
        block("text/output_head", "output.weight", 248320, 5120);
        {
            const auto& head = tensor("output.weight", {248320, 5120});
            Segment selected;
            selected.row_bytes = head.bytes / 248320;
            selected.bytes     = selected.row_bytes * 131072;
            selected.source    = file_.data() + head.offset;
            selected.rows.assign(std::begin(kGgufQwenShortlist), std::end(kGgufQwenShortlist));
            auto id = add_segment(std::move(selected));
            add_object("proposal/head", {131072, 5120}, format(head.type),
                       StorageLayout::GgufBlocksV1, id);
            Segment ids;
            ids.bytes = sizeof(kGgufQwenShortlist);
            ids.owned.resize(ids.bytes);
            std::memcpy(ids.owned.data(), kGgufQwenShortlist, ids.bytes);
            id = add_segment(std::move(ids));
            add_object("proposal/token_ids", {131072}, NumericFormat::I32,
                       StorageLayout::ContiguousLeV1, id);
        }
        small("text/final_norm", "output_norm.weight", {5120}, false, 1);
        for (unsigned l = 0; l < 64; ++l) {
            auto src = "blk." + std::to_string(l) + ".";
            auto dst = "text/layers/" + std::to_string(l) + "/";
            small(dst + "input_norm", src + "attn_norm.weight", {5120}, false, 1);
            small(dst + "post_attention_norm", src + "post_attention_norm.weight", {5120}, false,
                  1);
            block(dst + "mlp/gate", src + "ffn_gate.weight", 17408, 5120);
            block(dst + "mlp/up", src + "ffn_up.weight", 17408, 5120);
            block(dst + "mlp/down", src + "ffn_down.weight", 5120, 17408);
            if (l % 4 == 3) {
                std::vector<std::uint32_t> r(12288);
                for (unsigned i = 0; i < 6144; ++i) {
                    r[i]        = (i / 256) * 512 + i % 256;
                    r[i + 6144] = (i / 256) * 512 + 256 + i % 256;
                }
                block(dst + "attention/query", src + "attn_q.weight", 12288, 5120, std::move(r));
                aliases[dst + "attention/gate"] = dst + "attention/query";
                block(dst + "attention/key", src + "attn_k.weight", 1024, 5120);
                block(dst + "attention/value", src + "attn_v.weight", 1024, 5120);
                block(dst + "attention/output", src + "attn_output.weight", 5120, 6144);
                small(dst + "attention/query_norm", src + "attn_q_norm.weight", {256}, false, 1);
                small(dst + "attention/key_norm", src + "attn_k_norm.weight", {256}, false, 1);
            } else {
                std::vector<std::uint32_t> r(10240), z(6144);
                std::iota(r.begin(), r.end(), 0);
                for (unsigned i = 0; i < 6144; ++i) {
                    z[i]        = grouped(i / 128) * 128 + i % 128;
                    r[4096 + i] = 4096 + z[i];
                }
                block(dst + "gdn/query", src + "attn_qkv.weight", 10240, 5120, std::move(r));
                block(dst + "gdn/z", src + "attn_gate.weight", 6144, 5120, std::move(z));
                block(dst + "gdn/output", src + "ssm_out.weight", 5120, 6144);
                small(dst + "gdn/a_projection", src + "ssm_alpha.weight", {48, 5120}, false, 0,
                      true);
                small(dst + "gdn/b_projection", src + "ssm_beta.weight", {48, 5120}, false, 0,
                      true);
                small(dst + "gdn/a_log", src + "ssm_a", {48}, true, 0, true, true);
                small(dst + "gdn/dt_bias", src + "ssm_dt.bias", {48}, true, 0, true);
                small(dst + "gdn/convolution", src + "ssm_conv1d.weight", {10240, 4}, false, 0,
                      false, false, true);
                small(dst + "gdn/norm", src + "ssm_norm.weight", {128});
            }
        }
        // The runtime norm kernel applies 1+weight. Store the GGUF scale minus one,
        // including MTP scales (the exporter does not add a base-layer shift here).
        if (tensors_.contains("blk.64.nextn.eh_proj.weight")) {
            const std::string src = "blk.64.", dst = "mtp/layers/0/";
            block("mtp/input_projection", src + "nextn.eh_proj.weight", 5120, 10240);
            small("mtp/embedding_norm", src + "nextn.enorm.weight", {5120}, false, 1);
            small("mtp/hidden_norm", src + "nextn.hnorm.weight", {5120}, false, 1);
            small("mtp/final_norm", src + "nextn.shared_head_norm.weight", {5120}, false, 1);
            small(dst + "input_norm", src + "attn_norm.weight", {5120}, false, 1);
            small(dst + "post_attention_norm", src + "post_attention_norm.weight", {5120}, false,
                  1);
            small(dst + "attention/query_norm", src + "attn_q_norm.weight", {256}, false, 1);
            small(dst + "attention/key_norm", src + "attn_k_norm.weight", {256}, false, 1);
            std::vector<std::uint32_t> rows(12288);
            for (unsigned i = 0; i < 6144; ++i) {
                rows[i]        = (i / 256) * 512 + i % 256;
                rows[i + 6144] = (i / 256) * 512 + 256 + i % 256;
            }
            block(dst + "attention/query", src + "attn_q.weight", 12288, 5120, std::move(rows));
            block(dst + "attention/key", src + "attn_k.weight", 1024, 5120);
            block(dst + "attention/value", src + "attn_v.weight", 1024, 5120);
            block(dst + "attention/output", src + "attn_output.weight", 5120, 6144);
            block(dst + "mlp/gate", src + "ffn_gate.weight", 17408, 5120);
            block(dst + "mlp/up", src + "ffn_up.weight", 17408, 5120);
            block(dst + "mlp/down", src + "ffn_down.weight", 5120, 17408);
        }
        Segment cols;
        cols.bytes = 6144 * 4;
        cols.owned.resize(cols.bytes);
        for (unsigned i = 0; i < 6144; ++i) {
            const unsigned head  = i / 128;
            const std::int32_t c = std::int32_t(((head % 16) * 3 + head / 16) * 128 + i % 128);
            std::memcpy(cols.owned.data() + i * 4, &c, 4);
        }
        auto id = add_segment(std::move(cols));
        add_object("text/gdn_input_columns", {6144}, NumericFormat::I32,
                   StorageLayout::ContiguousLeV1, id);
        for (unsigned l = 0; l < 64; ++l)
            if (l % 4 != 3)
                aliases["text/layers/" + std::to_string(l) + "/gdn/output/input_columns"] =
                    "text/gdn_input_columns";
        // The registered frontend consumes the GGUF token inventory and BPE rules
        // directly. Padding rows are output-only and never enter the tokenizer domain.
        if (m.value("tokenizer.ggml.model", std::string()) != "gpt2" ||
            m.value("tokenizer.ggml.pre", std::string()) != "qwen35" ||
            m.value("tokenizer.ggml.add_bos_token", true) ||
            m.value("tokenizer.ggml.eos_token_id", 0) != 248046 ||
            m.value("tokenizer.ggml.bos_token_id", 0) != 248044 ||
            m.value("tokenizer.ggml.padding_token_id", 0) != 248044)
            throw ArtifactError("GGUF Qwen: unsupported tokenizer configuration");
        const auto& tokens = m.at("tokenizer.ggml.tokens");
        const auto& types  = m.at("tokenizer.ggml.token_type");
        if (tokens.size() != 248320 || types.size() != tokens.size())
            throw ArtifactError("GGUF Qwen: inconsistent token inventory");
        nlohmann::json vocab = nlohmann::json::object();
        nlohmann::json added = nlohmann::json::array(), decoder = nlohmann::json::object();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto type = types.at(i).get<int>();
            if ((i < 248044 && type != 1) ||
                (i >= 248044 && i < 248077 && type != 3 && type != 4) || (i >= 248077 && type != 5))
                throw ArtifactError("GGUF Qwen: unsupported token domain");
            if (type == 1) {
                const auto token = tokens.at(i).get<std::string>();
                if (vocab.contains(token)) throw ArtifactError("GGUF Qwen: duplicate token");
                vocab[token] = i;
            } else if (type == 3 || type == 4) {
                nlohmann::json entry = {{"id", i},
                                        {"content", tokens.at(i)},
                                        {"single_word", false},
                                        {"lstrip", false},
                                        {"rstrip", false},
                                        {"normalized", false},
                                        {"special", i < 248058 || i >= 248070}};
                added.push_back(entry);
                entry.erase("id");
                decoder[std::to_string(i)] = std::move(entry);
            }
        }
        std::map<std::string, std::string> resources;
        resources["tokenizer.json"] = nlohmann::json{
            {"model",
             {{"type", "BPE"},
              {"vocab", std::move(vocab)},
              {"merges", m.at("tokenizer.ggml.merges")}}},
            {"added_tokens",
             std::move(added)}}.dump();
        resources["tokenizer_config.json"] =
            nlohmann::json{{"added_tokens_decoder", std::move(decoder)},
                           {"add_bos_token", false},
                           {"add_prefix_space", false},
                           {"pad_token", "<|endoftext|>"}}
                .dump();
        resources["generation_config.json"] = nlohmann::json{{"eos_token_id", {248046, 248044}},
                                                             {"bos_token_id", 248044},
                                                             {"pad_token_id", 248044}}
                                                  .dump();
        resources["chat_template.jinja"] = m.at("tokenizer.chat_template").get<std::string>();
        resources["preprocessor_config.json"] =
            R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":65536,"longest_edge":16777216}})";
        resources["video_preprocessor_config.json"] =
            R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824},"fps":2,"min_frames":4,"max_frames":768})";
        for (const auto& [name, text] : resources) {
            Segment segment;
            segment.bytes = text.size();
            segment.owned.resize(text.size());
            std::memcpy(segment.owned.data(), text.data(), text.size());
            const auto id      = add_segment(std::move(segment));
            const auto& stored = segments_[id];
            objects.push_back(ResourceDescriptor{"frontend/" + name, ResourceEncoding::RawBytesV1,
                                                 stored.offset, stored.bytes});
        }
    }

    void add_vision(std::span<const std::byte> file) {
        const auto directory = GgufDirectoryParser(file).parse();
        const auto& m        = directory.metadata;
        if (m.value("general.architecture", std::string()) != "clip" ||
            !m.value("clip.has_vision_encoder", false) ||
            m.value("clip.projector_type", std::string()) != "qwen3vl_merger" ||
            !m.value("clip.use_gelu", false))
            throw ArtifactError("GGUF mmproj: unsupported vision architecture");
        for (const auto& [key, expected] : std::initializer_list<std::pair<const char*, int>>{
                 {"clip.vision.projection_dim", 5120},
                 {"clip.vision.image_size", 768},
                 {"clip.vision.patch_size", 16},
                 {"clip.vision.embedding_length", 1152},
                 {"clip.vision.feed_forward_length", 4304},
                 {"clip.vision.block_count", 27},
                 {"clip.vision.attention.head_count", 16},
                 {"clip.vision.spatial_merge_size", 2}})
            if (!m.contains(key) || m.at(key) != expected)
                throw ArtifactError(std::string("GGUF mmproj: unsupported geometry at ") + key);
        if (std::abs(m.value("clip.vision.attention.layer_norm_epsilon", 0.0) - 1e-6) > 1e-11 ||
            m.at("clip.vision.image_mean") != nlohmann::json::array({0.5, 0.5, 0.5}) ||
            m.at("clip.vision.image_std") != nlohmann::json::array({0.5, 0.5, 0.5}) ||
            m.at("clip.vision.is_deepstack_layers") != nlohmann::json(std::vector<bool>(27, false)))
            throw ArtifactError(
                "GGUF mmproj: unsupported preprocessing or deepstack configuration");
        std::map<std::string, const GgufTensor*, std::less<>> tensors;
        for (const auto& t : directory.tensors) tensors.emplace(t.name, &t);
        const auto get = [&](const std::string& name,
                             std::vector<std::uint64_t> shape) -> const GgufTensor& {
            std::reverse(shape.begin(), shape.end());
            auto it = tensors.find(name);
            if (it == tensors.end() || it->second->shape != shape)
                throw ArtifactError("GGUF mmproj: missing or incompatible tensor " + name);
            return *it->second;
        };
        const auto load = [&](std::string name, const std::string& source,
                              std::vector<std::uint64_t> shape) {
            const auto& t = get(source, shape);
            Segment segment;
            std::uint64_t count = 1;
            for (auto d : shape) count *= d;
            segment.bytes = count * 2;
            if (t.type == 30)
                segment.source = file.data() + t.offset;
            else if (t.type == 0 || t.type == 1) {
                // Small F32 parameters and patch/position tensors are cast once at startup.
                // Large F16 projection matrices use bounded conversion reads.
                segment.source      = file.data() + t.offset;
                segment.scalar_type = t.type;
            } else
                throw ArtifactError("GGUF mmproj: requires BF16/F16/F32 tensors");
            auto id = add_segment(std::move(segment));
            add_object(std::move(name), std::move(shape), NumericFormat::BF16,
                       StorageLayout::ContiguousLeV1, id);
        };
        const auto& p0 = get("v.patch_embd.weight", {1152, 3, 16, 16});
        const auto& p1 = get("v.patch_embd.weight.1", {1152, 3, 16, 16});
        if (p0.type != 0 || p1.type != 0)
            throw ArtifactError("GGUF mmproj: patch tensors must be FP32");
        Segment patch;
        patch.bytes = 1152 * 1536 * 2;
        patch.owned.resize(patch.bytes);
        for (std::uint64_t o = 0; o < 1152; ++o)
            for (std::uint64_t c = 0; c < 3; ++c)
                for (std::uint64_t t = 0; t < 2; ++t)
                    for (std::uint64_t pixel = 0; pixel < 256; ++pixel) {
                        float value;
                        std::memcpy(&value,
                                    file.data() + (t ? p1.offset : p0.offset) +
                                        (o * 768 + c * 256 + pixel) * 4,
                                    4);
                        const auto bits = std::bit_cast<std::uint32_t>(value);
                        const auto bf   = std::uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                        const auto index = o * 1536 + (c * 2 + t) * 256 + pixel;
                        std::memcpy(patch.owned.data() + index * 2, &bf, 2);
                    }
        auto id = add_segment(std::move(patch));
        add_object("vision/patch_embedding", {1152, 1536}, NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, id);
        load("vision/patch_embedding_bias", "v.patch_embd.bias", {1152});
        load("vision/position_embedding", "v.position_embd.weight", {2304, 1152});
        for (unsigned i = 0; i < 27; ++i) {
            auto src = "v.blk." + std::to_string(i) + ".";
            auto dst = "vision/layers/" + std::to_string(i) + "/";
            load(dst + "attention/qkv", src + "attn_qkv.weight", {3456, 1152});
            load(dst + "attention/qkv_bias", src + "attn_qkv.bias", {3456});
            load(dst + "attention/output", src + "attn_out.weight", {1152, 1152});
            load(dst + "attention/output_bias", src + "attn_out.bias", {1152});
            load(dst + "mlp/fc1", src + "ffn_up.weight", {4304, 1152});
            load(dst + "mlp/fc1_bias", src + "ffn_up.bias", {4304});
            load(dst + "mlp/fc2", src + "ffn_down.weight", {1152, 4304});
            load(dst + "mlp/fc2_bias", src + "ffn_down.bias", {1152});
            for (unsigned n = 1; n <= 2; ++n) {
                load(dst + "norm" + std::to_string(n) + "/weight",
                     src + "ln" + std::to_string(n) + ".weight", {1152});
                load(dst + "norm" + std::to_string(n) + "/bias",
                     src + "ln" + std::to_string(n) + ".bias", {1152});
            }
        }
        load("vision/merger/fc1", "mm.0.weight", {4608, 4608});
        load("vision/merger/fc1_bias", "mm.0.bias", {4608});
        load("vision/merger/fc2", "mm.2.weight", {5120, 4608});
        load("vision/merger/fc2_bias", "mm.2.bias", {5120});
        load("vision/merger/norm/weight", "v.post_ln.weight", {1152});
        load("vision/merger/norm/bias", "v.post_ln.bias", {1152});
    }

    std::uint64_t size() const { return size_; }

    std::span<const std::byte> payload(const ObjectDescriptor& obj) const {
        auto offset = object_offset(obj);
        for (const auto& s : segments_)
            if (s.offset == offset) {
                if (!s.owned.empty()) return s.owned;
                if (s.rows.empty() && s.scalar_type < 0)
                    return {s.source, static_cast<std::size_t>(s.bytes)};
                throw ArtifactError(
                    "GGUF: row-mapped tensor must use bounded materialization reads");
            }
        throw ArtifactError("GGUF: unknown virtual object");
    }

    std::size_t read(std::uint64_t offset, std::span<std::byte> dest) const {
        if (offset >= size_) return 0;
        auto n = std::min<std::uint64_t>(dest.size(), size_ - offset);
        std::fill_n(dest.data(), static_cast<std::size_t>(n), std::byte{0});
        for (const auto& s : segments_) {
            auto begin = std::max(offset, s.offset), end = std::min(offset + n, s.offset + s.bytes);
            if (begin >= end) continue;
            auto rel       = begin - s.offset;
            auto* out      = dest.data() + begin - offset;
            auto remaining = end - begin;
            if (!s.owned.empty()) {
                std::memcpy(out, s.owned.data() + rel, remaining);
                continue;
            }
            if (s.scalar_type >= 0) {
                while (remaining) {
                    const auto i = rel / 2;
                    float value;
                    if (s.scalar_type == 0)
                        std::memcpy(&value, s.source + i * 4, 4);
                    else {
                        std::uint16_t h;
                        std::memcpy(&h, s.source + i * 2, 2);
                        value = half(h);
                    }
                    if (!std::isfinite(value)) throw ArtifactError("GGUF mmproj: nonfinite weight");
                    const auto bits  = std::bit_cast<std::uint32_t>(value);
                    const auto bf    = std::uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                    const auto bytes = std::min<std::uint64_t>(remaining, 2 - rel % 2);
                    std::memcpy(out, reinterpret_cast<const std::byte*>(&bf) + rel % 2, bytes);
                    rel += bytes;
                    out += bytes;
                    remaining -= bytes;
                }
                continue;
            }
            if (s.rows.empty()) {
                std::memcpy(out, s.source + rel, remaining);
                continue;
            }
            while (remaining) {
                auto row = rel / s.row_bytes, in_row = rel % s.row_bytes;
                auto bytes = std::min(remaining, s.row_bytes - in_row);
                std::memcpy(out, s.source + std::uint64_t(s.rows[row]) * s.row_bytes + in_row,
                            bytes);
                rel += bytes;
                out += bytes;
                remaining -= bytes;
            }
        }
        return static_cast<std::size_t>(n);
    }
};
} // namespace ninfer::artifact
