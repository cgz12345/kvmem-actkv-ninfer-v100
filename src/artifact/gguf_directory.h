#pragma once

#include "artifact/reader.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace ninfer::artifact {

// GGUF v2/v3 little-endian directory. The model adapter owns tensor semantics.
struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> shape; // GGML order: contiguous dimension first.
    std::uint32_t type   = 0;
    std::uint64_t offset = 0; // Absolute file offset.
    std::uint64_t bytes  = 0;
};

struct GgufDirectory {
    nlohmann::json metadata = nlohmann::json::object();
    std::vector<GgufTensor> tensors;
    std::uint64_t data_offset = 0;
};

class GgufDirectoryParser {
    std::span<const std::byte> file_;
    std::size_t pos_ = 0;

    void need(std::uint64_t n) const {
        if (n > file_.size() - pos_) throw ArtifactError("GGUF: truncated directory");
    }

    std::uint64_t word(unsigned n) {
        need(n);
        std::uint64_t v = 0;
        for (unsigned i = 0; i < n; ++i)
            v |= std::uint64_t(std::to_integer<unsigned char>(file_[pos_++])) << (8 * i);
        return v;
    }

    std::string string() {
        const auto n = word(8);
        need(n);
        std::string s(reinterpret_cast<const char*>(file_.data() + pos_),
                      static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return s;
    }

    nlohmann::json value(std::uint32_t type, bool in_array = false) {
        switch (type) {
        case 0:
            return word(1);
        case 1:
            return std::bit_cast<std::int8_t>(std::uint8_t(word(1)));
        case 2:
            return word(2);
        case 3:
            return std::bit_cast<std::int16_t>(std::uint16_t(word(2)));
        case 4:
            return word(4);
        case 5:
            return std::bit_cast<std::int32_t>(std::uint32_t(word(4)));
        case 6:
            return std::bit_cast<float>(std::uint32_t(word(4)));
        case 7: {
            const auto b = word(1);
            if (b > 1) throw ArtifactError("GGUF: invalid boolean");
            return b != 0;
        }
        case 8:
            return string();
        case 9: {
            if (in_array) throw ArtifactError("GGUF: nested metadata arrays are unsupported");
            const auto element = std::uint32_t(word(4));
            const auto count   = word(8);
            // Even the smallest scalar consumes a byte; reject before allocating.
            need(count);
            auto a = nlohmann::json::array();
            for (std::uint64_t i = 0; i < count; ++i) a.push_back(value(element, true));
            return a;
        }
        case 10:
            return word(8);
        case 11:
            return std::bit_cast<std::int64_t>(word(8));
        case 12:
            return std::bit_cast<double>(word(8));
        default:
            throw ArtifactError("GGUF: unknown metadata value type " + std::to_string(type));
        }
    }

    static std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
        if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
            throw ArtifactError("GGUF: tensor size overflow");
        return a * b;
    }

    static std::pair<unsigned, unsigned> geometry(std::uint32_t type) {
        switch (type) {
        case 0:
            return {1, 4};
        case 1:
        case 30:
            return {1, 2};
        case 8:
            return {32, 34};
        case 10:
            return {256, 84};
        case 11:
            return {256, 110};
        case 12:
            return {256, 144};
        case 13:
            return {256, 176};
        case 14:
            return {256, 210};
        case 16:
            return {256, 66};
        case 17:
            return {256, 74};
        case 18:
            return {256, 98};
        case 19:
            return {256, 50};
        case 20:
            return {32, 18};
        case 21:
            return {256, 110};
        case 22:
            return {256, 82};
        case 23:
            return {256, 136};
        case 29:
            return {256, 56};
        default:
            throw ArtifactError("GGUF: unsupported tensor quantization type " +
                                std::to_string(type));
        }
    }
public:
    explicit GgufDirectoryParser(std::span<const std::byte> file) : file_(file) {}

    GgufDirectory parse() {
        if (word(4) != 0x46554747) throw ArtifactError("GGUF: invalid magic");
        const auto version = word(4);
        if (version != 2 && version != 3)
            throw ArtifactError("GGUF: only little-endian versions 2 and 3 are supported");
        const auto tensors = word(8), keys = word(8);
        need(keys);
        need(tensors);
        GgufDirectory out;
        for (std::uint64_t i = 0; i < keys; ++i) {
            auto key = string();
            if (key.empty() || out.metadata.contains(key))
                throw ArtifactError("GGUF: empty or duplicate metadata key");
            const auto type   = std::uint32_t(word(4));
            out.metadata[key] = value(type);
        }
        std::unordered_set<std::string> names;
        for (std::uint64_t i = 0; i < tensors; ++i) {
            GgufTensor t;
            t.name = string();
            if (t.name.empty() || !names.insert(t.name).second)
                throw ArtifactError("GGUF: empty or duplicate tensor name");
            const auto rank = word(4);
            if (!rank || rank > 4) throw ArtifactError("GGUF: unsupported tensor rank");
            std::uint64_t elements = 1;
            for (unsigned j = 0; j < rank; ++j) {
                const auto d = word(8);
                if (!d) throw ArtifactError("GGUF: zero tensor dimension");
                t.shape.push_back(d);
                elements = mul(elements, d);
            }
            t.type                    = std::uint32_t(word(4));
            t.offset                  = word(8);
            const auto [block, bytes] = geometry(t.type);
            if (t.shape[0] % block) throw ArtifactError("GGUF: tensor row is not block aligned");
            t.bytes = mul(elements / block, bytes);
            out.tensors.push_back(std::move(t));
        }
        const auto alignment = out.metadata.value("general.alignment", std::uint64_t(32));
        if (!alignment || (alignment & (alignment - 1)) || alignment > std::uint64_t(1) << 20)
            throw ArtifactError("GGUF: invalid tensor alignment");
        if (pos_ > std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
            throw ArtifactError("GGUF: data offset overflow");
        out.data_offset = (pos_ + alignment - 1) & ~(alignment - 1);
        if (out.data_offset > file_.size()) throw ArtifactError("GGUF: missing tensor data");
        for (auto& t : out.tensors) {
            if (t.offset % alignment || t.offset > file_.size() - out.data_offset ||
                t.bytes > file_.size() - out.data_offset - t.offset)
                throw ArtifactError("GGUF: tensor range is outside file or unaligned");
            t.offset += out.data_offset;
        }
        auto sorted = out.tensors;
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.offset < b.offset; });
        for (std::size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i].offset < sorted[i - 1].offset + sorted[i - 1].bytes)
                throw ArtifactError("GGUF: tensor ranges overlap");
        return out;
    }
};
} // namespace ninfer::artifact
