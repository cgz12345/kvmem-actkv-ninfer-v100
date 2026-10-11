#include "artifact/gguf_directory.h"
#include <iostream>
#include <functional>
using namespace ninfer::artifact;
using Bytes = std::vector<std::byte>;

void put(Bytes& b, std::uint64_t x, unsigned n) {
    for (unsigned i = 0; i < n; ++i) b.push_back(std::byte((x >> (8 * i)) & 255));
}

void str(Bytes& b, std::string_view s) {
    put(b, s.size(), 8);
    for (char c : s) b.push_back(std::byte(c));
}

Bytes fixture() {
    Bytes b;
    put(b, 0x46554747, 4);
    put(b, 3, 4);
    put(b, 1, 8);
    put(b, 2, 8);
    str(b, "general.architecture");
    put(b, 8, 4);
    str(b, "qwen35");
    str(b, "tokenizer.ggml.tokens");
    put(b, 9, 4);
    put(b, 8, 4);
    put(b, 2, 8);
    str(b, "a");
    str(b, "b");
    str(b, "weight");
    put(b, 2, 4);
    put(b, 256, 8);
    put(b, 2, 8);
    put(b, 21, 4);
    put(b, 0, 8);
    b.resize((b.size() + 31) / 32 * 32);
    b.resize(b.size() + 220);
    return b;
}

void check(bool condition) {
    if (!condition) throw std::runtime_error("check failed");
}

void rejected(const Bytes& b) {
    try {
        (void)GgufDirectoryParser(b).parse();
    } catch (const ArtifactError&) { return; }
    throw std::runtime_error("malformed GGUF accepted");
}

int main() {
    try {
        auto b = fixture();
        auto d = GgufDirectoryParser(b).parse();
        check(d.metadata["general.architecture"] == "qwen35");
        check(d.metadata["tokenizer.ggml.tokens"].size() == 2);
        check(d.tensors.size() == 1);
        check(d.tensors[0].bytes == 220);
        check(d.tensors[0].shape == std::vector<std::uint64_t>({256, 2}));
        // Every truncation of this minimal file must be rejected, including payload truncation.
        for (std::size_t n = 0; n < b.size(); ++n) rejected(Bytes(b.begin(), b.begin() + n));
        auto bad = b;
        bad[4]   = std::byte(1);
        rejected(bad);
        bad    = b;
        bad[0] = std::byte(0);
        rejected(bad);
        // Corrupt the GGML type and first dimension without making the file shorter.
        const std::string_view tensor_name = "weight";
        auto begin_name                    = reinterpret_cast<const std::byte*>(tensor_name.data());
        auto name = std::search(b.begin(), b.end(), begin_name, begin_name + tensor_name.size());
        check(name != b.end());
        auto start      = std::size_t(name - b.begin()) + 6;
        bad             = b;
        bad[start + 20] = std::byte(255);
        rejected(bad);
        bad            = b;
        bad[start + 4] = std::byte(1);
        rejected(bad);
        std::cout << "GGUF directory tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
