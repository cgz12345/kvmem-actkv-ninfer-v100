#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/token_sums.h"
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        ninfer::DeviceContext device(0);
        // Real KV-channel width and prefill chunk size; exact BF16 dyadic inputs.
        constexpr int channels = 2048, tokens = 2048;
        ninfer::DeviceArena arena(16 * 1024 * 1024);
        auto input = arena.alloc(ninfer::DType::BF16, {channels, tokens});
        auto origin = arena.alloc(ninfer::DType::I32, {1});
        auto range = arena.alloc(ninfer::DType::I32, {2});
        auto output = arena.alloc(ninfer::DType::FP32, {channels, 33});
        auto query = arena.alloc(ninfer::DType::FP32, {channels, 1});
        std::vector<std::uint16_t> rows(channels * tokens);
        for (int t = 0; t < tokens; ++t)
            for (int d = 0; d < channels; ++d) {
                float value = float((t * 7 + d * 3) % 33 - 16) / 8;
                rows[t * channels + d] = std::uint16_t(std::bit_cast<std::uint32_t>(value) >> 16);
            }
        CUDA_CHECK(cudaMemcpy(input.data, rows.data(), input.bytes(), cudaMemcpyHostToDevice));
        ninfer::DecodeGraphDefinition definition;
        definition.capture(device.stream, [&] {
            ninfer::ops::token_sums(input, origin, range, 64, output, device.stream);
            ninfer::ops::token_sums(input, origin, range, 0, query, device.stream);
        });
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(definition);
        for (std::int32_t base : {0, 63, 64, 260909}) {
            const std::int32_t bounds[]{base + 11, base + 29};
            CUDA_CHECK(cudaMemcpy(origin.data, &base, sizeof(base), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(range.data, bounds, sizeof(bounds), cudaMemcpyHostToDevice));
            executable.launch(device.stream);
            device.synchronize();
            std::vector<float> actual(output.numel()), query_actual(channels);
            CUDA_CHECK(cudaMemcpy(actual.data(), output.data, output.bytes(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(query_actual.data(), query.data, query.bytes(), cudaMemcpyDeviceToHost));
            std::vector<double> expected(output.numel()), query_expected(channels);
            for (int t = tokens - 1; t >= 0; --t) {
                const auto position = base + t;
                if (position < bounds[0] || position >= bounds[1]) continue;
                const auto bucket = position / 64 - base / 64;
                for (int d = channels - 1; d >= 0; --d) {
                    const double represented = std::bit_cast<float>(std::uint32_t(rows[t * channels + d]) << 16);
                    expected[bucket * channels + d] += represented;
                    query_expected[d] += represented;
                }
            }
            for (std::size_t i = 0; i < actual.size(); ++i)
                if (actual[i] != expected[i]) throw std::runtime_error("logical-page reduction differs from FP64 oracle");
            for (int d = 0; d < channels; ++d)
                if (query_actual[d] != query_expected[d]) throw std::runtime_error("late query reduction differs from FP64 oracle");
        }
        std::cout << "TOKEN_SUMS logical origins and graph replay PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
