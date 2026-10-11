#include "ninfer/ops/linear_topk.h"

#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kFullRows  = 248320;
constexpr std::int32_t kValidRows = 248077;
constexpr std::int32_t kShortRows = 131072;
constexpr std::int32_t kTopK      = 16;
constexpr int kMaxColumns         = 257;

constexpr ReductionCriterion kA16ScoreCriterion{
    /*relative_l2=*/1.0 / 256.0,
    /*gross_absolute=*/1.0 / 256.0,
    /*gross_relative_to_max_reference=*/2.0 / 256.0,
};

struct FixtureWeight {
    DeviceBuffer payload;
    Weight weight;
    std::uint64_t scale_offset = 0;
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

FixtureWeight make_rowsplit(QType qtype, std::int32_t rows) {
    const std::int32_t group         = qtype == QType::W8G32_F16S ? 32 : 64;
    const std::uint64_t groups       = static_cast<std::uint64_t>(rows) * (kHidden / group);
    const std::uint64_t code_bytes   = qtype == QType::W8G32_F16S
                                           ? static_cast<std::uint64_t>(rows) * kHidden
                                           : static_cast<std::uint64_t>(rows) * kHidden / 2;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = groups * sizeof(std::uint16_t);
    FixtureWeight result{
        DeviceBuffer(static_cast<std::size_t>(scale_offset + scale_bytes)), {}, scale_offset};
    result.payload.fill(0);
    result.weight.payload          = result.payload.p;
    result.weight.payload_bytes    = result.payload.bytes;
    result.weight.high_plane_bytes = 0;
    result.weight.qtype            = qtype;
    result.weight.group_size       = group;
    result.weight.shape[0]         = rows;
    result.weight.shape[1]         = kHidden;
    result.weight.padded_shape[0]  = rows;
    result.weight.padded_shape[1]  = kHidden;
    result.weight.ndim             = 2;
    result.weight.qdata            = result.payload.p;
    result.weight.qhigh            = nullptr;
    result.weight.scales           = static_cast<std::uint8_t*>(result.payload.p) + scale_offset;
    result.weight.n                = rows;
    result.weight.k                = kHidden;
    result.weight.group            = group;
    result.weight.layout           = QuantLayout::RowSplit;
    result.weight.scale_dtype      = DType::FP16;
    return result;
}

FixtureWeight make_fp8() {
    const std::uint64_t code_bytes   = static_cast<std::uint64_t>(kFullRows) * kHidden;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = static_cast<std::uint64_t>(kFullRows) * 2;
    FixtureWeight result{
        DeviceBuffer(static_cast<std::size_t>(scale_offset + scale_bytes)), {}, scale_offset};
    result.payload.fill(0);
    result.weight.payload          = result.payload.p;
    result.weight.payload_bytes    = result.payload.bytes;
    result.weight.high_plane_bytes = 0;
    result.weight.qtype            = QType::FP8_E4M3FN_ROW_BF16S;
    result.weight.group_size       = kHidden;
    result.weight.shape[0]         = kFullRows;
    result.weight.shape[1]         = kHidden;
    result.weight.padded_shape[0]  = kFullRows;
    result.weight.padded_shape[1]  = kHidden;
    result.weight.ndim             = 2;
    result.weight.qdata            = result.payload.p;
    result.weight.qhigh            = nullptr;
    result.weight.scales           = static_cast<std::uint8_t*>(result.payload.p) + scale_offset;
    result.weight.n                = kFullRows;
    result.weight.k                = kHidden;
    result.weight.group            = kHidden;
    result.weight.layout           = QuantLayout::RowScale;
    result.weight.scale_dtype      = DType::BF16;
    result.weight.scale_ne[0]      = kFullRows;
    result.weight.scale_nb[0]      = 2;
    result.weight.scale_nb[1]      = static_cast<std::int64_t>(kFullRows) * 2;
    result.weight.scale_nb[2]      = result.weight.scale_nb[1];
    result.weight.scale_nb[3]      = result.weight.scale_nb[1];
    return result;
}

const std::array<std::int32_t, 17> kFullWinnerRows{
    3,     127,   511,   512,   1023,   4095,   8191,   15872,  16383,
    16384, 16895, 32767, 65535, 131071, 196607, 247808, 248076,
};

const std::array<std::int32_t, 17> kShortWinnerRows{
    3,     127,   511,   512,   1023,  4095,  8191,   15872,  16383,
    16384, 16895, 32767, 32768, 65535, 65536, 130560, 131071,
};

float factor_for(std::size_t index) { return index >= 15 ? 17.0F : static_cast<float>(index + 1); }

float group_base_scale(std::int32_t group) {
    return 0.0137F + static_cast<float>(group % 7) * 0.0071F;
}

std::int32_t rowsplit_code(QType qtype, std::int32_t k) {
    const int value = 1 + k % (qtype == QType::W8G32_F16S ? 113 : 7);
    return k % 5 == 0 ? -value : value;
}

std::uint8_t fp8_code(std::int32_t k) {
    constexpr std::uint8_t codes[]{0x30, 0x32, 0x34, 0x38}; // 0.5, 0.625, 0.75, 1.0
    return codes[k & 3] | (k % 5 == 0 ? 0x80 : 0);
}

void patch_rowsplit_row(FixtureWeight& fixture, QType qtype, std::int32_t row, float factor) {
    const std::size_t code_row_bytes = qtype == QType::W8G32_F16S ? kHidden : kHidden / 2;
    std::vector<std::uint8_t> codes(code_row_bytes);
    if (qtype == QType::W8G32_F16S) {
        for (std::int32_t k = 0; k < kHidden; ++k) {
            codes[static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(rowsplit_code(qtype, k));
        }
    } else {
        for (std::int32_t pair = 0; pair < kHidden / 2; ++pair) {
            const std::uint8_t low  = static_cast<std::uint8_t>(rowsplit_code(qtype, pair * 2));
            const std::uint8_t high = static_cast<std::uint8_t>(rowsplit_code(qtype, pair * 2 + 1));
            codes[static_cast<std::size_t>(pair)] =
                static_cast<std::uint8_t>((low & 15) | (high << 4));
        }
    }
    fixture.payload.copy_from_host(codes.data(), codes.size(),
                                   static_cast<std::size_t>(row) * code_row_bytes);
    const std::int32_t groups = kHidden / (qtype == QType::W8G32_F16S ? 32 : 64);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(groups));
    for (std::int32_t group = 0; group < groups; ++group) {
        scales[static_cast<std::size_t>(group)] =
            quantized_weight::detail::f32_to_f16(factor * group_base_scale(group));
    }
    fixture.payload.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t),
                                   static_cast<std::size_t>(fixture.scale_offset) +
                                       static_cast<std::size_t>(row) * groups *
                                           sizeof(std::uint16_t));
}

void patch_fp8_row(FixtureWeight& fixture, std::int32_t row, float factor) {
    std::vector<std::uint8_t> codes(kHidden);
    for (std::int32_t k = 0; k < kHidden; ++k) { codes[static_cast<std::size_t>(k)] = fp8_code(k); }
    fixture.payload.copy_from_host(codes.data(), codes.size(),
                                   static_cast<std::size_t>(row) * kHidden);
    const std::uint16_t scale = f32_to_bf16(factor);
    fixture.payload.copy_from_host(&scale, sizeof(scale),
                                   static_cast<std::size_t>(fixture.scale_offset) +
                                       static_cast<std::size_t>(row) * sizeof(scale));
}

std::vector<std::uint16_t> make_hidden() {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kHidden) * kMaxColumns);
    for (std::int32_t column = 0; column < kMaxColumns; ++column) {
        for (std::int32_t k = 0; k < kHidden; ++k) {
            const float source = 0.25F + static_cast<float>(k & 7) * 0.125F +
                                 static_cast<float>(column % 17) * 0.0625F;
            const std::uint16_t bits                               = f32_to_bf16(source);
            hidden[static_cast<std::size_t>(column) * kHidden + k] = bits;
        }
    }
    return hidden;
}

// Decode the actual represented signed code and stored scale, without staging BF16 casts.
std::vector<double> base_scores(QType qtype, const std::vector<std::uint16_t>& hidden) {
    std::vector<double> scores(18 * kMaxColumns);
    for (int factor = 1; factor <= 17; ++factor)
        for (int column = 0; column < kMaxColumns; ++column) {
            double sum = 0;
            for (int k = 0; k < kHidden; ++k) {
                double weight;
                if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
                    weight =
                        quantized_weight::detail::decode_e4m3fn(fp8_code(k)) *
                        static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(factor))));
                } else {
                    const int group = k / (qtype == QType::W8G32_F16S ? 32 : 64);
                    const auto scale =
                        quantized_weight::detail::f32_to_f16(factor * group_base_scale(group));
                    weight = static_cast<double>(rowsplit_code(qtype, k)) *
                             quantized_weight::detail::f16_to_f32(scale);
                }
                sum += weight * bf16_to_f32(hidden[static_cast<std::size_t>(column) * kHidden + k]);
            }
            scores[factor * kMaxColumns + column] = sum;
        }
    return scores;
}

template <std::size_t N>
std::vector<std::pair<float, std::int32_t>>
expected_order(const std::array<std::int32_t, N>& rows, const std::vector<std::int32_t>* id_map) {
    std::vector<std::pair<float, std::int32_t>> candidates;
    candidates.reserve(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const std::int32_t id = id_map == nullptr ? rows[index] : (*id_map)[rows[index]];
        candidates.emplace_back(factor_for(index), id);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    });
    candidates.resize(kTopK);
    return candidates;
}

int verify_invocation(const char* profile, std::int32_t columns, const Tensor& ids_tensor,
                      const Tensor& scores_tensor, const std::vector<double>& base_score,
                      const std::vector<std::pair<float, std::int32_t>>& expected) {
    const std::size_t count = static_cast<std::size_t>(kTopK) * columns;
    const auto ids          = from_device<std::int32_t>(ids_tensor.data, count);
    const auto scores       = from_device<float>(scores_tensor.data, count);
    std::vector<double> actual_scores(count);
    std::vector<double> reference_scores(count);
    int failures = 0;
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const std::size_t offset       = static_cast<std::size_t>(column) * kTopK + rank;
            const std::int32_t expected_id = expected[rank].second;
            const double expected_score =
                base_score[static_cast<int>(expected[rank].first) * kMaxColumns + column];
            if (ids[offset] != expected_id) {
                std::cerr << profile << " U=" << columns << " column=" << column << " rank=" << rank
                          << " id got=" << ids[offset] << " expected=" << expected_id << '\n';
                ++failures;
            }
            actual_scores[offset]    = static_cast<double>(scores[offset]);
            reference_scores[offset] = expected_score;
        }
    }
    failures += verify_reduction(std::string(profile) + " U=" + std::to_string(columns) + " scores",
                                 actual_scores, reference_scores, kA16ScoreCriterion);
    return failures;
}

template <class Launch>
void replay_graph_twice(Launch&& launch, cudaStream_t stream) {
    cudaGraph_t graph         = nullptr;
    cudaGraphExec_t execution = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "linear_topk begin capture");
    launch(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "linear_topk end capture");
    cuda_check(cudaGraphInstantiate(&execution, graph, 0), "linear_topk instantiate graph");
    cuda_check(cudaGraphLaunch(execution, stream), "linear_topk first graph replay");
    cuda_check(cudaGraphLaunch(execution, stream), "linear_topk second graph replay");
    cuda_synchronize(stream);
    cuda_check(cudaGraphExecDestroy(execution), "linear_topk destroy graph execution");
    cuda_check(cudaGraphDestroy(graph), "linear_topk destroy graph");
}

std::vector<int> test_columns() {
    std::vector<int> result;
    for (int n = 1; n <= 120; ++n) result.push_back(n);
    for (int n : {121, 127, 128, 129, 257}) result.push_back(n);
    return result;
}

int verify_zero_ties(const FixtureWeight& fixture, const Tensor* id_map,
                     const std::vector<std::int32_t>* host_map) {
    constexpr int columns = 3;
    DeviceBuffer hidden(static_cast<std::size_t>(kHidden) * columns * 2);
    hidden.fill(0);
    GuardedDeviceBuffer ids(kTopK * columns * 4), scores(kTopK * columns * 4);
    Tensor x(hidden.p, DType::BF16, {kHidden, columns});
    Tensor out_ids(ids.data(), DType::I32, {kTopK, columns});
    Tensor out_scores(scores.data(), DType::FP32, {kTopK, columns});
    WorkspaceArena workspace(ops::linear_topk_workspace_capacity_bytes(
        fixture.weight.qtype, fixture.weight.n, kHidden, columns, columns));
    if (id_map)
        ops::linear_topk(x, fixture.weight, *id_map, out_ids, out_scores, workspace, nullptr);
    else
        ops::linear_topk(x, fixture.weight, kValidRows, out_ids, out_scores, workspace, nullptr);
    cuda_synchronize();
    std::vector<int> expected;
    if (host_map) {
        expected = *host_map;
        std::sort(expected.begin(), expected.end());
        expected.resize(kTopK);
    } else
        for (int rank = 0; rank < kTopK; ++rank) expected.push_back(rank);
    const auto got_ids    = from_device<int>(ids.data(), kTopK * columns);
    const auto got_scores = from_device<float>(scores.data(), kTopK * columns);
    int failures          = 0;
    for (int i = 0; i < kTopK * columns; ++i)
        if (got_ids[i] != expected[i % kTopK] || got_scores[i] != 0) ++failures;
    if (failures) std::cerr << "linear_topk full-vocabulary zero tie failure\n";
    return failures + ids.verify_guards("zero-tie ids") + scores.verify_guards("zero-tie scores");
}

int run_full(QType qtype, const char* profile, const DeviceBuffer& hidden,
             const std::vector<double>& base_score) {
    FixtureWeight fixture =
        qtype == QType::W8G32_F16S ? make_rowsplit(qtype, kFullRows) : make_fp8();
    for (std::size_t index = 0; index < kFullWinnerRows.size(); ++index) {
        if (qtype == QType::W8G32_F16S) {
            patch_rowsplit_row(fixture, qtype, kFullWinnerRows[index], factor_for(index));
        } else {
            patch_fp8_row(fixture, kFullWinnerRows[index], factor_for(index));
        }
    }
    if (qtype == QType::W8G32_F16S) {
        patch_rowsplit_row(fixture, qtype, kFullRows - 1, 64.0F);
    } else {
        patch_fp8_row(fixture, kFullRows - 1, 64.0F);
    }

    const auto expected = expected_order(kFullWinnerRows, nullptr);
    const std::size_t capacity =
        ops::linear_topk_workspace_capacity_bytes(qtype, kFullRows, kHidden, 1, kMaxColumns);
    GuardedDeviceBuffer graph_scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{graph_scratch.data(), graph_scratch.bytes()});
    DeviceBuffer ids(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(std::int32_t));
    DeviceBuffer scores(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(float));
    int failures = 0;
    for (int columns : test_columns()) {
        GuardedDeviceBuffer x(static_cast<std::size_t>(kHidden) * columns * 2);
        GuardedDeviceBuffer out_ids(static_cast<std::size_t>(kTopK) * columns * 4);
        GuardedDeviceBuffer out_scores(static_cast<std::size_t>(kTopK) * columns * 4);
        cuda_check(cudaMemcpy(x.data(), hidden.p, x.bytes(), cudaMemcpyDeviceToDevice),
                   "copy input fixture");
        const auto before = from_device<std::uint16_t>(x.data(), x.bytes() / 2);
        const auto exact  = ops::linear_topk_workspace_capacity_bytes(
            fixture.weight.qtype, fixture.weight.n, kHidden, columns, columns);
        GuardedDeviceBuffer scratch(exact);
        WorkspaceArena point_workspace(DeviceSpan{scratch.data(), exact});
        Tensor hidden_tensor(x.data(), DType::BF16, {kHidden, columns});
        Tensor ids_tensor(out_ids.data(), DType::I32, {kTopK, columns});
        Tensor scores_tensor(out_scores.data(), DType::FP32, {kTopK, columns});
        out_ids.fill(0xcd);
        out_scores.fill(0xff);
        ops::linear_topk(hidden_tensor, fixture.weight, kValidRows, ids_tensor, scores_tensor,
                         point_workspace, nullptr);
        cuda_synchronize();
        failures +=
            verify_invocation(profile, columns, ids_tensor, scores_tensor, base_score, expected);
        failures += out_ids.verify_guards("ids tail");
        failures += out_scores.verify_guards("scores tail");
        failures += scratch.verify_guards("workspace tail");
        failures += x.verify_guards("hidden guards");
        if (from_device<std::uint16_t>(x.data(), x.bytes() / 2) != before) {
            std::cerr << "linear_topk modified hidden\n";
            ++failures;
        }
        if (point_workspace.used() != 0 || point_workspace.peak_used() > exact) {
            std::cerr << "linear_topk workspace scope/peak failure\n";
            ++failures;
        }
    }

    for (int graph_columns : {1,  8,  9,  16, 17, 24, 25, 32,  33,  48,  49, 64,
                              65, 80, 81, 88, 89, 96, 97, 112, 113, 120, 129}) {
        Tensor graph_hidden(hidden.p, DType::BF16, {kHidden, graph_columns});
        Tensor graph_ids(ids.p, DType::I32, {kTopK, graph_columns});
        Tensor graph_scores(scores.p, DType::FP32, {kTopK, graph_columns});
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreate(&stream), "linear_topk create graph stream");
        replay_graph_twice(
            [&](cudaStream_t captured_stream) {
                ops::linear_topk(graph_hidden, fixture.weight, kValidRows, graph_ids, graph_scores,
                                 workspace, captured_stream);
            },
            stream);
        cuda_check(cudaStreamDestroy(stream), "linear_topk destroy graph stream");
        failures += verify_invocation(profile, graph_columns, graph_ids, graph_scores, base_score,
                                      expected);
    }
    failures += graph_scratch.verify_guards("graph workspace tail");
    failures += verify_zero_ties(fixture, nullptr, nullptr);
    return failures;
}

int run_q4(const DeviceBuffer& hidden, const std::vector<double>& base_score) {
    FixtureWeight fixture = make_rowsplit(QType::Q4G64_F16S, kShortRows);
    for (std::size_t index = 0; index < kShortWinnerRows.size(); ++index) {
        patch_rowsplit_row(fixture, QType::Q4G64_F16S, kShortWinnerRows[index], factor_for(index));
    }
    std::vector<std::int32_t> host_map(kShortRows);
    for (std::int32_t row = 0; row < kShortRows; ++row) { host_map[row] = kValidRows - 1 - row; }
    DeviceBuffer map(host_map.size() * sizeof(std::int32_t));
    map.copy_from_host(host_map.data(), map.bytes);
    const auto expected = expected_order(kShortWinnerRows, &host_map);

    const std::size_t capacity = ops::linear_topk_workspace_capacity_bytes(
        QType::Q4G64_F16S, kShortRows, kHidden, 1, kMaxColumns);
    GuardedDeviceBuffer graph_scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{graph_scratch.data(), graph_scratch.bytes()});
    DeviceBuffer ids(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(std::int32_t));
    DeviceBuffer scores(static_cast<std::size_t>(kTopK) * kMaxColumns * sizeof(float));
    Tensor map_tensor(map.p, DType::I32, {kShortRows});
    int failures = 0;
    for (int columns : test_columns()) {
        GuardedDeviceBuffer x(static_cast<std::size_t>(kHidden) * columns * 2);
        GuardedDeviceBuffer out_ids(static_cast<std::size_t>(kTopK) * columns * 4);
        GuardedDeviceBuffer out_scores(static_cast<std::size_t>(kTopK) * columns * 4);
        cuda_check(cudaMemcpy(x.data(), hidden.p, x.bytes(), cudaMemcpyDeviceToDevice),
                   "copy input fixture");
        const auto before = from_device<std::uint16_t>(x.data(), x.bytes() / 2);
        const auto exact  = ops::linear_topk_workspace_capacity_bytes(
            fixture.weight.qtype, fixture.weight.n, kHidden, columns, columns);
        GuardedDeviceBuffer scratch(exact);
        WorkspaceArena point_workspace(DeviceSpan{scratch.data(), exact});
        Tensor hidden_tensor(x.data(), DType::BF16, {kHidden, columns});
        Tensor ids_tensor(out_ids.data(), DType::I32, {kTopK, columns});
        Tensor scores_tensor(out_scores.data(), DType::FP32, {kTopK, columns});
        out_ids.fill(0xcd);
        out_scores.fill(0xff);
        ops::linear_topk(hidden_tensor, fixture.weight, map_tensor, ids_tensor, scores_tensor,
                         point_workspace, nullptr);
        cuda_synchronize();
        failures += verify_invocation("q4-optimized", columns, ids_tensor, scores_tensor,
                                      base_score, expected);
        failures += out_ids.verify_guards("ids tail");
        failures += out_scores.verify_guards("scores tail");
        failures += scratch.verify_guards("workspace tail");
        failures += x.verify_guards("hidden guards");
        if (from_device<std::uint16_t>(x.data(), x.bytes() / 2) != before) {
            std::cerr << "linear_topk modified hidden\n";
            ++failures;
        }
        if (point_workspace.used() != 0 || point_workspace.peak_used() > exact) {
            std::cerr << "linear_topk workspace scope/peak failure\n";
            ++failures;
        }
    }

    for (int graph_columns : {1,  8,  9,  16, 17, 24, 25, 32,  33,  48,  49, 64,
                              65, 80, 81, 88, 89, 96, 97, 112, 113, 120, 129}) {
        Tensor graph_hidden(hidden.p, DType::BF16, {kHidden, graph_columns});
        Tensor graph_ids(ids.p, DType::I32, {kTopK, graph_columns});
        Tensor graph_scores(scores.p, DType::FP32, {kTopK, graph_columns});
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreate(&stream), "linear_topk create graph stream");
        replay_graph_twice(
            [&](cudaStream_t captured_stream) {
                ops::linear_topk(graph_hidden, fixture.weight, map_tensor, graph_ids, graph_scores,
                                 workspace, captured_stream);
            },
            stream);
        cuda_check(cudaStreamDestroy(stream), "linear_topk destroy graph stream");
        failures += verify_invocation("q4-optimized graph", graph_columns, graph_ids, graph_scores,
                                      base_score, expected);
    }
    failures += graph_scratch.verify_guards("graph workspace tail");
    failures += verify_zero_ties(fixture, &map_tensor, &host_map);
    if (from_device<std::int32_t>(map.p, host_map.size()) != host_map) {
        std::cerr << "linear_topk modified id map\n";
        ++failures;
    }
    return failures;
}


// Q4_K blocks are independently constructed/decoded here, not through the CUDA dequantizer.
int run_gguf(bool optimized) {
    const int rows = optimized ? kShortRows : kFullRows;
    constexpr int block_bytes = 144, block_elements = 256;
    const std::size_t row_bytes = kHidden / block_elements * block_bytes;
    FixtureWeight fixture{DeviceBuffer(row_bytes * rows), {}, 0};
    fixture.payload.fill(0);
    auto& w = fixture.weight;
    w.payload = w.qdata = fixture.payload.p;
    w.payload_bytes = fixture.payload.bytes;
    w.qtype = QType::GGUF_Q4_K;
    w.layout = QuantLayout::GgufBlocks;
    w.ndim = 2; w.n = rows; w.k = kHidden;
    w.shape[0] = w.padded_shape[0] = rows;
    w.shape[1] = w.padded_shape[1] = kHidden;
    std::vector<int> winner_rows(18);
    std::vector<double> reference(18);
    for (int i = 0; i < 18; ++i) {
        winner_rows[i] = i == 17 ? (optimized ? rows - 2 : kValidRows - 2) : i * 4001;
        std::vector<std::uint8_t> row(row_bytes, 0);
        const auto scale = quantized_weight::detail::f32_to_f16(float(i == 17 ? 17 : i + 1));
        for (int b = 0; b < kHidden / block_elements; ++b) {
            auto* block = row.data() + b * block_bytes;
            std::memcpy(block, &scale, 2);
            // Vary group scales and both nibble planes; dmin produces negative values at q=0.
            const auto minimum = quantized_weight::detail::f32_to_f16(float(i == 17 ? 17 : i+1) * 0.5F);
            std::memcpy(block+2,&minimum,2);
            for (int g=0;g<4;++g) {
                block[4+g]=std::uint8_t(g+1);
                block[8+g]=1;
                block[12+g]=std::uint8_t((g+5) | 0x10);
            }
            for (int j=0;j<128;++j) block[16+j]=std::uint8_t((j%16) | (((j*7+3)%16)<<4));
        }
        // Independently decode every represented scalar and accumulate its dot with x=1 in FP64.
        double sum = 0;
        for (int k = 0; k < kHidden; ++k) {
            const auto* block = row.data() + (k / block_elements) * block_bytes;
            std::uint16_t d, dmin; std::memcpy(&d, block, 2); std::memcpy(&dmin, block+2, 2);
            const int j = k % block_elements, group = j / 32;
            const auto* sc = block + 4;
            const int a = group < 4 ? sc[group] & 63 : (sc[group+4] & 15) | ((sc[group-4] >> 6) << 4);
            const int m = group < 4 ? sc[group+4] & 63 : (sc[group+4] >> 4) | ((sc[group] >> 6) << 4);
            const int q = (block[16 + (j / 64) * 32 + j % 32] >> ((group & 1) * 4)) & 15;
            sum += quantized_weight::detail::f16_to_f32(d) * a * q -
                   quantized_weight::detail::f16_to_f32(dmin) * m;
        }
        reference[i] = sum;
        fixture.payload.copy_from_host(row.data(), row.size(), winner_rows[i] * row_bytes);
    }
    // A large invalid padding row must be excluded by full-head valid_rows.
    if (!optimized) {
        std::vector<std::uint8_t> padding(row_bytes, 0);
        auto d = quantized_weight::detail::f32_to_f16(128.0F);
        for (int b=0; b<kHidden/256; ++b) {
            auto* block=padding.data()+b*144; std::memcpy(block,&d,2);
            for(int g=0;g<4;++g) block[4+g]=block[12+g]=1;
            std::fill(block+16,block+144,std::uint8_t(0x11));
        }
        fixture.payload.copy_from_host(padding.data(),padding.size(),(rows-1)*row_bytes);
    }
    std::vector<std::int32_t> host_map(rows);
    for(int r=0;r<rows;++r) host_map[r]=rows-1-r;
    DeviceBuffer map(rows*4); map.copy_from_host(host_map.data(),map.bytes);
    Tensor map_tensor(map.p,DType::I32,{rows});
    std::vector<int> expected(18); for(int i=0;i<18;++i) expected[i]=i;
    const auto global_id=[&](int i){return optimized?host_map[winner_rows[i]]:winner_rows[i];};
    std::sort(expected.begin(),expected.end(),[&](int a,int b){
        return reference[a]>reference[b] || (reference[a]==reference[b] && global_id(a)<global_id(b));
    });
    int failures=0;
    cudaStream_t stream=nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"gguf test stream");
    for(int columns : {1,7,8,9,16,33,129}) {
        std::vector<std::uint16_t> hx(kHidden*columns,f32_to_bf16(1.0F));
        DeviceBuffer hidden(hx.size()*2); hidden.copy_from_host(hx.data(),hidden.bytes);
        GuardedDeviceBuffer ids(kTopK*columns*4),scores(kTopK*columns*4);
        Tensor x(hidden.p,DType::BF16,{kHidden,columns});
        Tensor out_ids(ids.data(),DType::I32,{kTopK,columns});
        Tensor out_scores(scores.data(),DType::FP32,{kTopK,columns});
        const auto bytes=ops::linear_topk_workspace_capacity_bytes(w.qtype,rows,kHidden,columns,columns);
        GuardedDeviceBuffer scratch(bytes); WorkspaceArena workspace(DeviceSpan{scratch.data(),bytes});
        auto launch=[&](cudaStream_t stream){
            if(optimized) ops::linear_topk(x,w,map_tensor,out_ids,out_scores,workspace,stream);
            else ops::linear_topk(x,w,kValidRows,out_ids,out_scores,workspace,stream);
        };
        // Guard initialization runs on the default stream; the execution stream is
        // nonblocking, so explicitly finish initialization before launching the op.
        cuda_synchronize();
        launch(stream); cuda_synchronize(stream);
        if(columns==16) replay_graph_twice(launch,stream);
        const auto got_ids=from_device<int>(ids.data(),kTopK*columns);
        const auto got_scores=from_device<float>(scores.data(),kTopK*columns);
        std::vector<double> actual(kTopK*columns), oracle(kTopK*columns);
        for(int c=0;c<columns;++c) for(int rank=0;rank<kTopK;++rank) {
            int offset=c*kTopK+rank, i=expected[rank];
            if(got_ids[offset]!=global_id(i)) {
                std::cerr << "GGUF columns=" << columns << " rank=" << rank << " id=" << got_ids[offset] << " expected=" << global_id(i) << '\n';
                ++failures;
            }
            actual[offset]=got_scores[offset]; oracle[offset]=reference[i];
        }
        failures+=verify_reduction("gguf-q4-k topk scores",actual,oracle,kA16ScoreCriterion);
        failures+=scratch.verify_guards("gguf topk workspace");
        failures+=ids.verify_guards("gguf topk ids")+scores.verify_guards("gguf topk scores");
    }
    cuda_check(cudaStreamDestroy(stream),"gguf test stream destroy");
    failures+=verify_zero_ties(fixture,optimized?&map_tensor:nullptr,optimized?&host_map:nullptr);
    std::cout << "GGUF " << (optimized?"mapped":"full") << " failures=" << failures << '\n';
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (argc > 1 && std::string(argv[1]) == "--gguf-only") {
            return run_gguf(false) + run_gguf(true) == 0 ? 0 : 1;
        }
        const std::vector<std::uint16_t> host_hidden = make_hidden();
        DeviceBuffer hidden(host_hidden.size() * sizeof(std::uint16_t));
        hidden.copy_from_host(host_hidden.data(), hidden.bytes);

        int failures = run_gguf(false) + run_gguf(true);
        failures += run_full(QType::W8G32_F16S, "w8-full", hidden,
                             base_scores(QType::W8G32_F16S, host_hidden));
        failures += run_full(QType::FP8_E4M3FN_ROW_BF16S, "fp8-full", hidden,
                             base_scores(QType::FP8_E4M3FN_ROW_BF16S, host_hidden));
        failures += run_q4(hidden, base_scores(QType::Q4G64_F16S, host_hidden));
        std::cout << (failures == 0 ? "OK" : "FAIL") << " linear_topk\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_topk: " << error.what() << '\n';
        return 1;
    }
}
