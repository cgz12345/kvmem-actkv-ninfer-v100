#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    if (binder.is_native_gguf()) out.patch_format = NumericFormat::BF16;
    out.patch_embedding = bind("vision/patch_embedding", out.patch_format,
                               {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target = out.layers[layer];
        if (binder.is_native_gguf()) {
            target.qkv_format = target.output_format = target.fc1_format = target.fc2_format =
                NumericFormat::BF16;
        }
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", target.qkv_format,
                                        {3 * VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * VisionBackboneConfig::hidden});
        target.output            = bind(prefix + "attention/output", target.output_format,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", target.fc1_format,
                                        {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {VisionBackboneConfig::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", target.fc2_format,
                                        {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    const auto format = binder.is_native_gguf() ? NumericFormat::BF16 : NumericFormat::W8G32_F16S;
    return VisionMergerInputPlan{
        .fc1_format = format,
        .fc1        = bind("vision/merger/fc1", format,
                           {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias   = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                           {VisionBackboneConfig::merger_hidden}),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {VisionBackboneConfig::hidden}),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm) {
    using artifact::NumericFormat;

    VisionCommonWeights out;
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, backbone.patch_format, VisionBackboneConfig::hidden,
        VisionBackboneConfig::patch_dim);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {VisionBackboneConfig::hidden, VisionBackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv      = artifact::materialized_weight(materialized, source.qkv, source.qkv_format,
                                                        3 * VisionBackboneConfig::hidden,
                                                        VisionBackboneConfig::hidden);
        target.qkv_bias = artifact::materialized_tensor(
            materialized, source.qkv_bias, NumericFormat::BF16, {3 * VisionBackboneConfig::hidden});
        target.output = artifact::materialized_weight(
            materialized, source.output, source.output_format, VisionBackboneConfig::hidden,
            VisionBackboneConfig::hidden);
        target.output_bias = artifact::materialized_tensor(
            materialized, source.output_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.fc1 = artifact::materialized_weight(materialized, source.fc1, source.fc1_format,
                                                   VisionBackboneConfig::intermediate,
                                                   VisionBackboneConfig::hidden);
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {VisionBackboneConfig::intermediate});
        target.fc2      = artifact::materialized_weight(materialized, source.fc2, source.fc2_format,
                                                        VisionBackboneConfig::hidden,
                                                        VisionBackboneConfig::intermediate);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    }

    out.merger_fc1 = artifact::materialized_weight(
        materialized, merger_input.fc1, merger_input.fc1_format,
        VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {VisionBackboneConfig::merger_hidden});
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {VisionBackboneConfig::hidden});
    return out;
}

} // namespace ninfer::targets::qwen3_6
