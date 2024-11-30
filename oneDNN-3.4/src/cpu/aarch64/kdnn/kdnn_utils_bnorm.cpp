#include "kdnn.hpp"

#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace kdnn_utils {

std::pair<bool, KDNN::BatchNormalizationLayerFWD*> convert_to_kdnn_batch_normalization_fwd(
    const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_dst,
    const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
    bool use_global_stats, bool use_scale, bool use_shift, const dim_t channel_size,
    bool fuse_relu, bool is_training) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src)) {
        return {false, nullptr};
    }
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo stats = KDNN::TensorInfo({{static_cast<KDNN::SizeType>(channel_size)}, KDNN::Element::TypeT::F32, KDNN::Layout::A});
    KDNN::TensorInfo scale_shift = KDNN::TensorInfo({{static_cast<KDNN::SizeType>(channel_size)}, KDNN::Element::TypeT::F32, KDNN::Layout::A});

    if (mem_desc_stats != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_stats)) {
            return {false, nullptr};
        }
        stats = get_kdnn_tensor_info(mem_desc_stats);
    }

    if (mem_desc_scaleshift != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_scaleshift)) {
            return {false, nullptr};
        }
        scale_shift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    KDNN::NormalizationFlags flags =  KDNN::NormalizationFlags::NONE;
    flags |= use_global_stats ? KDNN::NormalizationFlags::USE_GLOBAL_STATS : KDNN::NormalizationFlags::NONE;
    flags |= use_scale ? KDNN::NormalizationFlags::USE_SCALE : KDNN::NormalizationFlags::NONE;
    flags |= use_shift ? KDNN::NormalizationFlags::USE_SHIFT : KDNN::NormalizationFlags::NONE;
    flags |= fuse_relu ? KDNN::NormalizationFlags::FUSE_NORM_RELU : KDNN::NormalizationFlags::NONE;
    KDNN::Propagation p_kind = is_training ? KDNN::Propagation::FORWARD_TRAINING : KDNN::Propagation::FORWARD_INFERENCE;
    if (KDNN::Status::SUCCESS != KDNN::BatchNormalizationLayerFWD::ValidateInput(p_kind, src,
        stats, scale_shift, dst, flags)) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::BatchNormalizationLayerFWD{p_kind, src, stats, scale_shift, dst, flags}};
    }
}

std::pair<bool, KDNN::BatchNormalizationLayerBWD*> convert_to_kdnn_batch_normalization_bwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_diff_dst, const memory_desc_wrapper& mem_desc_stats,
        const memory_desc_wrapper& mem_desc_scaleshift, const memory_desc_wrapper& mem_desc_diff_src,
        const memory_desc_wrapper& mem_desc_diff_ss, bool use_global_stats, bool use_scale,
        bool use_shift, const dim_t channel_size, bool fuse_relu) noexcept(false) {

    if (!common_tensor_checks(mem_desc_src, mem_desc_diff_dst, mem_desc_diff_src)) {
        return {false, nullptr};
    }
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KDNN::TensorInfo stats = KDNN::TensorInfo({{static_cast<KDNN::SizeType>(channel_size)}, KDNN::Element::TypeT::F32, KDNN::Layout::A});
    KDNN::TensorInfo scale_shift = KDNN::TensorInfo({{static_cast<KDNN::SizeType>(channel_size)}, KDNN::Element::TypeT::F32, KDNN::Layout::A});
    KDNN::TensorInfo diff_scale_shift = KDNN::TensorInfo({{static_cast<KDNN::SizeType>(channel_size)}, KDNN::Element::TypeT::F32, KDNN::Layout::A});

    if (mem_desc_stats != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_stats)) {
            return {false, nullptr};
        }
        stats = get_kdnn_tensor_info(mem_desc_stats);
    }
    if (mem_desc_scaleshift != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_scaleshift)) {
            return {false, nullptr};
        }
        scale_shift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    if (mem_desc_diff_ss != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_diff_ss)) {
            return {false, nullptr};
        }
        diff_scale_shift = get_kdnn_tensor_info(mem_desc_diff_ss);
    }

    KDNN::NormalizationFlags flags =  KDNN::NormalizationFlags::NONE;
    flags |= use_global_stats ? KDNN::NormalizationFlags::USE_GLOBAL_STATS : KDNN::NormalizationFlags::NONE;
    flags |= use_scale ? KDNN::NormalizationFlags::USE_SCALE : KDNN::NormalizationFlags::NONE;
    flags |= use_shift ? KDNN::NormalizationFlags::USE_SHIFT : KDNN::NormalizationFlags::NONE;
    flags |= fuse_relu ? KDNN::NormalizationFlags::FUSE_NORM_RELU : KDNN::NormalizationFlags::NONE;
    if (KDNN::Status::SUCCESS != KDNN::BatchNormalizationLayerBWD::ValidateInput(src,
        diff_dst,  stats, scale_shift, diff_src, diff_scale_shift, flags)) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::BatchNormalizationLayerBWD{src, diff_dst,  stats,
            scale_shift, diff_src, diff_scale_shift, flags}};
    }
}

} // namespace kdnn_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
