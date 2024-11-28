#ifndef CPU_AARCH64_KDNN_UTILS_HPP
#define CPU_AARCH64_KDNN_UTILS_HPP

#include <mutex>
#include <vector>
#include <utility>

#include "kdnn.hpp"

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/resource.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace kdnn_utils {

void set_kdnn_threading() noexcept;

KDNN::ResamplingAlg get_kdnn_resampling_alg(const alg_kind_t& resampling_alg) noexcept;

KDNN::ConvolutionAlgorithm get_kdnn_conv_alg(const alg_kind_t& conv_alg) noexcept;

KDNN::DeconvolutionAlgorithm get_kdnn_deconv_alg(const alg_kind_t& deconv_alg) noexcept;

KDNN::Shape get_kdnn_shape(const dims_t& shape, std::size_t num_dims) noexcept(false);

bool common_tensor_checks(const memory_desc_wrapper& mem_desc);

template <typename ... Descs>
bool common_tensor_checks(const memory_desc_wrapper& mem_desc, Descs && ... descs) {
    return common_tensor_checks(mem_desc) && common_tensor_checks(descs...);
}

template <typename... Descs>
format_tag_t get_layout(Descs&& ... descs) noexcept {
    using namespace format_tag;
    for (auto && tag : {a, ab, ba,
        abc, acb, bac, bca, cab, cba,
        abcd, acdb, abdc, acbd, adbc, adcb,
        bacd, bcda, cdab, cdba, dcab,
        abcde, acdeb, abced, abdec, acbde,
        adecb, bacde, bcdea, cdeab, cdeba, decab, any}) {
        if (all_memory_desc_matches_one_tag(tag, std::forward<Descs>(descs)...)) {
            return tag;
        }
    }
    return format_tag::undef;
}

template <typename Desc>
typename std::remove_reference<Desc>::type get_first_non_any_format_kind(Desc&& desc) noexcept {
    if ((desc.format_kind != format_kind::any) && (desc.format_kind != format_kind::undef)) {
        return desc;
    } else {
        return dnnl_memory_desc{};
    }
}

template <typename Desc, typename... Descs>
typename std::remove_reference<Desc>::type get_first_non_any_format_kind(Desc&& desc, Descs&& ... descs) noexcept {
    if ((desc.format_kind != format_kind::any) && (desc.format_kind != format_kind::undef)) {
        return desc;
    } else {
        return get_first_non_any_format_kind(std::forward<Descs>(descs)...);
    }
}

bool is_data_type_supported_by_kdnn(const dnnl_data_type_t& dt) noexcept;

bool is_data_layout_supported_by_kdnn(const memory_desc_wrapper& mem_desc) noexcept;

KDNN::Propagation get_kdnn_prop_kind_t(const dnnl_prop_kind_t& pk) noexcept;

KDNN::Element::TypeT get_kdnn_data_t(const dnnl_data_type_t& dt) noexcept;

KDNN::ActivationFunction get_kdnn_alg(const alg_kind_t& eltwise_alg) noexcept;

KDNN::ActivationFunction get_kdnn_with_dst_alg(const alg_kind_t& eltwise_alg) noexcept;

KDNN::ReductionFunction get_kdnn_reduction_alg(const alg_kind_t& reduction_alg) noexcept;
KDNN::PoolingFunction get_kdnn_pooling_alg(const alg_kind_t& pooling_alg) noexcept;

KDNN::BinaryFunction get_kdnn_op(const alg_kind_t& binary_op) noexcept;

KDNN::Layout get_kdnn_layout(const memory_desc_wrapper& mem_desc) noexcept;
KDNN::Layout get_kdnn_layout(format_tag_t tag) noexcept;
format_tag_t get_layout_without_last_dim(format_tag_t tag) noexcept;

KDNN::TensorInfo get_kdnn_tensor_info(const memory_desc_wrapper& mem_desc) noexcept(false);

std::pair<bool, KDNN::ActivationLayerFWD*> convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false);

std::pair<bool, KDNN::ActivationLayerBWD*> convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_diff,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false);

std::pair<bool, KDNN::ActivationLayerBWD*> convert_to_kdnn_act_bwd_with_dst(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_diff,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false);

std::pair<bool, KDNN::BinaryLayer*> convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
        const memory_desc_wrapper& mem_desc_src1, const memory_desc_wrapper& mem_desc_dst,
        const alg_kind_t& binary_op) noexcept(false);

std::pair<bool, KDNN::ConvolutionLayerFWD*> convert_to_kdnn_conv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false);

std::pair<bool, KDNN::ConvolutionLayerBWDData*> convert_to_kdnn_conv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false);

std::pair<bool, KDNN::ConvolutionLayerBWDWeights*> convert_to_kdnn_conv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false);

std::pair<bool, KDNN::DeconvolutionLayerFWD*> convert_to_kdnn_deconv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const deconvolution_desc_t &dd, const alg_kind_t &alg) noexcept(false);

std::pair<bool, KDNN::DeconvolutionLayerBWDData*> convert_to_kdnn_deconv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const deconvolution_desc_t &dd, const alg_kind_t &alg) noexcept(false);

std::pair<bool, KDNN::DeconvolutionLayerBWDWeights*> convert_to_kdnn_deconv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const deconvolution_desc_t &dd,
        const alg_kind_t &alg) noexcept(false);

std::pair<bool, KDNN::SoftmaxLayerFWD*> convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, std::size_t axis_size, bool is_logsoftmax) noexcept(false);


std::pair<bool, KDNN::SoftmaxLayerBWD*> convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_dst_diff, const memory_desc_wrapper& mem_desc_src_diff,
        std::size_t axis, bool is_logsoftmax) noexcept(false) ;

std::pair<bool, KDNN::InnerProductLayerFWD*> convert_to_kdnn_ip_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia) noexcept(false);

std::pair<bool, KDNN::InnerProductLayerBWDData*> convert_to_kdnn_ip_bwd_d(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src)
        noexcept(false);

std::pair<bool, KDNN::InnerProductLayerBWDWeights*> convert_to_kdnn_ip_bwd_w(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia) noexcept(false);

std::pair<bool, KDNN::PReLULayerFWD*> convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept(false);

std::pair<bool, KDNN::PReLULayerBWD*> convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const memory_desc_wrapper& mem_desc_diff_dst, const memory_desc_wrapper& mem_desc_diff_wei) noexcept(false);


std::pair<bool, KDNN::Gemm<KDNN::GemmPack::NO_PACK>*> convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bias) noexcept(false);

std::pair<bool, KDNN::PoolingLayerFWD*> convert_to_kdnn_pooling_fwd(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_dst, const pooling_desc_t& cd) noexcept(false);

std::pair<bool, KDNN::PoolingLayerBWD*> convert_to_kdnn_pooling_bwd(const memory_desc_wrapper& mem_desc_diff_src,
    const memory_desc_wrapper& mem_desc_diff_dst, const pooling_desc_t& cd) noexcept(false);

std::pair<bool, KDNN::ReductionLayer*> convert_to_kdnn_reduction(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& reduction_alg, float p, float eps) noexcept(false);

std::pair<bool, KDNN::ReorderLayer*> convert_to_kdnn_reorder(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst) noexcept(false);

std::pair<bool, KDNN::NormalizationLayerFWD*> convert_to_kdnn_layer_normalization_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
        const memory_desc_wrapper& mem_desc_dst, bool use_global_stats, bool use_scale, bool use_shift) noexcept(false);

std::pair<bool, KDNN::NormalizationLayerBWD*> convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d,
        const memory_desc_wrapper& stat_d, const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d,
        const memory_desc_wrapper& scale_shift_d, const memory_desc_wrapper& diff_scale_shift_d, bool use_global_stats,
        bool use_scale, bool use_shift) noexcept(false);

std::pair<bool, KDNN::LocalResponseNormalizationLayerFWD*> convert_to_kdnn_lrn_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, float alpha, float beta, float k, std::size_t local_size,
        bool across_channels) noexcept(false);

std::pair<bool, KDNN::LocalResponseNormalizationLayerBWD*> convert_to_kdnn_lrn_bwd(const memory_desc_wrapper& src_d,
        const memory_desc_wrapper& diff_dst_d, const memory_desc_wrapper& diff_src_d, float alpha, float beta, float k,
        std::size_t local_size, bool across_channels) noexcept(false);

std::pair<bool, KDNN::SumLayer*> convert_to_kdnn_sum(const std::vector<memory_desc_wrapper>& mem_desc_src,
        const float *scales, const memory_desc_wrapper& mem_desc_dst) noexcept(false);

std::pair<bool, KDNN::ShuffleLayer*> convert_to_kdnn_shuffle(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, std::size_t axis, std::size_t groupSize) noexcept(false);

std::pair<bool, KDNN::BatchNormalizationLayerFWD*> convert_to_kdnn_batch_normalization_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_stats,
        const memory_desc_wrapper& mem_desc_scaleshift, bool use_global_stats, bool use_scale,
        bool use_shift, const dim_t channel_size, bool fuse_relu, bool is_training) noexcept(false);

std::pair<bool, KDNN::BatchNormalizationLayerBWD*> convert_to_kdnn_batch_normalization_bwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_diff_dst, const memory_desc_wrapper& mem_desc_stats,
        const memory_desc_wrapper& mem_desc_scaleshift, const memory_desc_wrapper& mem_desc_diff_src,
        const memory_desc_wrapper& mem_desc_diff_ss, bool use_global_stats, bool use_scale,
        bool use_shift, const dim_t channel_size, bool fuse_relu) noexcept(false);
		
std::pair<bool, KDNN::ResamplingLayerFWD*> convert_to_kdnn_resampling(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& resampling_alg) noexcept(false);

std::pair<bool, KDNN::ResamplingLayerBWD*> convert_to_kdnn_resampling_bwd(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_diff_src, const alg_kind_t& resampling_alg) noexcept(false);

std::pair<bool, KDNN::ConcatLayer*> convert_to_kdnn_concat(const std::vector<memory_desc_wrapper>& mem_desc_src,
        const int concat_dim, const memory_desc_wrapper& mem_desc_dst) noexcept(false);

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_UTILS_HPP
