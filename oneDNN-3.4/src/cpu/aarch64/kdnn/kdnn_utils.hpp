#ifndef CPU_AARCH64_KDNN_UTILS_HPP
#define CPU_AARCH64_KDNN_UTILS_HPP

#include <mutex>
#include <vector>

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

bool is_data_type_supported_by_kdnn(const dnnl_data_type_t& dt) noexcept;

bool is_data_layout_supported_by_kdnn(const memory_desc_wrapper& mem_desc) noexcept;

KDNN::Element::TypeT get_kdnn_data_t(const dnnl_data_type_t& dt) noexcept;

KDNN::ActivationFunction get_kdnn_alg(const alg_kind_t& eltwise_alg) noexcept;

KDNN::ReductionFunction get_kdnn_reduction_alg(const alg_kind_t& reduction_alg) noexcept;

KDNN::BinaryFunction get_kdnn_op(const alg_kind_t& binary_op) noexcept;

KDNN::Layout get_kdnn_layout(const memory_desc_wrapper& mem_desc) noexcept;

KDNN::TensorInfo get_kdnn_tensor_info(const memory_desc_wrapper& mem_desc) noexcept(false);

KDNN::ActivationLayerFWD* convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false);

KDNN::ActivationLayerBWD* convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_diff,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false);

KDNN::BinaryLayer* convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
        const memory_desc_wrapper& mem_desc_src1, const memory_desc_wrapper& mem_desc_dst,
        const alg_kind_t& binary_op) noexcept(false);

KDNN::ConvolutionLayerFWD* convert_to_kdnn_conv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false);

KDNN::DeconvolutionLayerFWD* convert_to_kdnn_deconv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const deconvolution_desc_t &dd, const alg_kind_t &alg) noexcept(false);

KDNN::ConvolutionLayerBWDData* convert_to_kdnn_conv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false);

KDNN::ConvolutionLayerBWDWeights* convert_to_kdnn_conv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false);

KDNN::SoftmaxLayerFWD* convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, std::size_t axis_size) noexcept(false);


KDNN::SoftmaxLayerBWD* convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_dst,
                                               const memory_desc_wrapper& mem_desc_dst_diff,
                                               const memory_desc_wrapper& mem_desc_src_diff,
                                               std::size_t axis) noexcept(false) ;

KDNN::InnerProductLayerFWD* convert_to_kdnn_ip_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia) noexcept(false);

KDNN::InnerProductLayerBWDData* convert_to_kdnn_ip_bwd_d(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src)
        noexcept(false);

KDNN::InnerProductLayerBWDWeights* convert_to_kdnn_ip_bwd_w(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia) noexcept(false);

KDNN::PReLULayerFWD* convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept(false);

KDNN::PReLULayerBWD* convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
                    const memory_desc_wrapper& mem_desc_wei,
                    const memory_desc_wrapper& mem_desc_diff_src,
                    const memory_desc_wrapper& mem_desc_diff_dst,
                    const memory_desc_wrapper& mem_desc_diff_wei) noexcept(false);


KDNN::Gemm<KDNN::GemmPack::NO_PACK>* convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bias) noexcept(false);

KDNN::Gemm<KDNN::GemmPack::NO_PACK>* convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept(false);

KDNN::ReductionLayer* convert_to_kdnn_reduction(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& reduction_alg) noexcept(false);

KDNN::NormalizationLayerFWD* convert_to_kdnn_layer_normalization_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
        const memory_desc_wrapper& mem_desc_dst, bool use_global_stats, bool use_scale, bool use_shift) noexcept(false);

KDNN::NormalizationLayerBWD* convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d,
        const memory_desc_wrapper& sc_d, const memory_desc_wrapper& diff_sc_d) noexcept(false);

KDNN::NormalizationLayerBWD* convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d) noexcept(false);

KDNN::SumLayer* convert_to_kdnn_sum(const std::vector<memory_desc_wrapper>& mem_desc_src, const float *scales,
                                  const memory_desc_wrapper& mem_desc_dst) noexcept(false);

bool may_convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept;

bool may_convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_diff,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept;

bool may_convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
        const memory_desc_wrapper& mem_desc_src1, const memory_desc_wrapper& mem_desc_dst,
        const alg_kind_t& binary_op) noexcept;

bool may_convert_to_kdnn_conv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept;

bool may_convert_to_kdnn_deconv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const deconvolution_desc_t &cd, const alg_kind_t &alg) noexcept;

bool may_convert_to_kdnn_conv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const convolution_desc_t &cd, const alg_kind_t &alg) noexcept;

bool may_convert_to_kdnn_conv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept;

bool may_convert_to_kdnn_ip_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia) noexcept;

bool may_convert_to_kdnn_ip_bwd_d(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src) noexcept;

bool may_convert_to_kdnn_ip_bwd_w(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia) noexcept;

bool may_convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept;

bool may_convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
                    const memory_desc_wrapper& mem_desc_diff_src,
                    const memory_desc_wrapper& mem_desc_wei,
                    const memory_desc_wrapper& mem_desc_diff_wei,
                    const memory_desc_wrapper& mem_desc_diff_dst) noexcept;

bool may_convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, std::size_t axis_size) noexcept;

bool may_convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_dst,
                                const memory_desc_wrapper& mem_desc_dst_diff,
                                const memory_desc_wrapper& mem_desc_src_diff,
                                std::size_t axis) noexcept;

bool may_convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bias) noexcept;

bool may_convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept;

bool may_convert_to_kdnn_reduction(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& reduction_alg) noexcept;

bool may_convert_to_kdnn_layer_normalization_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
        const memory_desc_wrapper& mem_desc_dst, bool use_global_stats, bool use_scale, bool use_shift) noexcept;

bool may_convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d,
        const memory_desc_wrapper& sc_d, const memory_desc_wrapper& diff_sc_d) noexcept;

bool may_convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d) noexcept;

bool may_convert_to_kdnn_sum(const std::vector<memory_desc_wrapper>& mem_desc_src, const float *scales,
                                  const memory_desc_wrapper& mem_desc_dst) noexcept;

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_UTILS_HPP
