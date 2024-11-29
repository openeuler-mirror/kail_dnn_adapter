#include "kdnn.hpp"

#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace kdnn_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

std::pair<bool, KDNN::DeconvolutionLayerFWD*> convert_to_kdnn_deconv_fwd(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
    const memory_desc_wrapper& mem_desc_bia, const deconvolution_desc_t &dd, const alg_kind_t &alg) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_wei, mem_desc_dst)) {
        return {false, nullptr};
    }
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = KDNN::TensorInfo({0}, wei.GetType(), KDNN::Layout::A);
    if (mem_desc_bia != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_bia)) {
            return {false, nullptr};
        }
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(dd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(dd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(dd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(dd.dilates, mem_desc_src.ndims() - 2);
    if (KDNN::Status::SUCCESS != KDNN::DeconvolutionLayerFWD::ValidateInput(src, wei, dst, bias,
        strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg))) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::DeconvolutionLayerFWD{src, wei, dst, bias,
            strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg)}};
    }
}

std::pair<bool, KDNN::DeconvolutionLayerBWDData*> convert_to_kdnn_deconv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const deconvolution_desc_t &dd, const alg_kind_t &alg) noexcept(false) {
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_wei, mem_desc_diff_src)) {
        return {false, nullptr};
    }
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KDNN::Shape strides = get_kdnn_shape(dd.strides, mem_desc_diff_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(dd.padding[0], mem_desc_diff_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(dd.padding[1], mem_desc_diff_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(dd.dilates, mem_desc_diff_src.ndims() - 2);
    if (KDNN::Status::SUCCESS != KDNN::DeconvolutionLayerBWDData::ValidateInput(diff_dst,
        wei, diff_src, strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg))) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::DeconvolutionLayerBWDData{diff_dst, wei, diff_src,
            strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg)}};
    }
}

std::pair<bool, KDNN::DeconvolutionLayerBWDWeights*> convert_to_kdnn_deconv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const deconvolution_desc_t &cd, const alg_kind_t &alg) noexcept(false)
{
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_src, mem_desc_diff_wei)) {
        return {false, nullptr};
    }
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KDNN::TensorInfo diff_bias = KDNN::TensorInfo({0}, diff_wei.GetType(), KDNN::Layout::A);
    if (mem_desc_diff_bia != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_diff_bia)) {
            return {false, nullptr};
        }
        diff_bias = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_src.ndims() - 2);
    if (KDNN::Status::SUCCESS != KDNN::DeconvolutionLayerBWDWeights::ValidateInput(diff_dst,
        src, diff_wei, diff_bias, strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg))) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::DeconvolutionLayerBWDWeights{diff_dst, src, diff_wei, diff_bias,
            strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg)}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
