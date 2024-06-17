#include "kdnn.hpp"

#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace kdnn_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

bool is_data_type_supported_by_kdnn(const dnnl_data_type_t& dt) noexcept {
    if (get_kdnn_data_t(dt) == KDNN::Element::TypeT::UNDEFINED) {
        return false;
    } else {
        return true;
    }
}

bool is_data_layout_supported_by_kdnn(const memory_desc_wrapper& mem_desc) noexcept {
    // It was necessary to change the order of layouts.
    // Because there is no preset layout in oneDNN and it is selected based on data, for example from strides,
    // so you must first check those that kdnn supports, then the rest.
    auto dnn_layout = mem_desc.matches_one_of_tag(dnnl_a, dnnl_ab, dnnl_ba,
        dnnl_abc, dnnl_acb, dnnl_bac, dnnl_bca, dnnl_cab, dnnl_cba,
        dnnl_abcd, dnnl_acdb, dnnl_abdc, dnnl_acbd, dnnl_adbc, dnnl_adcb,
        dnnl_bacd, dnnl_bcda, dnnl_cdab, dnnl_cdba, dnnl_dcab,
        dnnl_abcde, dnnl_acdeb, dnnl_abced, dnnl_abdec, dnnl_acbde,
        dnnl_adecb, dnnl_bacde, dnnl_bcdea, dnnl_cdeab, dnnl_cdeba, dnnl_decab);

    return dnn_layout != format_tag::undef;
}

KDNN::Element::TypeT get_kdnn_data_t(const dnnl_data_type_t& dt) noexcept {
    switch (dt) {
        case dnnl_f32: return KDNN::Element::TypeT::F32;
        case dnnl_f16: return KDNN::Element::TypeT::F16;
        case dnnl_bf16: return KDNN::Element::TypeT::BF16;
        case dnnl_s8: return KDNN::Element::TypeT::S8;
        case dnnl_u8: return KDNN::Element::TypeT::U8;
        case dnnl_s32: return KDNN::Element::TypeT::S32;
        default: return KDNN::Element::TypeT::UNDEFINED;
    }
}

KDNN::ActivationFunction get_kdnn_alg(const alg_kind_t& eltwise_alg) noexcept {
    switch (eltwise_alg) {
        case eltwise_abs : return KDNN::ActivationFunction::ABS;
        case eltwise_exp: return KDNN::ActivationFunction::EXP;
        case eltwise_linear: return KDNN::ActivationFunction::LINEAR;
        case eltwise_log : return KDNN::ActivationFunction::LOG;
        case eltwise_relu : return KDNN::ActivationFunction::RELU;
        case eltwise_round : return KDNN::ActivationFunction::ROUND;
        case eltwise_sqrt : return KDNN::ActivationFunction::SQRT;
        case eltwise_square : return KDNN::ActivationFunction::SQUARE;
        case eltwise_tanh: return KDNN::ActivationFunction::TANH;
        case eltwise_logistic: return KDNN::ActivationFunction::SIGMOID;
        default: return KDNN::ActivationFunction::UNIMPLEMENTED;
    }
}

KDNN::ConvolutionAlgorithm get_kdnn_conv_alg(const alg_kind_t& conv_alg) noexcept {
    switch (conv_alg) {
        case convolution_auto: return KDNN::ConvolutionAlgorithm::AUTO;
        case convolution_direct: return KDNN::ConvolutionAlgorithm::DIRECT;
        case convolution_winograd : return KDNN::ConvolutionAlgorithm::WINOGRAD;
        default: return KDNN::ConvolutionAlgorithm::UNIMPLEMENTED;
    }
}

KDNN::DeconvolutionAlgorithm get_kdnn_deconv_alg(const alg_kind_t& deconv_alg) noexcept {
    switch (deconv_alg) {
        case deconvolution_direct: return KDNN::DeconvolutionAlgorithm::DIRECT;
        case deconvolution_winograd: return KDNN::DeconvolutionAlgorithm::WINOGRAD;
        default: return KDNN::DeconvolutionAlgorithm::UNIMPLEMENTED;
    }
}

KDNN::ReductionFunction get_kdnn_reduction_alg(const alg_kind_t& reduction_alg) noexcept {
    switch (reduction_alg) {
        case reduction_max : return KDNN::ReductionFunction::MAX;
        case reduction_min: return KDNN::ReductionFunction::MIN;
        case reduction_sum : return KDNN::ReductionFunction::SUM;
        case reduction_mul : return KDNN::ReductionFunction::MUL;
        case reduction_mean : return KDNN::ReductionFunction::MEAN;
        default: return KDNN::ReductionFunction::UNIMPLEMENTED;
    }
}

KDNN::BinaryFunction get_kdnn_op(const alg_kind_t& binary_op) noexcept {
    switch (binary_op) {
        case binary_add : return KDNN::BinaryFunction::ADD;
        case binary_mul : return KDNN::BinaryFunction::MUL;
        case binary_max : return KDNN::BinaryFunction::MAX;
        case binary_min : return KDNN::BinaryFunction::MIN;
        case binary_div : return KDNN::BinaryFunction::DIV;
        case binary_sub : return KDNN::BinaryFunction::SUB;
        case binary_ge  : return KDNN::BinaryFunction::GE;
        case binary_gt  : return KDNN::BinaryFunction::GT;
        case binary_le  : return KDNN::BinaryFunction::LE;
        case binary_lt  : return KDNN::BinaryFunction::LT;
        case binary_eq  : return KDNN::BinaryFunction::EQ;
        case binary_ne  : return KDNN::BinaryFunction::NE;
        default         : return KDNN::BinaryFunction::UNIMPLEMENTED;
    }
}

KDNN::Layout get_kdnn_layout(const memory_desc_wrapper& mem_desc) noexcept {
    auto dnn_layout = mem_desc.matches_one_of_tag(dnnl_a, dnnl_ab, dnnl_ba,
        dnnl_abc, dnnl_acb, dnnl_bac, dnnl_bca, dnnl_cab, dnnl_cba,
        dnnl_abcd, dnnl_acdb, dnnl_abdc, dnnl_acbd, dnnl_adbc, dnnl_adcb,
        dnnl_bacd, dnnl_bcda, dnnl_cdab, dnnl_cdba, dnnl_dcab,
        dnnl_abcde, dnnl_acdeb, dnnl_abced, dnnl_abdec, dnnl_acbde,
        dnnl_adecb, dnnl_bacde, dnnl_bcdea, dnnl_cdeab, dnnl_cdeba, dnnl_decab);
    KDNN::Layout l = KDNN::Layout::UNDEFINED;
    switch (dnn_layout) {
        case(dnnl_a): {
            l = KDNN::Layout::A;
        } break;
        case(dnnl_ab): {
            l = KDNN::Layout::AB;
        } break;
        case(dnnl_ba): {
            l = KDNN::Layout::BA;
        } break;
        case(dnnl_abc): {
            l = KDNN::Layout::ABC;
        } break;
        case(dnnl_acb): {
            l = KDNN::Layout::ACB;
        } break;
        case(dnnl_bac): {
            l = KDNN::Layout::BAC;
        } break;
        case(dnnl_bca): {
            l = KDNN::Layout::BCA;
        } break;
        case(dnnl_cab): {
            l = KDNN::Layout::CAB;
        } break;
        case(dnnl_cba): {
            l = KDNN::Layout::CBA;
        } break;
        case(dnnl_abcd): {
            l = KDNN::Layout::ABCD;
        } break;
        case(dnnl_abdc): {
            l = KDNN::Layout::ABDC;
        } break;
        case(dnnl_acbd): {
            l = KDNN::Layout::ACBD;
        } break;
        case(dnnl_acdb): {
            l = KDNN::Layout::ACDB;
        } break;
        case(dnnl_adbc): {
            l = KDNN::Layout::ADBC;
        } break;
        case(dnnl_adcb): {
            l = KDNN::Layout::ADCB;
        } break;
        case(dnnl_bacd): {
            l = KDNN::Layout::BACD;
        } break;
        case(dnnl_bcda): {
            l = KDNN::Layout::BCDA;
        } break;
        case(dnnl_cdab): {
            l = KDNN::Layout::CDAB;
        } break;
        case(dnnl_cdba): {
            l = KDNN::Layout::CDBA;
        } break;
        case(dnnl_dcab): {
            l = KDNN::Layout::DCAB;
        } break;
        case(dnnl_abcde): {
            l = KDNN::Layout::ABCDE;
        } break;
        case(dnnl_abced): {
            l = KDNN::Layout::ABCED;
        } break;
        case(dnnl_abdec): {
            l = KDNN::Layout::ABDEC;
        } break;
        case(dnnl_acbde): {
            l = KDNN::Layout::ACBDE;
        } break;
        case(dnnl_acdeb): {
            l = KDNN::Layout::ACDEB;
        } break;
        case(dnnl_adecb): {
            l = KDNN::Layout::ADECB;
        } break;
        case(dnnl_bacde): {
            l = KDNN::Layout::BACDE;
        } break;
        case(dnnl_bcdea): {
            l = KDNN::Layout::BCDEA;
        } break;
        case(dnnl_cdeab): {
            l = KDNN::Layout::CDEAB;
        } break;
        case(dnnl_cdeba): {
            l = KDNN::Layout::CDEBA;
        } break;
        case(dnnl_decab): {
            l = KDNN::Layout::DECAB;
        } break;
        default: {} break;
    }
    return l;
}

KDNN::Shape get_kdnn_shape(const dims_t& shape, std::size_t num_dims) noexcept(false) {
    KDNN::Shape result;
    switch (num_dims) {
        case 1: {
            result = {static_cast<KDNN::SizeType>(shape[0])};
        } break;
        case 2: {
            result = {static_cast<KDNN::SizeType>(shape[0]),
                      static_cast<KDNN::SizeType>(shape[1])};
        } break;
        case 3: {
            result = {static_cast<KDNN::SizeType>(shape[0]),
                      static_cast<KDNN::SizeType>(shape[1]),
                      static_cast<KDNN::SizeType>(shape[2])};
        } break;
        case 4: {
            result = {static_cast<KDNN::SizeType>(shape[0]),
                      static_cast<KDNN::SizeType>(shape[1]),
                      static_cast<KDNN::SizeType>(shape[2]),
                      static_cast<KDNN::SizeType>(shape[3])};
        } break;
        case 5: {
            result = {static_cast<KDNN::SizeType>(shape[0]),
                      static_cast<KDNN::SizeType>(shape[1]),
                      static_cast<KDNN::SizeType>(shape[2]),
                      static_cast<KDNN::SizeType>(shape[3]),
                      static_cast<KDNN::SizeType>(shape[4])};
        } break;
        default: {
            result = {0};
        } break;
    }
    return result;
}

KDNN::TensorInfo get_kdnn_tensor_info(const memory_desc_wrapper& mem_desc) noexcept(false) {
    KDNN::Layout lt = get_kdnn_layout(mem_desc);
    std::size_t num_dims = mem_desc.ndims();
    KDNN::Shape dimensions = get_kdnn_shape(mem_desc.dims(), num_dims);
    KDNN::Shape strides = get_kdnn_shape(mem_desc.strides(), num_dims);
    strides = KDNN::Service::FlushStrides(dimensions, strides);
    return KDNN::TensorInfo(dimensions, get_kdnn_data_t(mem_desc.data_type()), lt, strides);
}

KDNN::ActivationLayerFWD* convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    return new KDNN::ActivationLayerFWD{get_kdnn_tensor_info(mem_desc_src), get_kdnn_tensor_info(mem_desc_dst), get_kdnn_alg(eltwise_alg), alpha, beta};
}

KDNN::ActivationLayerBWD* convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_ds,
        const memory_desc_wrapper& mem_desc_dd, const memory_desc_wrapper& mem_desc_src,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    return new KDNN::ActivationLayerBWD{get_kdnn_tensor_info(mem_desc_ds), get_kdnn_tensor_info(mem_desc_dd), 
        get_kdnn_tensor_info(mem_desc_src), get_kdnn_alg(eltwise_alg), alpha, beta};
}

bool may_convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept {
    return KDNN::Status::SUCCESS == KDNN::ActivationLayerFWD::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_alg(eltwise_alg), alpha, beta);
}

bool may_convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_ds,
        const memory_desc_wrapper& mem_desc_dd, const memory_desc_wrapper& mem_desc_src,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept {
    return KDNN::Status::SUCCESS == KDNN::ActivationLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_ds),
        get_kdnn_tensor_info(mem_desc_dd), get_kdnn_tensor_info(mem_desc_src), get_kdnn_alg(eltwise_alg), alpha, beta);
}

KDNN::BinaryLayer* convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
        const memory_desc_wrapper& mem_desc_src1, const memory_desc_wrapper& mem_desc_dst,
        const alg_kind_t& binary_op) noexcept(false) {
    KDNN::TensorInfo src0 = get_kdnn_tensor_info(mem_desc_src0);
    KDNN::TensorInfo src1 = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src1).GetDims(),
        get_kdnn_data_t(mem_desc_src1.data_type()), src0.GetLayout());
    return new KDNN::BinaryLayer{src0, src1,
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_op(binary_op)};
}

bool may_convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
        const memory_desc_wrapper& mem_desc_src1, const memory_desc_wrapper& mem_desc_dst,
        const alg_kind_t& binary_op) noexcept {
    KDNN::TensorInfo src0 = get_kdnn_tensor_info(mem_desc_src0);
    KDNN::TensorInfo src1 = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src1).GetDims(),
        get_kdnn_data_t(mem_desc_src1.data_type()), src0.GetLayout());
    return KDNN::Status::SUCCESS == KDNN::BinaryLayer::ValidateInput(src0, src1,
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_op(binary_op));
}

KDNN::ConvolutionLayerFWD* convert_to_kdnn_conv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = src;
    if (mem_desc_bia == &glob_zero_md) {
        bias = KDNN::TensorInfo({0}, wei.GetType(), KDNN::Layout::A);
    } else {
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_src.ndims() - 2);
    return new KDNN::ConvolutionLayerFWD{src, wei, dst, bias,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg)};
}

bool may_convert_to_kdnn_conv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = src;
    if (mem_desc_bia == &glob_zero_md) {
        bias = KDNN::TensorInfo({0}, wei.GetType(), KDNN::Layout::A);
    } else {
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_src.ndims() - 2);
    return KDNN::Status::SUCCESS == KDNN::ConvolutionLayerFWD::ValidateInput(src, wei, dst, bias,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg));
}

KDNN::ConvolutionLayerBWDData* convert_to_kdnn_conv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false) {
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_diff_dst.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_diff_dst.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_diff_dst.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_diff_dst.ndims() - 2);
    return new KDNN::ConvolutionLayerBWDData{diff_dst, wei, diff_src,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg)};
}

bool may_convert_to_kdnn_conv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const convolution_desc_t &cd, const alg_kind_t &alg) noexcept {
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_diff_dst.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_diff_dst.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_diff_dst.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_diff_dst.ndims() - 2);
    return KDNN::Status::SUCCESS == KDNN::ConvolutionLayerBWDData::ValidateInput(diff_dst, wei, diff_src,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg));
}

KDNN::ConvolutionLayerBWDWeights* convert_to_kdnn_conv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false)
{
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KDNN::TensorInfo diff_bias = src;
    if (mem_desc_diff_bia == &glob_zero_md) {
        diff_bias = KDNN::TensorInfo({0}, diff_wei.GetType(), KDNN::Layout::A);
    } else {
        diff_bias = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_src.ndims() - 2);
    return new KDNN::ConvolutionLayerBWDWeights{diff_dst, src, diff_wei, diff_bias,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg)};
}

bool may_convert_to_kdnn_conv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept
{
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KDNN::TensorInfo diff_bias = src;
    if (mem_desc_diff_bia == &glob_zero_md) {
        diff_bias = KDNN::TensorInfo({0}, diff_wei.GetType(), KDNN::Layout::A);
    } else {
        diff_bias = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_src.ndims() - 2);
    return KDNN::Status::SUCCESS == KDNN::ConvolutionLayerBWDWeights::ValidateInput(diff_dst, src, diff_wei, diff_bias,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg));
}

KDNN::DeconvolutionLayerFWD* convert_to_kdnn_deconv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const deconvolution_desc_t &dd, const alg_kind_t &alg)
        noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = src;
    if (mem_desc_bia == &glob_zero_md) {
        bias = KDNN::TensorInfo({0}, wei.GetType(), KDNN::Layout::A);
    } else {
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(dd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(dd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(dd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(dd.dilates, mem_desc_src.ndims() - 2);
    return new KDNN::DeconvolutionLayerFWD{src, wei, dst, bias,
        strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg)};
}

bool may_convert_to_kdnn_deconv_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const deconvolution_desc_t &dd, const alg_kind_t &alg) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = src;
    if (mem_desc_bia == &glob_zero_md) {
        bias = KDNN::TensorInfo({0}, wei.GetType(), KDNN::Layout::A);
    } else {
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    KDNN::Shape strides = get_kdnn_shape(dd.strides, mem_desc_src.ndims() - 2);
    KDNN::Shape paddingL = get_kdnn_shape(dd.padding[0], mem_desc_src.ndims() - 2);
    KDNN::Shape paddingR = get_kdnn_shape(dd.padding[1], mem_desc_src.ndims() - 2);
    KDNN::Shape dilates = get_kdnn_shape(dd.dilates, mem_desc_src.ndims() - 2);
    return KDNN::Status::SUCCESS == KDNN::DeconvolutionLayerFWD::ValidateInput(src, wei, dst, bias,
        strides, dilates, paddingL, paddingR, get_kdnn_deconv_alg(alg));
}

KDNN::InnerProductLayerFWD* convert_to_kdnn_ip_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = src;
    KDNN::Element::TypeT biasType = static_cast<KDNN::Element::TypeT>(wei.GetType()) == KDNN::Element::TypeT::S8 ?
        KDNN::Element::TypeT::S32 : static_cast<KDNN::Element::TypeT>(src.GetType());
    if (mem_desc_bia == &glob_zero_md) {
        bias = KDNN::TensorInfo({0, 0}, biasType, KDNN::Layout::AB);
    } else {
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    return new KDNN::InnerProductLayerFWD{src, wei, dst, bias};
}

bool may_convert_to_kdnn_ip_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo bias = src;
    KDNN::Element::TypeT biasType = static_cast<KDNN::Element::TypeT>(wei.GetType()) == KDNN::Element::TypeT::S8 ?
        KDNN::Element::TypeT::S32 : static_cast<KDNN::Element::TypeT>(src.GetType());
    if (mem_desc_bia == &glob_zero_md) {
        bias = KDNN::TensorInfo({0, 0}, biasType, KDNN::Layout::AB);
    } else {
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    return KDNN::Status::SUCCESS == KDNN::InnerProductLayerFWD::ValidateInput(src,
        wei, dst, bias);
}

KDNN::InnerProductLayerBWDData* convert_to_kdnn_ip_bwd_d(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src) noexcept(false) {
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    return new KDNN::InnerProductLayerBWDData{diff_dst, wei, diff_src};
}

bool may_convert_to_kdnn_ip_bwd_d(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src) noexcept {
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    return KDNN::Status::SUCCESS == KDNN::InnerProductLayerBWDData::ValidateInput(diff_dst,
        wei, diff_src);
}

KDNN::InnerProductLayerBWDWeights* convert_to_kdnn_ip_bwd_w(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia) noexcept(false) {
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KDNN::TensorInfo diff_bia = diff_dst;
    if (mem_desc_diff_bia == &glob_zero_md) {
        diff_bia = KDNN::TensorInfo({0, 0}, src.GetType(), KDNN::Layout::AB);
    } else {
        diff_bia = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    return new KDNN::InnerProductLayerBWDWeights{diff_dst, src, diff_wei, diff_bia};
}

bool may_convert_to_kdnn_ip_bwd_w(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia) noexcept {
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KDNN::TensorInfo diff_bia = diff_dst;
    if (mem_desc_diff_bia == &glob_zero_md) {
        diff_bia = KDNN::TensorInfo({0, 0}, src.GetType(), KDNN::Layout::AB);
    } else {
        diff_bia = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    return KDNN::Status::SUCCESS == KDNN::InnerProductLayerBWDWeights::ValidateInput(diff_dst, src, diff_wei, diff_bia);
}

KDNN::PReLULayerFWD* convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_wei).GetDims(),
        get_kdnn_data_t(mem_desc_wei.data_type()), src.GetLayout());
    return new KDNN::PReLULayerFWD{src, wei, get_kdnn_tensor_info(mem_desc_dst)};
}

bool may_convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo wei = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_wei).GetDims(),
        get_kdnn_data_t(mem_desc_wei.data_type()), src.GetLayout());
    return KDNN::Status::SUCCESS == KDNN::PReLULayerFWD::ValidateInput(src, wei, get_kdnn_tensor_info(mem_desc_dst));
}

KDNN::PReLULayerBWD* convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
                    const memory_desc_wrapper& mem_desc_diff_src,
                    const memory_desc_wrapper& mem_desc_wei,
                    const memory_desc_wrapper& mem_desc_diff_wei,
                    const memory_desc_wrapper& mem_desc_diff_dst) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KDNN::TensorInfo wei = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_wei).GetDims(),
        get_kdnn_data_t(mem_desc_wei.data_type()), src.GetLayout());
    KDNN::TensorInfo diff_wei = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_diff_wei).GetDims(),
        get_kdnn_data_t(mem_desc_diff_wei.data_type()), src.GetLayout());
    return new KDNN::PReLULayerBWD{src, diff_src, wei, diff_wei,
        get_kdnn_tensor_info(mem_desc_diff_dst)};
}

bool may_convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
                    const memory_desc_wrapper& mem_desc_diff_src,
                    const memory_desc_wrapper& mem_desc_wei,
                    const memory_desc_wrapper& mem_desc_diff_wei,
                    const memory_desc_wrapper& mem_desc_diff_dst) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KDNN::TensorInfo wei = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_wei).GetDims(),
        get_kdnn_data_t(mem_desc_wei.data_type()), src.GetLayout());
    KDNN::TensorInfo diff_wei = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_diff_wei).GetDims(),
        get_kdnn_data_t(mem_desc_diff_wei.data_type()), src.GetLayout());
    return KDNN::Status::SUCCESS == KDNN::PReLULayerBWD::ValidateInput(src, diff_src, wei, diff_wei,
        get_kdnn_tensor_info(mem_desc_diff_dst));
}

KDNN::SoftmaxLayerFWD* convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_src,
                                               const memory_desc_wrapper& mem_desc_dst,
                                               std::size_t axis) noexcept(false) {
    return new KDNN::SoftmaxLayerFWD{get_kdnn_tensor_info(mem_desc_src), get_kdnn_tensor_info(mem_desc_dst), axis};
}

bool may_convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_src,
                                const memory_desc_wrapper& mem_desc_dst,
                                std::size_t axis) noexcept {
    return KDNN::Status::SUCCESS == KDNN::SoftmaxLayerFWD::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_dst), axis);
}

KDNN::SoftmaxLayerBWD* convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_dst,
                                               const memory_desc_wrapper& mem_desc_dst_diff,
                                               const memory_desc_wrapper& mem_desc_src_diff,
                                               std::size_t axis) noexcept(false) {
    return new KDNN::SoftmaxLayerBWD{get_kdnn_tensor_info(mem_desc_dst), get_kdnn_tensor_info(mem_desc_dst_diff), get_kdnn_tensor_info(mem_desc_src_diff), axis};
}

bool may_convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_dst,
                                const memory_desc_wrapper& mem_desc_dst_diff,
                                const memory_desc_wrapper& mem_desc_src_diff,
                                std::size_t axis) noexcept {
    return KDNN::Status::SUCCESS == KDNN::SoftmaxLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_dst),
                                                                          get_kdnn_tensor_info(mem_desc_dst_diff),
                                                                          get_kdnn_tensor_info(mem_desc_src_diff), axis);
}

KDNN::Gemm<KDNN::GemmPack::NO_PACK>* convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_wei,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_bias) noexcept(false) {
    return new KDNN::Gemm<KDNN::GemmPack::NO_PACK>{get_kdnn_tensor_info(mem_desc_src), get_kdnn_tensor_info(mem_desc_wei),
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_tensor_info(mem_desc_bias)};
}

bool may_convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_wei,
        const memory_desc_wrapper& mem_desc_dst, const memory_desc_wrapper& mem_desc_bias) noexcept {
    return KDNN::Status::SUCCESS == KDNN::Gemm<KDNN::GemmPack::NO_PACK>::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_wei), get_kdnn_tensor_info(mem_desc_dst), get_kdnn_tensor_info(mem_desc_bias));
}

KDNN::Gemm<KDNN::GemmPack::NO_PACK>* convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_wei,
        const memory_desc_wrapper& mem_desc_dst) noexcept(false) {
    return new KDNN::Gemm<KDNN::GemmPack::NO_PACK>{get_kdnn_tensor_info(mem_desc_src), get_kdnn_tensor_info(mem_desc_wei),
        get_kdnn_tensor_info(mem_desc_dst)};
}

bool may_convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_wei,
        const memory_desc_wrapper& mem_desc_dst) noexcept {
    return KDNN::Status::SUCCESS == KDNN::Gemm<KDNN::GemmPack::NO_PACK>::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_wei), get_kdnn_tensor_info(mem_desc_dst));
}

KDNN::ReductionLayer* convert_to_kdnn_reduction(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& reduction_alg) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo dst = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
        get_kdnn_data_t(mem_desc_dst.data_type()), src.GetLayout());
    return new KDNN::ReductionLayer{src, dst, get_kdnn_reduction_alg(reduction_alg)};
}

bool may_convert_to_kdnn_reduction(const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_dst,
                             const alg_kind_t& reduction_alg) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo dst = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
        get_kdnn_data_t(mem_desc_dst.data_type()), src.GetLayout());
    return KDNN::Status::SUCCESS == KDNN::ReductionLayer::ValidateInput(src, dst, get_kdnn_reduction_alg(reduction_alg));
}

KDNN::NormalizationLayerFWD* convert_to_kdnn_layer_normalization_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
        const memory_desc_wrapper& mem_desc_dst, bool use_global_stats, bool use_scale, bool use_shift) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo stats = src;
    KDNN::TensorInfo scaleshift = src;
    if (mem_desc_stats == &glob_zero_md) {
        KDNN::Shape sh;
        KDNN::Layout l;
        switch (src.GetNumDims())
        {
        case 2:
            sh = {src.GetDims()[0]};
            l = KDNN::Layout::A;
            break;
        case 3:
            sh = {src.GetDims()[0], src.GetDims()[1]};
            l = KDNN::Layout::AB;
            break;
        case 4:
            sh = {src.GetDims()[0], src.GetDims()[1], src.GetDims()[2]};
            l = KDNN::Layout::ABC;
            break;
        case 5:
            sh = {src.GetDims()[0], src.GetDims()[1], src.GetDims()[2], src.GetDims()[3]};
            l = KDNN::Layout::ABCD;
            break;
        default:
            break;
        }
        stats = KDNN::TensorInfo(sh ,src.GetType(), l);
    } else {
        stats = get_kdnn_tensor_info(mem_desc_stats);
    }
    if (mem_desc_scaleshift == &glob_zero_md) {
        scaleshift = KDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KDNN::Layout::A);
    } else {
        scaleshift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    std::uint32_t flags =  static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    flags |= use_global_stats ? static_cast<std::uint32_t>(KDNN::NormalizationFlags::USE_GLOBAL_STATS) : static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    flags |= use_scale ? static_cast<std::uint32_t>(KDNN::NormalizationFlags::USE_SCALE) : static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    flags |= use_shift ? static_cast<std::uint32_t>(KDNN::NormalizationFlags::USE_SHIFT) : static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    return new KDNN::NormalizationLayerFWD{src,
        stats, scaleshift, dst, static_cast<KDNN::NormalizationFlags>(flags)};
}

bool may_convert_to_kdnn_layer_normalization_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
        const memory_desc_wrapper& mem_desc_dst, bool use_global_stats, bool use_scale, bool use_shift) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KDNN::TensorInfo stats = src;
    KDNN::TensorInfo scaleshift = src;
    if (mem_desc_stats == &glob_zero_md) {
        KDNN::Shape sh;
        KDNN::Layout l;
        switch (src.GetNumDims())
        {
        case 2:
            sh = {src.GetDims()[0]};
            l = KDNN::Layout::A;
            break;
        case 3:
            sh = {src.GetDims()[0], src.GetDims()[1]};
            l = KDNN::Layout::AB;
            break;
        case 4:
            sh = {src.GetDims()[0], src.GetDims()[1], src.GetDims()[2]};
            l = KDNN::Layout::ABC;
            break;
        case 5:
            sh = {src.GetDims()[0], src.GetDims()[1], src.GetDims()[2], src.GetDims()[3]};
            l = KDNN::Layout::ABCD;
            break;
        default:
            break;
        }
        stats = KDNN::TensorInfo(sh ,src.GetType(), l);
    } else {
        stats = get_kdnn_tensor_info(mem_desc_stats);
    }
    if (mem_desc_scaleshift == &glob_zero_md) {
        scaleshift = KDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KDNN::Layout::A);
    } else {
        scaleshift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    std::uint32_t flags =  static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    flags |= use_global_stats ? static_cast<std::uint32_t>(KDNN::NormalizationFlags::USE_GLOBAL_STATS) : static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    flags |= use_scale ? static_cast<std::uint32_t>(KDNN::NormalizationFlags::USE_SCALE) : static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    flags |= use_shift ? static_cast<std::uint32_t>(KDNN::NormalizationFlags::USE_SHIFT) : static_cast<std::uint32_t>(KDNN::NormalizationFlags::NONE);
    return KDNN::Status::SUCCESS == KDNN::NormalizationLayerFWD::ValidateInput(src,
        stats, scaleshift, dst, static_cast<KDNN::NormalizationFlags>(flags));
}

KDNN::NormalizationLayerBWD* convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d,
        const memory_desc_wrapper& sc_d, const memory_desc_wrapper& diff_sc_d) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(src_d);
    KDNN::TensorInfo stat = get_kdnn_tensor_info(stat_d);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(diff_src_d);
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(diff_dst_d);
    KDNN::TensorInfo sc = get_kdnn_tensor_info(sc_d);
    KDNN::TensorInfo diff_sc = get_kdnn_tensor_info(diff_sc_d);
    return new KDNN::NormalizationLayerBWD{src, stat, diff_src,
        diff_dst, sc, diff_sc};
}

bool may_convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d,
        const memory_desc_wrapper& sc_d, const memory_desc_wrapper& diff_sc_d) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(src_d);
    KDNN::TensorInfo stat = get_kdnn_tensor_info(stat_d);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(diff_src_d);
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(diff_dst_d);
    KDNN::TensorInfo sc = get_kdnn_tensor_info(sc_d);
    KDNN::TensorInfo diff_sc = get_kdnn_tensor_info(diff_sc_d);
    return KDNN::Status::SUCCESS == KDNN::NormalizationLayerBWD::ValidateInput(src, stat, diff_src,
        diff_dst, sc, diff_sc);
}

KDNN::NormalizationLayerBWD* convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d) noexcept(false) {
    KDNN::TensorInfo src = get_kdnn_tensor_info(src_d);
    KDNN::TensorInfo stat = get_kdnn_tensor_info(stat_d);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(diff_src_d);
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(diff_dst_d);
    return new KDNN::NormalizationLayerBWD{src, stat, diff_src,
        diff_dst, KDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KDNN::Layout::A),
        KDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KDNN::Layout::A)};
}

bool may_convert_to_kdnn_layer_normalization_bwd(const memory_desc_wrapper& src_d, const memory_desc_wrapper& stat_d,
        const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d) noexcept {
    KDNN::TensorInfo src = get_kdnn_tensor_info(src_d);
    KDNN::TensorInfo stat = get_kdnn_tensor_info(stat_d);
    KDNN::TensorInfo diff_src = get_kdnn_tensor_info(diff_src_d);
    KDNN::TensorInfo diff_dst = get_kdnn_tensor_info(diff_dst_d);
    return KDNN::Status::SUCCESS == KDNN::NormalizationLayerBWD::ValidateInput(src, stat, diff_src,
        diff_dst, KDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KDNN::Layout::A),
        KDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KDNN::Layout::A));
}

KDNN::SumLayer* convert_to_kdnn_sum(const std::vector<memory_desc_wrapper>& mem_desc_src, const float *scales,
                                  const memory_desc_wrapper& mem_desc_dst) noexcept(false) {
    std::vector<KDNN::TensorInfo> mem_desc_src_kdnn;
    for (size_t ind = 0; ind < mem_desc_src.size(); ++ind) {
        mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(mem_desc_src[ind]));
    }
    return new KDNN::SumLayer{mem_desc_src_kdnn,
                            scales,
                            get_kdnn_tensor_info(mem_desc_dst)};
}

bool may_convert_to_kdnn_sum(const std::vector<memory_desc_wrapper>& mem_desc_src, const float *scales,
                                  const memory_desc_wrapper& mem_desc_dst) noexcept {
    std::vector<KDNN::TensorInfo> mem_desc_src_kdnn;
    for (size_t ind = 0; ind < mem_desc_src.size(); ++ind) {
        mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(mem_desc_src[ind]));
    }
    return KDNN::Status::SUCCESS == KDNN::SumLayer::ValidateInput(mem_desc_src_kdnn, scales,
        get_kdnn_tensor_info(mem_desc_dst));
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
