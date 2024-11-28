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

std::pair<bool, KDNN::ActivationLayerFWD*> convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    if (KDNN::Status::SUCCESS != KDNN::ActivationLayerFWD::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_alg(eltwise_alg), alpha, beta)) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::ActivationLayerFWD{get_kdnn_tensor_info(mem_desc_src),
            get_kdnn_tensor_info(mem_desc_dst), get_kdnn_alg(eltwise_alg), alpha, beta}};
    }
}

std::pair<bool, KDNN::ActivationLayerBWD*> convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_ds,
        const memory_desc_wrapper& mem_desc_dd, const memory_desc_wrapper& mem_desc_src,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    if (!common_tensor_checks(mem_desc_ds, mem_desc_dd, mem_desc_src)) {
        return {false, nullptr};
    }
    if (KDNN::Status::SUCCESS != KDNN::ActivationLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_ds),
        get_kdnn_tensor_info(mem_desc_dd), get_kdnn_tensor_info(mem_desc_src), get_kdnn_alg(eltwise_alg), alpha, beta)) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::ActivationLayerBWD{get_kdnn_tensor_info(mem_desc_ds), get_kdnn_tensor_info(mem_desc_dd),
        get_kdnn_tensor_info(mem_desc_src), get_kdnn_alg(eltwise_alg), alpha, beta}};
    }
}

std::pair<bool, KDNN::ActivationLayerBWD*> convert_to_kdnn_act_bwd_with_dst(const memory_desc_wrapper& mem_desc_ds,
        const memory_desc_wrapper& mem_desc_dd, const memory_desc_wrapper& mem_desc_src,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    if (!common_tensor_checks(mem_desc_ds, mem_desc_dd, mem_desc_src)) {
        return {false, nullptr};
    }
    if (KDNN::Status::SUCCESS != KDNN::ActivationLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_ds),
        get_kdnn_tensor_info(mem_desc_dd), get_kdnn_tensor_info(mem_desc_src), get_kdnn_with_dst_alg(eltwise_alg), alpha, beta)) {
        return {false, nullptr};
    } else {
        return {true, new KDNN::ActivationLayerBWD{get_kdnn_tensor_info(mem_desc_ds), get_kdnn_tensor_info(mem_desc_dd),
        get_kdnn_tensor_info(mem_desc_src), get_kdnn_with_dst_alg(eltwise_alg), alpha, beta}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
