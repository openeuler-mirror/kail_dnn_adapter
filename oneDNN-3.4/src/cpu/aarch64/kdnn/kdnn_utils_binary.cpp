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

std::pair<bool, KDNN::BinaryLayer*> convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
                                                           const memory_desc_wrapper& mem_desc_src1,
                                                           const memory_desc_wrapper& mem_desc_dst,
                                                           const alg_kind_t& binary_op) noexcept(false)
{
    using namespace format_tag;

    if (!common_tensor_checks(mem_desc_src0, mem_desc_src1, mem_desc_dst)) {
        return {false, nullptr};
    }
    KDNN::TensorInfo src0 = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src0).GetDims(),
                                             get_kdnn_data_t(mem_desc_src0.data_type()),
                                             get_kdnn_layout(mem_desc_src0));
    KDNN::TensorInfo src1 = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src1).GetDims(),
                                             get_kdnn_data_t(mem_desc_src1.data_type()),
                                             get_kdnn_layout(mem_desc_src1));
    KDNN::TensorInfo dst = KDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
                                            get_kdnn_data_t(mem_desc_dst.data_type()),
                                            get_kdnn_layout(mem_desc_dst));
    if (KDNN::Status::SUCCESS != KDNN::BinaryLayer::ValidateInput(src0, src1, dst, get_kdnn_op(binary_op))) {
        return {false, nullptr};
    }

    return {true, new KDNN::BinaryLayer{src0, src1, dst, get_kdnn_op(binary_op)}};
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
