#ifndef CPU_AARCH64_KDNN_POST_OPS_HPP
#define CPU_AARCH64_KDNN_POST_OPS_HPP

#include "kdnn.hpp"

#include "cpu/aarch64/kdnn/kdnn_binary.hpp"
#include "cpu/aarch64/kdnn/kdnn_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_post_ops_t {

    kdnn_post_ops_t() = default;

    status_t init(engine_t *engine, post_ops_t &post_ops,
            const memory_desc_t &dst_md) {

        CHECK(post_ops.set_default_formats(&dst_md));
        dst_data_type = dst_md.data_type;

        post_op_primitives = {};

        for (int i = 0; i < post_ops.len(); i++) {
            auto &po = post_ops.entry_[i];

            if (po.is_sum()) {
                // [TODO]: A temporary dst is needed for this case
                return status::unimplemented;
            } else if (po.is_binary()) {
                binary_desc_t po_desc;
                po_desc.primitive_kind = primitive_kind::binary;
                po_desc.alg_kind = po.binary.alg;
                po_desc.src_desc[0] = dst_md;
                po_desc.src_desc[1] = po.binary.src1_desc;
                po_desc.dst_desc = dst_md;
                auto empty_attr = dnnl_primitive_attr();
                typename kdnn_binary_t::pd_t kdnn_binary_pd(
                        &po_desc, &empty_attr, nullptr);
                CHECK(kdnn_binary_pd.init(engine));

                auto kdnn_binary
                        = std::make_shared<kdnn_binary_t>(&kdnn_binary_pd);
                CHECK(kdnn_binary->init(engine));
                post_op_primitives.push_back(kdnn_binary);
            } else if (po.is_eltwise()) {
                // Eltwise post op scale must be 1 (no scale)
                if (po.eltwise.scale != 1.0f) {
                    return dnnl::impl::status::unimplemented;
                }

                eltwise_desc_t eltwise_desc;
                eltwise_desc.primitive_kind = primitive_kind::eltwise;
                eltwise_desc.alg_kind = po.eltwise.alg;
                eltwise_desc.alpha = po.eltwise.alpha;
                eltwise_desc.beta = po.eltwise.beta;
                // When will we support fp16: pass eltwise a desc with
                // f32 datatype to perform the operation in fp32 rather than fp16
                eltwise_desc.src_desc = dst_md;
                eltwise_desc.dst_desc = dst_md;
                eltwise_desc.prop_kind = prop_kind_t::dnnl_forward;
                auto empty_attr = dnnl_primitive_attr();
                typename kdnn_eltwise_fwd_t::pd_t kdnn_eltwise_pd(
                        &eltwise_desc, &empty_attr, nullptr);
                CHECK(kdnn_eltwise_pd.init(engine));
                auto kdnn_eltwise
                        = std::make_shared<kdnn_eltwise_fwd_t>(&kdnn_eltwise_pd);
                CHECK(kdnn_eltwise->init(engine));
                post_op_primitives.push_back(kdnn_eltwise);
            } else {
                return status::unimplemented;
            }
        }

        return status::success;
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const {
        for (const auto &post_op : post_op_primitives) {
            CHECK(post_op->create_resource(engine, mapper));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx, void *src) const {

        int post_op_index = 0;

        for (auto &post_op : post_op_primitives) {
            if (post_op->kind() == primitive_kind::binary) {
                auto binary_post_op
                    = static_cast<kdnn_binary_t *>(post_op.get());
                
                const void *src1 = CTX_IN_MEM(const void *,
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                | DNNL_ARG_SRC_1));
                CHECK(binary_post_op->execute_forward(ctx, src, src1, src));
            } else if (post_op->kind() == primitive_kind::eltwise) {
                auto eltwise_post_op
                    = static_cast<kdnn_eltwise_fwd_t *>(post_op.get());

                CHECK(eltwise_post_op->execute_forward(ctx, src, src));
            } else {
                return status::runtime_error;
            }

            ++post_op_index;
        }

        return status::success;
    }

private:
    data_type_t dst_data_type;
    // Vector of primitives used to execute the post ops. They are constructed
    // in init to be either kdnn_binary_t (for sum, add, sub, div, mul, min and
    // max) or kdnn_eltwise_fwd_t (for relu, elu, tanh, square, abs etc)
    std::vector<std::shared_ptr<primitive_t>> post_op_primitives;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_POST_OPS_HPP
