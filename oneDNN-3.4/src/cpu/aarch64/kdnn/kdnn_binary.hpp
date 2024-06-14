#ifndef CPU_AARCH64_KDNN_BINARY_HPP
#define CPU_AARCH64_KDNN_BINARY_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_binary_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_binary_resource_t : public resource_t {
    kdnn_binary_resource_t(const std::unique_ptr<KDNN::BinaryLayer> &kdnn_binary_prim) noexcept
        : kdnn_binary_obj_(new KDNN::BinaryLayer{*(kdnn_binary_prim.get())}) {}

    KDNN::BinaryLayer &get_kdnn_obj() const noexcept { return *kdnn_binary_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_binary_resource_t);

private:
    std::unique_ptr<KDNN::BinaryLayer> kdnn_binary_obj_;
}; // kdnn_binary_resource_t

struct kdnn_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        pd_t(const pd_t& p) : cpu_binary_pd_t(p) {
            if (p.kdnn_binary_prim_) {
                this->kdnn_binary_prim_.reset(new KDNN::BinaryLayer{*(p.kdnn_binary_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_binary_t);

        status_t init(engine_t *engine) {
 
            CHECK(set_default_params());

            if (!attr()->has_default_values()) return status::unimplemented;

            using namespace format_tag;
            auto src0_tag = memory_desc_matches_one_of_tag(src0_md_, ndhwc, ncdhw, nchw, nhwc, nwc, ncw, nc, x);
            auto src1_tag = memory_desc_matches_one_of_tag(src1_md_, src0_tag);
            if (utils::one_of(format_tag::undef, src0_tag, src1_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());
            if (!kdnn_utils::is_data_type_supported_by_kdnn(src0_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(src1_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(dst_d.data_type())) {
                    return status::unimplemented;
            }
            if (src0_d.ndims() < 1 || src0_d.ndims() > 5 ||
                src1_d.ndims() < 1 || src1_d.ndims() > 5 ||
                dst_d.ndims() < 1 || dst_d.ndims() > 5) {
                return status::unimplemented;
            }
            // Check only src0 and dst layouts, because layout src1 must be equal to layout src0.
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(src0_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(dst_d)) {
                    return status::unimplemented;
            }
            if (!kdnn_utils::may_convert_to_kdnn_binary(src0_d, src1_d, dst_d, desc_.alg_kind)) {
                return status::unimplemented;
            } else {
                kdnn_binary_prim_.reset(kdnn_utils::convert_to_kdnn_binary(src0_d, src1_d, dst_d, desc_.alg_kind));
            }

            return status::success;
        }

        std::unique_ptr<KDNN::BinaryLayer> kdnn_binary_prim_;

        friend struct kdnn_post_ops_t;
    }; // pd_t

    kdnn_binary_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_binary_resource_t>(pd()->kdnn_binary_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        auto src0 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_0);
        auto src1 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_1);
        auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src0, src1, dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src0,
            const void *src1, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::BinaryLayer &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_binary_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src0, src1, dst);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    friend struct kdnn_post_ops_t;
}; // kdnn_binary_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_BINARY_HPP
