#ifndef CPU_AARCH64_KDNN_REDUCTION_HPP
#define CPU_AARCH64_KDNN_REDUCTION_HPP

#include "kdnn.hpp"

#include "cpu/cpu_reduction_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_reduction_resource_t : public resource_t {
    kdnn_reduction_resource_t(const std::unique_ptr<KDNN::ReductionLayer> &kdnn_reduction_prim) noexcept
        : kdnn_reduction_obj_(new KDNN::ReductionLayer{*(kdnn_reduction_prim.get())}) {}

    KDNN::ReductionLayer &get_kdnn_obj() const noexcept { return *kdnn_reduction_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_reduction_resource_t);

private:
    std::unique_ptr<KDNN::ReductionLayer> kdnn_reduction_obj_;
}; // kdnn_reduction_resource_t

struct kdnn_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_reduction_pd_t(p) {
            if (p.kdnn_reduction_prim_) {
                this->kdnn_reduction_prim_.reset(new KDNN::ReductionLayer{*(p.kdnn_reduction_prim_.get())});
                this->post_ops = p.post_ops;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_reduction_t);
        
        status_t init(engine_t *engine) {
            const bool ok = set_default_params() == status::success
                    && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops)
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto src_tag = memory_desc_matches_one_of_tag(src_md_, ndhwc, ncdhw, nchw, nhwc, nwc, ncw, nc, x);
            auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, src_tag);
            if (utils::one_of(format_tag::undef, src_tag, dst_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            if (!kdnn_utils::is_data_type_supported_by_kdnn(src_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(dst_d.data_type())) {
                    return status::unimplemented;
            }
            if (src_d.ndims() < 1 || src_d.ndims() > 5 ||
                dst_d.ndims() < 1 || dst_d.ndims() > 5) {
                return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(src_d)) {
                    return status::unimplemented;
            }
            if (!kdnn_utils::may_convert_to_kdnn_reduction(src_d, dst_d, desc_.alg_kind)) {
                return status::unimplemented;
            } else {
                kdnn_reduction_prim_.reset(kdnn_utils::convert_to_kdnn_reduction(src_d, dst_d, desc_.alg_kind));
            }

            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_));

            return status::success;
        }

        kdnn_post_ops_t post_ops;

        std::unique_ptr<KDNN::ReductionLayer> kdnn_reduction_prim_;

    }; // pd_t

    kdnn_reduction_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_reduction_resource_t>(pd()->kdnn_reduction_prim_));
        CHECK(pd()->post_ops.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::ReductionLayer &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_reduction_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        pd()->post_ops.execute(ctx, dst);

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_reduction_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_REDUCTION_HPP
