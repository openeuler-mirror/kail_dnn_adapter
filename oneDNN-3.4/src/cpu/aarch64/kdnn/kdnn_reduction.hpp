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
                this->post_ops_ = p.post_ops_;
                this->need_tmp_dst_ = p.need_tmp_dst_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_reduction_t);
        
        status_t init(engine_t *engine) {
            const bool ok = set_default_params() == status::success
                    && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops)
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            auto&& reduction = kdnn_utils::convert_to_kdnn_reduction(src_d, dst_d, desc_.alg_kind, desc_.p, desc_.eps);
            if (!reduction.first) {
                return status::unimplemented;
            } else {
                kdnn_reduction_prim_.reset(reduction.second);
                CHECK(post_ops_.init(engine, attr_.post_ops_, dst_md_));
                if (post_ops_.has_sum()) {
                    need_tmp_dst_ = true;
                    dst_size_ = dst_d.nelems() * dst_d.data_type_size();
                } else {
                    need_tmp_dst_ = false;
                    dst_size_ = 0;
                }
                return status::success;
            }
        }

        bool need_tmp_dst_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;

        std::unique_ptr<KDNN::ReductionLayer> kdnn_reduction_prim_;

    }; // pd_t

    kdnn_reduction_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_reduction_resource_t>(pd()->kdnn_reduction_prim_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *dst;

        if (pd()->need_tmp_dst_) {
            dst = KDNN::Service::AlignedAlloc(pd()->dst_size_);
        } else {
            dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        }

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

        pd()->post_ops_.execute(ctx, dst);

        if (pd()->need_tmp_dst_) {
            KDNN::Service::Deallocate(dst);
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_reduction_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_REDUCTION_HPP
