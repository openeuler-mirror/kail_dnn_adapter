#ifndef CPU_AARCH64_KDNN_POOLING_HPP
#define CPU_AARCH64_KDNN_POOLING_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_pooling_fwd_resource_t : public resource_t {
    kdnn_pooling_fwd_resource_t(const std::unique_ptr<KDNN::PoolingLayerFWD> &kdnn_pooling_prim) noexcept
        : kdnn_pooling_obj_(new KDNN::PoolingLayerFWD{*(kdnn_pooling_prim.get())}) {}

    KDNN::PoolingLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_pooling_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_pooling_fwd_resource_t);

private:
    std::unique_ptr<KDNN::PoolingLayerFWD> kdnn_pooling_obj_;
}; // kdnn_pooling_fwd_resource_t

struct kdnn_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        pd_t(const pd_t& p) : cpu_pooling_fwd_pd_t(p) {
            if (p.kdnn_pooling_prim_) {
                this->kdnn_pooling_prim_.reset(new KDNN::PoolingLayerFWD{*(p.kdnn_pooling_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_tmp_dst_ = p.need_tmp_dst_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_pooling_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = set_default_params() == status::success
                    && is_fwd()
                    && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops)
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            auto&& pooling = kdnn_utils::convert_to_kdnn_pooling_fwd(
                src_d, dst_d, *desc());
            if (!pooling.first) {
                return status::unimplemented;
            } else {
                kdnn_pooling_prim_.reset(pooling.second);
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

        std::unique_ptr<KDNN::PoolingLayerFWD> kdnn_pooling_prim_;

    }; // pd_t

    kdnn_pooling_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_pooling_fwd_resource_t>(pd()->kdnn_pooling_prim_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *ws = CTX_OUT_MEM(void *, DNNL_ARG_WORKSPACE);
        void *dst;

        if (pd()->need_tmp_dst_) {
            dst = KDNN::Service::AlignedAlloc(pd()->dst_size_);
        } else {
            dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        }

        return execute_forward(ctx, src, dst, ws);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            void *dst, void *ws) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::PoolingLayerFWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_pooling_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst, ws);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        pd()->post_ops_.execute(ctx, dst);

        if (pd()->need_tmp_dst_) {
            KDNN::Service::Deallocate(dst);
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_pooling_fwd_t

struct kdnn_pooling_bwd_resource_t : public resource_t {
    kdnn_pooling_bwd_resource_t(const std::unique_ptr<KDNN::PoolingLayerBWD> &kdnn_pooling_prim) noexcept
        : kdnn_pooling_obj_(new KDNN::PoolingLayerBWD{*(kdnn_pooling_prim.get())}) {}

    KDNN::PoolingLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_pooling_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_pooling_bwd_resource_t);

private:
    std::unique_ptr<KDNN::PoolingLayerBWD> kdnn_pooling_obj_;
}; // kdnn_pooling_bwd_resource_t

struct kdnn_pooling_bwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        pd_t(const pd_t& p) : cpu_pooling_bwd_pd_t(p) {
            if (p.kdnn_pooling_prim_) {
                this->kdnn_pooling_prim_.reset(new KDNN::PoolingLayerBWD{*(p.kdnn_pooling_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_pooling_bwd_t);

        status_t init(engine_t *engine) {
            const bool ok = set_default_params() == status::success
                    && !is_fwd()
                    && attr()->has_default_values()
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == alg_kind::pooling_max) {
                const auto ws_dt = hint_fwd_pd_->workspace_md()->data_type;
                init_default_ws(ws_dt);
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }
            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            auto&& pooling = kdnn_utils::convert_to_kdnn_pooling_bwd(
                diff_src_d, diff_dst_d, *desc());
            if (!pooling.first) {
                return status::unimplemented;
            } else {
                kdnn_pooling_prim_.reset(pooling.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::PoolingLayerBWD> kdnn_pooling_prim_;

    }; // pd_t

    kdnn_pooling_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_pooling_bwd_resource_t>(pd()->kdnn_pooling_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    // execute_backward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_backward(const exec_ctx_t &ctx) const {

        void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *ws = CTX_IN_MEM(const void *, DNNL_ARG_WORKSPACE);

        return execute_backward(ctx, diff_src, diff_dst, ws);
    }

    // Execute backward with arbitrary src and dst
    status_t execute_backward(const exec_ctx_t &ctx, void *diff_src,
            const void *diff_dst, const void *ws) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::PoolingLayerBWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_pooling_bwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_src, diff_dst, ws);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_pooling_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_POOLING_HPP
