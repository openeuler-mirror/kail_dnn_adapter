#ifndef CPU_AARCH64_KDNN_LOCAL_RESPONSE_NORMALIZATION_HPP
#define CPU_AARCH64_KDNN_LOCAL_RESPONSE_NORMALIZATION_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_lrn_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_lrn_fwd_resource_t : public resource_t {
    kdnn_lrn_fwd_resource_t(const std::unique_ptr<KDNN::LocalResponseNormalizationLayerFWD> &kdnn_lrn_prim) noexcept
        : kdnn_lrn_obj_(new KDNN::LocalResponseNormalizationLayerFWD{*(kdnn_lrn_prim.get())}) {}

    KDNN::LocalResponseNormalizationLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_lrn_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_lrn_fwd_resource_t);

private:
    std::unique_ptr<KDNN::LocalResponseNormalizationLayerFWD> kdnn_lrn_obj_;
}; // kdnn_lrn_fwd_resource_t

struct kdnn_lrn_fwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_fwd_pd_t {
        using cpu_lrn_fwd_pd_t::
                cpu_lrn_fwd_pd_t;
        
        pd_t(const pd_t& p) : cpu_lrn_fwd_pd_t(p) {
            if (p.kdnn_lrn_prim_) {
                this->kdnn_lrn_prim_.reset(new KDNN::LocalResponseNormalizationLayerFWD{*(p.kdnn_lrn_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_lrn_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = is_fwd()
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            bool across_channels = (desc_.alg_kind == alg_kind::lrn_across_channels);
            auto&& lrnorm_fwd = kdnn_utils::convert_to_kdnn_lrn_fwd(src_d, dst_d, desc_.lrn_alpha, desc_.lrn_beta,
                desc_.lrn_k, desc_.local_size, across_channels);
            if (!lrnorm_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_lrn_prim_.reset(lrnorm_fwd.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::LocalResponseNormalizationLayerFWD> kdnn_lrn_prim_;
    }; // pd_t

    kdnn_lrn_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_lrn_fwd_resource_t>(pd()->kdnn_lrn_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, dst);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::LocalResponseNormalizationLayerFWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_lrn_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_lrn_fwd_t

struct kdnn_lrn_bwd_resource_t : public resource_t {
    kdnn_lrn_bwd_resource_t(const std::unique_ptr<KDNN::LocalResponseNormalizationLayerBWD> &kdnn_lrn_prim) noexcept
        : kdnn_lrn_obj_(new KDNN::LocalResponseNormalizationLayerBWD{*(kdnn_lrn_prim.get())}) {}

    KDNN::LocalResponseNormalizationLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_lrn_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_lrn_bwd_resource_t);

private:
    std::unique_ptr<KDNN::LocalResponseNormalizationLayerBWD> kdnn_lrn_obj_;
}; // kdnn_lrn_bwd_resource_t

struct kdnn_lrn_bwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_bwd_pd_t {
        using cpu_lrn_bwd_pd_t::
                cpu_lrn_bwd_pd_t;
        
        pd_t(const pd_t& p) : cpu_lrn_bwd_pd_t(p) {
            if (p.kdnn_lrn_prim_) {
                this->kdnn_lrn_prim_.reset(new KDNN::LocalResponseNormalizationLayerBWD{*(p.kdnn_lrn_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_lrn_bwd_t);

        status_t init(engine_t *engine) {

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            bool ok = !is_fwd()
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            bool across_channels = (desc_.alg_kind == alg_kind::lrn_across_channels);
            auto&& lrnorm_bwd = kdnn_utils::convert_to_kdnn_lrn_bwd(src_d,
                diff_dst_d, diff_src_d,
                desc_.lrn_alpha, desc_.lrn_beta,
                desc_.lrn_k, desc_.local_size, across_channels);
            if (!lrnorm_bwd.first) {
                return status::unimplemented;
            } else {
                kdnn_lrn_prim_.reset(lrnorm_bwd.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::LocalResponseNormalizationLayerBWD> kdnn_lrn_prim_;
    }; // pd_t

    kdnn_lrn_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_lrn_bwd_resource_t>(pd()->kdnn_lrn_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    // execute_backward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_backward(const exec_ctx_t &ctx) const {
        auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
        return execute_backward(ctx, src, diff_dst, diff_src);
    }

    status_t execute_backward(const exec_ctx_t &ctx, const void *src,
            const void *diff_dst, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::LocalResponseNormalizationLayerBWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_lrn_bwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, diff_dst,
                diff_src);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_lrn_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_LOCAL_RESPONSE_NORMALIZATION_HPP
