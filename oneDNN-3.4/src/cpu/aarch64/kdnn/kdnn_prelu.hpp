#ifndef CPU_AARCH64_KDNN_PRELU_HPP
#define CPU_AARCH64_KDNN_PRELU_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_prelu_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_prelu_fwd_resource_t : public resource_t {
    kdnn_prelu_fwd_resource_t(const std::unique_ptr<KDNN::PReLULayerFWD> &kdnn_prelu_prim) noexcept
        : kdnn_prelu_obj_(new KDNN::PReLULayerFWD{*(kdnn_prelu_prim.get())}) {}

    KDNN::PReLULayerFWD &get_kdnn_obj() const noexcept { return *kdnn_prelu_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_prelu_fwd_resource_t);

private:
    std::unique_ptr<KDNN::PReLULayerFWD> kdnn_prelu_obj_;
}; // kdnn_prelu_fwd_resource_t

struct kdnn_prelu_fwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_fwd_pd_t {
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_prelu_fwd_pd_t(p) {
            if (p.kdnn_prelu_prim_) {
                this->kdnn_prelu_prim_.reset(new KDNN::PReLULayerFWD{*(p.kdnn_prelu_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_prelu_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = is_fwd() && set_default_formats() &&
                            attr()->has_default_values();

            if (!ok) return status::unimplemented;

            auto&& first_non_any_md = kdnn_utils::get_first_non_any_format_kind(src_md_, weights_md_);
            auto&& layout = kdnn_utils::get_layout(first_non_any_md);
            if (src_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(src_md_, layout));
                CHECK(memory_desc_init_by_tag(dst_md_, layout));
            }
            if (weights_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(weights_md_, layout));
            }

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper wei_d(&weights_md_);
            const memory_desc_wrapper dst_d(&dst_md_);
            auto&& prelu_fwd = kdnn_utils::convert_to_kdnn_prelu(src_d, wei_d, dst_d);
            if (!prelu_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_prelu_prim_.reset(prelu_fwd.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::PReLULayerFWD> kdnn_prelu_prim_;
        
        friend struct kdnn_post_ops_t;
    }; // pd_t

    kdnn_prelu_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_prelu_fwd_resource_t>(pd()->kdnn_prelu_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, wei, dst);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::PReLULayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_prelu_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, wei, dst);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    friend struct kdnn_post_ops_t;
}; // kdnn_prelu_fwd_t

struct kdnn_prelu_bwd_resource_t : public resource_t {
    kdnn_prelu_bwd_resource_t(const std::unique_ptr<KDNN::PReLULayerBWD> &kdnn_prelu_prim) noexcept
        : kdnn_prelu_obj_(new KDNN::PReLULayerBWD{*(kdnn_prelu_prim.get())}) {}

    KDNN::PReLULayerBWD &get_kdnn_obj() const noexcept { return *kdnn_prelu_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_prelu_bwd_resource_t);

private:
    std::unique_ptr<KDNN::PReLULayerBWD> kdnn_prelu_obj_;
}; // kdnn_prelu_bwd_resource_t

struct kdnn_prelu_bwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_bwd_pd_t {
        using cpu_prelu_bwd_pd_t::cpu_prelu_bwd_pd_t;

        pd_t(const pd_t& p) : cpu_prelu_bwd_pd_t(p) {
            if (p.kdnn_prelu_prim_) {
                this->kdnn_prelu_prim_.reset(new KDNN::PReLULayerBWD{*(p.kdnn_prelu_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_prelu_bwd_t);

        status_t init(engine_t *engine) {
            const bool ok = !is_fwd() && set_default_formats()
                        && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            auto&& first_non_any_md = kdnn_utils::get_first_non_any_format_kind(src_md_, weights_md_);
            auto&& layout = kdnn_utils::get_layout(first_non_any_md);
            if (src_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(src_md_, layout));
                CHECK(memory_desc_init_by_tag(diff_src_md_, layout));
                CHECK(memory_desc_init_by_tag(diff_dst_md_, layout));
            }
            if (weights_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(weights_md_, layout));
                CHECK(memory_desc_init_by_tag(diff_weights_md_, layout));
            }

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper wei_d(weights_md());
            const memory_desc_wrapper diff_wei_d(diff_weights_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            auto&& prelu_bwd = kdnn_utils::convert_to_kdnn_prelu(src_d, diff_src_d, wei_d, diff_wei_d, diff_dst_d);
            if (!prelu_bwd.first) {
                return status::unimplemented;
            } else {
                kdnn_prelu_prim_.reset(prelu_bwd.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::PReLULayerBWD> kdnn_prelu_prim_;
        
    }; // pd_t

    kdnn_prelu_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_prelu_bwd_resource_t>(pd()->kdnn_prelu_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward(const exec_ctx_t &ctx) const {
        const void *src      = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *diff_src       = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
        const void *weights  = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *diff_weights   = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);

        return execute_backward(ctx, src, diff_src, weights, diff_weights, diff_dst);
    }

    status_t execute_backward(const exec_ctx_t &ctx, const void *src,
            void *diff_src, const void *weights, void *diff_weights,
            const void *diff_dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::PReLULayerBWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_prelu_bwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, diff_src, weights, diff_weights, diff_dst);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_prelu_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_PRELU_HPP
