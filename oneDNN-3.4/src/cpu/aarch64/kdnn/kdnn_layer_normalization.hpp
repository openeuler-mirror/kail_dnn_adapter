#ifndef CPU_AARCH64_KDNN_LAYER_NORMALIZATION_HPP
#define CPU_AARCH64_KDNN_LAYER_NORMALIZATION_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_layer_normalization_fwd_resource_t : public resource_t {
    kdnn_layer_normalization_fwd_resource_t(const std::unique_ptr<KDNN::NormalizationLayerFWD> &kdnn_layer_normalization_prim) noexcept
        : kdnn_layer_normalization_obj_(new KDNN::NormalizationLayerFWD{*(kdnn_layer_normalization_prim.get())}) {}

    KDNN::NormalizationLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_layer_normalization_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_layer_normalization_fwd_resource_t);

private:
    std::unique_ptr<KDNN::NormalizationLayerFWD> kdnn_layer_normalization_obj_;
}; // kdnn_layer_normalization_fwd_resource_t

struct kdnn_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;
        
        pd_t(const pd_t& p) : cpu_layer_normalization_fwd_pd_t(p) {
            if (p.kdnn_layer_normalization_prim_) {
                this->kdnn_layer_normalization_prim_.reset(new KDNN::NormalizationLayerFWD{*(p.kdnn_layer_normalization_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_layer_normalization_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = is_fwd()
                    && attr()->has_default_values(primitive_attr_t::skip_mask_t::scales_runtime)
                    && attr_scales_ok()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper stats_d((!stats_are_src() && is_training()) ? dst_md(1) : src_md(1));
            const memory_desc_wrapper scaleshift_d(scaleshift_md_);
            if (!kdnn_utils::is_data_type_supported_by_kdnn(src_d.data_type()) ||
                ((stats_are_src() || (!stats_are_src() && is_training())) &&
                !kdnn_utils::is_data_type_supported_by_kdnn(stats_d.data_type())) ||
                (((use_scale()) || use_shift()) &&
                !kdnn_utils::is_data_type_supported_by_kdnn(scaleshift_d.data_type())) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(dst_d.data_type())) {
                    return status::unimplemented;
            }
            if ((src_d.ndims() < 1 || src_d.ndims() > 5) ||
                ((stats_are_src() || (!stats_are_src() && is_training())) &&
                (stats_d.ndims() < 1 || stats_d.ndims() > 5)) ||
                (((use_scale()) || use_shift()) &&
                (scaleshift_d.ndims() < 1 || scaleshift_d.ndims() > 5)) ||
                (dst_d.ndims() < 1 || dst_d.ndims() > 5)) {
                return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(src_d) ||
                ((stats_are_src() || (!stats_are_src() && is_training())) &&
                !kdnn_utils::is_data_layout_supported_by_kdnn(stats_d)) ||
                (((use_scale()) || use_shift()) &&
                !kdnn_utils::is_data_layout_supported_by_kdnn(scaleshift_d)) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(dst_d)) {
                    return status::unimplemented;
            }
            if (!kdnn_utils::may_convert_to_kdnn_layer_normalization_fwd(src_d, stats_d, scaleshift_d, dst_d, stats_are_src(), use_scale(), use_shift())) {
               return status::unimplemented;
            } else {
                kdnn_layer_normalization_prim_.reset(kdnn_utils::convert_to_kdnn_layer_normalization_fwd(src_d, stats_d, scaleshift_d, dst_d, stats_are_src(), use_scale(), use_shift()));
            }

            return status::success;
        }

        std::unique_ptr<KDNN::NormalizationLayerFWD> kdnn_layer_normalization_prim_;
    }; // pd_t

    kdnn_layer_normalization_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_layer_normalization_fwd_resource_t>(pd()->kdnn_layer_normalization_prim_));

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

        auto scale = CTX_IN_MEM(const void *, DNNL_ARG_SCALE);
        auto shift = CTX_IN_MEM(const void *, DNNL_ARG_SHIFT);

        auto mean = CTX_OUT_MEM(void *, DNNL_ARG_MEAN);
        auto variance = CTX_OUT_MEM(void *, DNNL_ARG_VARIANCE);

        const float eps = pd()->desc()->layer_norm_epsilon;
        const bool save_stats = pd()->is_training();

        return execute_forward(ctx, src, dst, scale, shift,
                mean, variance, save_stats, eps);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            void *dst, const void *scale, const void *shift,
            void *mean, void *variance,
            bool save_stats, const float eps) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::NormalizationLayerFWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_layer_normalization_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst, scale, shift, mean, variance, save_stats, eps);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_layer_normalization_fwd_t

struct kdnn_layer_normalization_bwd_resource_t : public resource_t {
    kdnn_layer_normalization_bwd_resource_t(const std::unique_ptr<KDNN::NormalizationLayerBWD> &kdnn_layer_normalization_prim) noexcept
        : kdnn_layer_normalization_obj_(new KDNN::NormalizationLayerBWD{*(kdnn_layer_normalization_prim.get())}) {}

    KDNN::NormalizationLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_layer_normalization_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_layer_normalization_bwd_resource_t);

private:
    std::unique_ptr<KDNN::NormalizationLayerBWD> kdnn_layer_normalization_obj_;
}; // kdnn_layer_normalization_bwd_resource_t

struct kdnn_layer_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        using cpu_layer_normalization_bwd_pd_t::
                cpu_layer_normalization_bwd_pd_t;
        
        pd_t(const pd_t& p) : cpu_layer_normalization_bwd_pd_t(p) {
            if (p.kdnn_layer_normalization_prim_) {
                this->kdnn_layer_normalization_prim_.reset(new KDNN::NormalizationLayerBWD{*(p.kdnn_layer_normalization_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_layer_normalization_bwd_t);

        status_t init(engine_t *engine) {
            bool ok = !is_fwd()
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper stat_d(src_md(1));
            const memory_desc_wrapper diff_src_d(diff_src_md(0));
            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper sc_d(weights_md(0));
            const memory_desc_wrapper diff_sc_d(diff_weights_md(0));
            if (!kdnn_utils::is_data_type_supported_by_kdnn(src_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(stat_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(diff_src_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(diff_dst_d.data_type()) ||
                ((use_scale() || use_shift()) && (!kdnn_utils::is_data_type_supported_by_kdnn(sc_d.data_type()))) ||
                ((use_scale() || use_shift()) && (!kdnn_utils::is_data_type_supported_by_kdnn(diff_sc_d.data_type())))) {
                    return status::unimplemented;
            }
            if ((src_d.ndims() < 1 || src_d.ndims() > 5) ||
                (stat_d.ndims() < 1 || stat_d.ndims() > 5) ||
                (diff_src_d.ndims() < 1 || diff_src_d.ndims() > 5) ||
                (diff_dst_d.ndims() < 1 || diff_dst_d.ndims() > 5) ||
                ((use_scale() || use_shift()) && ((sc_d.ndims() < 1) || (sc_d.ndims() > 5))) ||
                ((use_scale() || use_shift()) && ((diff_sc_d.ndims() < 1) || (diff_sc_d.ndims() > 5)))) {
                    return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(src_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(stat_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(diff_src_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(diff_dst_d) ||
                ((use_scale() || use_shift()) && (!kdnn_utils::is_data_layout_supported_by_kdnn(sc_d))) ||
                ((use_scale() || use_shift()) && (!kdnn_utils::is_data_layout_supported_by_kdnn(diff_sc_d)))) {
                    return status::unimplemented;
            }
            if (use_scale() || use_shift()) {
                if (!kdnn_utils::may_convert_to_kdnn_layer_normalization_bwd(src_d, stat_d, diff_src_d, diff_dst_d, sc_d, diff_sc_d)) {
                   return status::unimplemented;
                } else {
                    kdnn_layer_normalization_prim_.reset(
                            kdnn_utils::convert_to_kdnn_layer_normalization_bwd(src_d, stat_d, diff_src_d, diff_dst_d, sc_d, diff_sc_d));
                }
            } else {
                if (!kdnn_utils::may_convert_to_kdnn_layer_normalization_bwd(src_d, stat_d, diff_src_d, diff_dst_d)) {
                   return status::unimplemented;
                } else {
                    kdnn_layer_normalization_prim_.reset(
                            kdnn_utils::convert_to_kdnn_layer_normalization_bwd(src_d, stat_d, diff_src_d, diff_dst_d));
                }
            }

            return status::success;
        }

        std::unique_ptr<KDNN::NormalizationLayerBWD> kdnn_layer_normalization_prim_;
    }; // pd_t

    kdnn_layer_normalization_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_layer_normalization_bwd_resource_t>(pd()->kdnn_layer_normalization_prim_));

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
        auto mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        auto variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
        auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        auto scale = CTX_IN_MEM(const void *, DNNL_ARG_SCALE);
        auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        auto diff_scale =  pd()->use_scale()
            ? CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SCALE)
            : nullptr;
        auto diff_shift = pd()->use_shift()
            ? CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SHIFT)
            : nullptr;

        return execute_backward(ctx, src, mean, variance, diff_dst,
                scale, diff_src, diff_scale, diff_shift);
    }

    status_t execute_backward(const exec_ctx_t &ctx, const void *src,
            const float *mean, const float *variance, const void *diff_dst,
            const void *scale, void *diff_src, void *diff_scale,
            void *diff_shift) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::NormalizationLayerBWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_layer_normalization_bwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, mean, variance, diff_dst,
                scale, diff_src, diff_scale, diff_shift, !pd()->use_global_stats(), pd()->desc()->layer_norm_epsilon);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_layer_normalization_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_LAYER_NORMALIZATION_HPP
