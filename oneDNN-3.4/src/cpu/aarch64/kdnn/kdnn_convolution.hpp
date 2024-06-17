#ifndef CPU_AARCH64_KDNN_CONVOLUTION_HPP
#define CPU_AARCH64_KDNN_CONVOLUTION_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_convolution_fwd_resource_t : public resource_t {
    kdnn_convolution_fwd_resource_t(const std::unique_ptr<KDNN::ConvolutionLayerFWD> &kdnn_convolution_prim) noexcept
        : kdnn_convolution_fwd_obj_(new KDNN::ConvolutionLayerFWD{*(kdnn_convolution_prim.get())}) {}

    KDNN::ConvolutionLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_convolution_fwd_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_convolution_fwd_resource_t);

private:
    std::unique_ptr<KDNN::ConvolutionLayerFWD> kdnn_convolution_fwd_obj_;
}; // kdnn_convolution_fwd_resource_t

struct kdnn_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        pd_t(const pd_t& p) : cpu_convolution_fwd_pd_t(p) {
            if (p.kdnn_convolution_prim_) {
                this->kdnn_convolution_prim_.reset(new KDNN::ConvolutionLayerFWD{*(p.kdnn_convolution_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_convolution_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = is_fwd()
                && set_default_formats()
                && !has_zero_dim_memory()
                && attr()->has_default_values()
                && with_groups() == false
                && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper bias_d(weights_md(1));

            if (!kdnn_utils::is_data_type_supported_by_kdnn(src_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(weights_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(dst_d.data_type()) ||
                ((bias_d != &glob_zero_md) && !kdnn_utils::is_data_type_supported_by_kdnn(bias_d.data_type()))) {
                    return status::unimplemented;
            }
            if (src_d.ndims() < 1 || src_d.ndims() > 5 ||
                weights_d.ndims() < 1 || weights_d.ndims() > 5 ||
                dst_d.ndims() < 1 || dst_d.ndims() > 5 ||
                ((bias_d != &glob_zero_md) && (bias_d.ndims() < 1 || bias_d.ndims() > 5))) {
                return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(src_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(weights_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(dst_d) || 
                ((bias_d != &glob_zero_md) && !kdnn_utils::is_data_layout_supported_by_kdnn(bias_d))) {
                    return status::unimplemented;
            }

            if (!kdnn_utils::may_convert_to_kdnn_conv_fwd(src_d,
                weights_d, dst_d, bias_d, *desc(), desc_.alg_kind)) {
                return status::unimplemented;
            } else {
                kdnn_convolution_prim_.reset(kdnn_utils::convert_to_kdnn_conv_fwd(src_d,
                    weights_d, dst_d, bias_d, *desc(), desc_.alg_kind));
            }

            return status::success;
        }

        std::unique_ptr<KDNN::ConvolutionLayerFWD> kdnn_convolution_prim_;

    protected:

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

    }; // pd_t

    kdnn_convolution_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_convolution_fwd_resource_t>(pd()->kdnn_convolution_prim_));

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
        const void *bia = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, wei, dst, bia);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst, const void *bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::ConvolutionLayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_convolution_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, wei, dst, bias);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

}; // kdnn_convolution_fwd_t

struct kdnn_convolution_bwd_data_resource_t : public resource_t {
    kdnn_convolution_bwd_data_resource_t(const std::unique_ptr<KDNN::ConvolutionLayerBWDData> &kdnn_convolution_prim) noexcept
        : kdnn_convolution_bwd_data_obj_(new KDNN::ConvolutionLayerBWDData{*(kdnn_convolution_prim.get())}) {}

    KDNN::ConvolutionLayerBWDData &get_kdnn_obj() const noexcept { return *kdnn_convolution_bwd_data_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_convolution_bwd_data_resource_t);

private:
    std::unique_ptr<KDNN::ConvolutionLayerBWDData> kdnn_convolution_bwd_data_obj_;
}; // kdnn_convolution_bwd_data_resource_t

struct kdnn_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        pd_t(const pd_t& p) : cpu_convolution_bwd_data_pd_t(p) {
            if (p.kdnn_convolution_prim_) {
                this->kdnn_convolution_prim_.reset(new KDNN::ConvolutionLayerBWDData{*(p.kdnn_convolution_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            const bool ok = desc()->prop_kind == prop_kind::backward_data
                && set_default_formats()
                && !has_zero_dim_memory()
                && with_groups() == false
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());

            if (!kdnn_utils::is_data_type_supported_by_kdnn(diff_dst_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(weights_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(diff_src_d.data_type())) {
                    return status::unimplemented;
            }
            if (diff_dst_d.ndims() < 1 || diff_dst_d.ndims() > 5 ||
                weights_d.ndims() < 1 || weights_d.ndims() > 5 ||
                diff_src_d.ndims() < 1 || diff_src_d.ndims() > 5) {
                return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(diff_dst_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(weights_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(diff_src_d)) {
                    return status::unimplemented;
            }

            if (!kdnn_utils::may_convert_to_kdnn_conv_bwd_data(diff_dst_d,
                    weights_d, diff_src_d, *desc(), desc_.alg_kind)) {
                return status::unimplemented;
            } else {
                kdnn_convolution_prim_.reset(kdnn_utils::convert_to_kdnn_conv_bwd_data(diff_dst_d,
                    weights_d, diff_src_d, *desc(), desc_.alg_kind));
            }

            return status::success;
        }

        std::unique_ptr<KDNN::ConvolutionLayerBWDData> kdnn_convolution_prim_;

    protected:

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

    }; // pd_t

    kdnn_convolution_bwd_data_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_convolution_bwd_data_resource_t>(
            pd()->kdnn_convolution_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward_data(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        return execute_backward_data(ctx, diff_dst, wei, diff_src);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_backward_data(const exec_ctx_t &ctx, const void *diff_dst,
            const void *wei, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::ConvolutionLayerBWDData &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_convolution_bwd_data_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, wei, diff_src);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

}; // kdnn_convolution_bwd_data_t

struct kdnn_convolution_bwd_weights_resource_t : public resource_t {
    kdnn_convolution_bwd_weights_resource_t(const std::unique_ptr<KDNN::ConvolutionLayerBWDWeights> &kdnn_convolution_prim) noexcept
        : kdnn_convolution_bwd_weights_obj_(new KDNN::ConvolutionLayerBWDWeights{*(kdnn_convolution_prim.get())}) {}

    KDNN::ConvolutionLayerBWDWeights &get_kdnn_obj() const noexcept { return *kdnn_convolution_bwd_weights_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_convolution_bwd_weights_resource_t);

private:
    std::unique_ptr<KDNN::ConvolutionLayerBWDWeights> kdnn_convolution_bwd_weights_obj_;
}; // kdnn_convolution_bwd_weights_resource_t

struct kdnn_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::cpu_convolution_bwd_weights_pd_t;

        pd_t(const pd_t& p) : cpu_convolution_bwd_weights_pd_t(p) {
            if (p.kdnn_convolution_prim_) {
                this->kdnn_convolution_prim_.reset(new KDNN::ConvolutionLayerBWDWeights{*(p.kdnn_convolution_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            const bool ok = desc()->prop_kind == prop_kind::backward_weights
                && set_default_formats()
                && !has_zero_dim_memory()
                && with_groups() == false
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper src(src_md());
            const memory_desc_wrapper diff_wei_d(diff_weights_md(0));
            const memory_desc_wrapper diff_bias_d(diff_weights_md(1));
            if (!kdnn_utils::is_data_type_supported_by_kdnn(diff_dst_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(src.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(diff_wei_d.data_type()) ||
                ((diff_bias_d != &glob_zero_md) && !kdnn_utils::is_data_type_supported_by_kdnn(diff_bias_d.data_type()))) {
                    return status::unimplemented;
            }
            if (diff_dst_d.ndims() < 1 || diff_dst_d.ndims() > 5 ||
                src.ndims() < 1 || src.ndims() > 5 ||
                diff_wei_d.ndims() < 1 || diff_wei_d.ndims() > 5 ||
                ((diff_bias_d != &glob_zero_md) && (diff_bias_d.ndims() < 1 || diff_bias_d.ndims() > 5))) {
                return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(diff_dst_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(src) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(diff_wei_d) ||
                ((diff_bias_d != &glob_zero_md) && (!kdnn_utils::is_data_layout_supported_by_kdnn(diff_bias_d)))) {
                    return status::unimplemented;
            }
            if (!kdnn_utils::may_convert_to_kdnn_conv_bwd_weights(diff_dst_d,
                    src, diff_wei_d, diff_bias_d, *desc(), desc_.alg_kind)) {
                return status::unimplemented;
            } else {
                kdnn_convolution_prim_.reset(kdnn_utils::convert_to_kdnn_conv_bwd_weights(diff_dst_d,
                    src, diff_wei_d, diff_bias_d, *desc(), desc_.alg_kind));
            }
            return status::success;
        }

        std::unique_ptr<KDNN::ConvolutionLayerBWDWeights> kdnn_convolution_prim_;

    protected:

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

    }; // pd_t

    kdnn_convolution_bwd_weights_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_convolution_bwd_weights_resource_t>(pd()->kdnn_convolution_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward_weights(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *diff_wei = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);
        void *diff_bia = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        return execute_backward_weights(ctx, diff_dst, src, diff_wei, diff_bia);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_backward_weights(const exec_ctx_t &ctx, const void *diff_dst,
            const void *src, void *diff_wei, void *diff_bia) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::ConvolutionLayerBWDWeights &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_convolution_bwd_weights_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, src, diff_wei, diff_bia);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

}; // kdnn_convolution_bwd_weights_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_CONVOLUTION_HPP
