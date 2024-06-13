#ifndef CPU_AARCH64_KDNN_DECONVOLUTION_HPP
#define CPU_AARCH64_KDNN_DECONVOLUTION_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_deconvolution_fwd_resource_t : public resource_t {
    kdnn_deconvolution_fwd_resource_t(const std::unique_ptr<KDNN::DeconvolutionLayerFWD> &kdnn_deconvolution_prim) noexcept
        : kdnn_deconvolution_fwd_obj_(new KDNN::DeconvolutionLayerFWD{*(kdnn_deconvolution_prim.get())}) {}

    KDNN::DeconvolutionLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_deconvolution_fwd_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_deconvolution_fwd_resource_t);

private:
    std::unique_ptr<KDNN::DeconvolutionLayerFWD> kdnn_deconvolution_fwd_obj_;
}; // kdnn_deconvolution_fwd_resource_t

struct kdnn_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        pd_t(const pd_t& p) : cpu_deconvolution_fwd_pd_t(p) {
            if (p.kdnn_deconvolution_prim_) {
                this->kdnn_deconvolution_prim_.reset(new KDNN::DeconvolutionLayerFWD{*(p.kdnn_deconvolution_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_deconvolution_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = is_fwd() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            // Set memory formats
            const memory_desc_wrapper check_src_d(src_md());
            const memory_desc_wrapper check_weights_d(weights_md(0));
            const memory_desc_wrapper check_dst_d(dst_md());
            const memory_desc_wrapper check_bias_d(weights_md(1));
            using namespace format_tag;
            auto src_tag = check_src_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                    : memory_desc_matches_one_of_tag(src_md_, ncdhw, nchw, ncw);
            auto dst_tag = check_dst_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                    : memory_desc_matches_one_of_tag(dst_md_, ncdhw, nchw, ncw);
            auto wei_tag = check_weights_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, oiw, oihw, oidhw)
                    : memory_desc_matches_one_of_tag(weights_md_, oidhw, oihw, oiw);
            CHECK(memory_desc_init_by_tag(src_md_, src_tag));
            CHECK(memory_desc_init_by_tag(dst_md_, dst_tag));
            CHECK(memory_desc_init_by_tag(weights_md_, wei_tag));
            if (check_bias_d != &glob_zero_md) {
                CHECK(memory_desc_init_by_tag(bias_md_, x));
            }

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper weights_d(&weights_md_);
            const memory_desc_wrapper dst_d(&dst_md_);
            const memory_desc_wrapper bias_d(&bias_md_);
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
            if (!kdnn_utils::may_convert_to_kdnn_deconv_fwd(src_d, weights_d, dst_d, bias_d,
                    *desc(), desc_.alg_kind)) {
                 return status::unimplemented;
            } else {
                kdnn_deconvolution_prim_.reset(kdnn_utils::convert_to_kdnn_deconv_fwd(src_d,
                    weights_d, dst_d, bias_d, *desc(), desc_.alg_kind));
            }

            return status::success;
        }

        std::unique_ptr<KDNN::DeconvolutionLayerFWD> kdnn_deconvolution_prim_;

    }; // pd_t

    kdnn_deconvolution_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
        engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_deconvolution_fwd_resource_t>(pd()->kdnn_deconvolution_prim_));

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

        KDNN::DeconvolutionLayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_deconvolution_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, wei, dst, bias);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

}; // kdnn_deconvolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_DECONVOLUTION_HPP
