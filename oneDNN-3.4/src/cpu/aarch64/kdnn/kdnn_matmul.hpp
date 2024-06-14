#ifndef CPU_AARCH64_KDNN_MATMUL_HPP
#define CPU_AARCH64_KDNN_MATMUL_HPP

#include "kdnn.hpp"

#include "common/utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct kdnn_matmul_resource_t : public resource_t {
    kdnn_matmul_resource_t(const std::unique_ptr<KDNN::Gemm<KDNN::GemmPack::NO_PACK>> &kdnn_matmul_prim) noexcept
        : kdnn_matmul_obj_(new KDNN::Gemm<KDNN::GemmPack::NO_PACK>{*(kdnn_matmul_prim.get())}) {}

    KDNN::Gemm<KDNN::GemmPack::NO_PACK> &get_kdnn_obj() const noexcept { return *kdnn_matmul_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_matmul_resource_t);

private:
    std::unique_ptr<KDNN::Gemm<KDNN::GemmPack::NO_PACK>> kdnn_matmul_obj_;
}; // kdnn_matmul_resource_t

struct kdnn_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_matmul_pd_t(p) {
            if (p.kdnn_matmul_prim_) {
                this->kdnn_matmul_prim_.reset(new KDNN::Gemm<KDNN::GemmPack::NO_PACK>{*(p.kdnn_matmul_prim_.get())});
                this->post_ops = p.post_ops;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_matmul_t);
        
        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper wei_d(&weights_md_);
            const memory_desc_wrapper dst_d(&dst_md_);
            const memory_desc_wrapper bias_d(&bias_md_);

            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            bool ok = !has_zero_dim_memory() &&
                attr()->has_default_values(dnnl_primitive_attr::skip_mask_t::oscale |
                dnnl_primitive_attr::skip_mask_t::post_ops) &&
                attr_scales_ok() &&
                (attr()->output_scales_.mask_ == 0) &&
                !has_runtime_dims_or_strides();

            if (!ok) return status::unimplemented;
            if (!kdnn_utils::is_data_type_supported_by_kdnn(src_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(wei_d.data_type()) ||
                !kdnn_utils::is_data_type_supported_by_kdnn(dst_d.data_type())) {
                    return status::unimplemented;
            }
            if (src_d.ndims() < 1 || src_d.ndims() > 5 ||
                wei_d.ndims() < 1 || wei_d.ndims() > 5 ||
                dst_d.ndims() < 1 || dst_d.ndims() > 5) {
                return status::unimplemented;
            }
            if (!kdnn_utils::is_data_layout_supported_by_kdnn(src_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(wei_d) ||
                !kdnn_utils::is_data_layout_supported_by_kdnn(dst_d)) {
                    return status::unimplemented;
            }
            if (with_bias()) {
                if (!kdnn_utils::is_data_type_supported_by_kdnn(bias_d.data_type())) {
                    return status::unimplemented;
                }
                if (bias_d.ndims() < 1 || bias_d.ndims() > 5) {
                    return status::unimplemented;
                }
                if (!kdnn_utils::is_data_layout_supported_by_kdnn(bias_d)) {
                    return status::unimplemented;
                }
                if (!kdnn_utils::may_convert_to_kdnn_gemm(src_d, wei_d, dst_d, bias_d)) {
                    return status::unimplemented;
                } else {
                    kdnn_matmul_prim_.reset(kdnn_utils::convert_to_kdnn_gemm(src_d, wei_d, dst_d, bias_d));
                }
            } else {
                if (!kdnn_utils::may_convert_to_kdnn_gemm(src_d, wei_d, dst_d)) {
                    return status::unimplemented;
                } else {
                    kdnn_matmul_prim_.reset(kdnn_utils::convert_to_kdnn_gemm(src_d, wei_d, dst_d));
                }
            }

            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_));

            return status::success;
        }

        kdnn_post_ops_t post_ops;
        
        std::unique_ptr<KDNN::Gemm<KDNN::GemmPack::NO_PACK>> kdnn_matmul_prim_;

    }; // pd_t

    kdnn_matmul_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_matmul_resource_t>(pd()->kdnn_matmul_prim_));
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
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);

        return execute_forward(ctx, src, wei, dst, bias);
    }

    // Execute forward with arbitrary src, wei, bias and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst, const void *bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::Gemm<KDNN::GemmPack::NO_PACK> &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_matmul_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, wei, dst, bias);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        pd()->post_ops.execute(ctx, dst);

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_MATMUL_HPP
