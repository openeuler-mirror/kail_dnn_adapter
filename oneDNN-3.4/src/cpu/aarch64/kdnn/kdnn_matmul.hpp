#ifndef CPU_AARCH64_KDNN_MATMUL_HPP
#define CPU_AARCH64_KDNN_MATMUL_HPP

#include "kdnn.hpp"

#include "common/utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_post_ops.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

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
                this->post_ops_ = p.post_ops_;
                this->need_tmp_dst_ = p.need_tmp_dst_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_matmul_t);
        
        status_t init(engine_t *engine) {
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            bool ok = !has_zero_dim_memory() &&
                attr()->has_default_values(dnnl_primitive_attr::skip_mask_t::post_ops |
                    dnnl_primitive_attr::skip_mask_t::scales_runtime) &&
                attr_scales_ok() &&
                (attr()->output_scales_.mask_ == 0) &&
                (attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ == 0) &&
                !has_runtime_dims_or_strides();
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper wei_d(&weights_md_);
            const memory_desc_wrapper dst_d(&dst_md_);
            const memory_desc_wrapper bias_d(&bias_md_);
            auto&& matmul = kdnn_utils::convert_to_kdnn_gemm(src_d, wei_d, dst_d, bias_d);
            if (!matmul.first) {
                return status::unimplemented;
            } else {
                kdnn_matmul_prim_.reset(matmul.second);
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

        bool attr_scales_ok(
                const std::vector<int> &supported_args = {DNNL_ARG_SRC,
                        DNNL_ARG_WEIGHTS}) const override {
            if (attr()->scales_.has_default_values()) return true;

            bool ok = attr()->scales_.has_default_values(supported_args);
            for (int arg : supported_args) {
                const auto &sc = attr()->scales_.get(arg);
                const auto &mask = sc.mask_;
                if (!sc.has_default_values()) {
                    if (arg == DNNL_ARG_WEIGHTS) {
                        ok = ok
                                && utils::one_of(mask, 0, wei_qmask_N(),
                                        wei_qmask_N() + wei_qmask_K());
                        ok = ok && utils::one_of(sc.ndims_, 0, 2)
                                && IMPLICATION(sc.ndims_ == 2,
                                        sc.group_dims_[1] == 1
                                                && K() % sc.group_dims_[0]
                                                        == 0);
                    } else
                        ok = ok && (mask == 0);
                }
            }
            return ok;
        }

        bool need_tmp_dst_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;
        
        std::unique_ptr<KDNN::Gemm<KDNN::GemmPack::NO_PACK>> kdnn_matmul_prim_;

    }; // pd_t

    kdnn_matmul_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_matmul_resource_t>(pd()->kdnn_matmul_prim_));
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
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        void *dst;

        if (pd()->need_tmp_dst_) {
            dst = KDNN::Service::AlignedAlloc(pd()->dst_size_);
        } else {
            dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        }

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
            DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
            DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);

            auto scratchpad = ctx.get_scratchpad_grantor();
            
            const int ndims = pd()->ndims();
            const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
            const float *scales = precompute_scales(scratchpad, src_scales, wei_scales,
                    dst_d.dims()[ndims - 1], pd()->attr());

            kdnn_obj.Run(src, wei, dst, bias, scales[0], 0.0f);
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
}; // kdnn_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_MATMUL_HPP
