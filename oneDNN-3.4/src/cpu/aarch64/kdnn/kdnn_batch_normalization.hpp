#ifndef CPU_AARCH64_KDNN_BATCH_NORMALIZATION_HPP
#define CPU_AARCH64_KDNN_BATCH_NORMALIZATION_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/cpu_batch_normalization_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_batch_normalization_fwd_resource_t : public resource_t {
    kdnn_batch_normalization_fwd_resource_t(const std::unique_ptr<KDNN::BatchNormalizationLayerFWD> &kdnn_batch_normalization_prim) noexcept
        : kdnn_batch_normalization_obj_(new KDNN::BatchNormalizationLayerFWD{*(kdnn_batch_normalization_prim.get())}) {}

    KDNN::BatchNormalizationLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_batch_normalization_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_batch_normalization_fwd_resource_t);

private:
    std::unique_ptr<KDNN::BatchNormalizationLayerFWD> kdnn_batch_normalization_obj_;
}; // kdnn_batch_normalization_fwd_resource_t

struct kdnn_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        using cpu_batch_normalization_fwd_pd_t::
                cpu_batch_normalization_fwd_pd_t;
        
        pd_t(const pd_t& p) : cpu_batch_normalization_fwd_pd_t(p) {
            if (p.kdnn_batch_normalization_prim_) {
                this->kdnn_batch_normalization_prim_.reset(new KDNN::BatchNormalizationLayerFWD{*(p.kdnn_batch_normalization_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_tmp_dst_ = p.need_tmp_dst_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(src_md_);
            const memory_desc_wrapper dst_d(dst_md_);
            const memory_desc_wrapper stats_d(stat_md_);
            const memory_desc_wrapper scaleshift_d(scaleshift_md_);

            if (!set_default_formats_common()) {
                return status::unimplemented;
            }

            // fusing is not supported yet
            if (fuse_norm_add_relu()) {
                return status::unimplemented;
            }

            // needed s8 dtype support from eltwise
            if (with_relu_post_op() && dst_d.data_type() == data_type_t::dnnl_s8) {
                return status::unimplemented;
            }

            if (is_training() && fuse_norm_relu()) {
                init_default_ws(8);
            }

            auto&& bnorm_fwd = kdnn_utils::convert_to_kdnn_batch_normalization_fwd(src_d, dst_d, stats_d,
                scaleshift_d, use_global_stats(), use_scale(), use_shift(), C(), fuse_norm_relu(), is_training());
            if (!bnorm_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_batch_normalization_prim_.reset(bnorm_fwd.second);
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

        std::unique_ptr<KDNN::BatchNormalizationLayerFWD> kdnn_batch_normalization_prim_;
    }; // pd_t

    kdnn_batch_normalization_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_batch_normalization_fwd_resource_t>(pd()->kdnn_batch_normalization_prim_));
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
        auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *dst;

        if (pd()->need_tmp_dst_) {
            dst = KDNN::Service::AlignedAlloc(pd()->dst_size_);
        } else {
            dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        }

        auto mean = CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        auto variance = CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);

        const float *scale = nullptr;
        const float *shift = nullptr;
        if (pd()->use_scale()) {
            scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
        }
        if (pd()->use_shift()) {
            shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);
        }

        const float eps = pd()->desc()->batch_norm_epsilon;
        const bool save_stats = pd()->is_training();

        std::uint8_t *ws = nullptr;
        if (pd()->fuse_norm_relu()) {
            status_t s = status_t::dnnl_success;
            ws = CTX_OUT_CLEAN_MEM(uint8_t *, DNNL_ARG_WORKSPACE, s);
            CHECK(s);
        }

        return execute_forward(ctx, src, dst, scale, shift,
                mean, variance, save_stats, eps, ws);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            void *dst, const float *scale, const float *shift,
            float *mean, float *variance,
            bool save_stats, const float eps, std::uint8_t *ws) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::BatchNormalizationLayerFWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_batch_normalization_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst, scale, shift, mean, variance, save_stats, eps, ws);
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
}; // kdnn_batch_normalization_fwd_t

struct kdnn_batch_normalization_bwd_resource_t : public resource_t {
    kdnn_batch_normalization_bwd_resource_t(const std::unique_ptr<KDNN::BatchNormalizationLayerBWD> &kdnn_batch_normalization_prim) noexcept
        : kdnn_batch_normalization_obj_(new KDNN::BatchNormalizationLayerBWD{*(kdnn_batch_normalization_prim.get())}) {}

    KDNN::BatchNormalizationLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_batch_normalization_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_batch_normalization_bwd_resource_t);

private:
    std::unique_ptr<KDNN::BatchNormalizationLayerBWD> kdnn_batch_normalization_obj_;
}; // kdnn_batch_normalization_bwd_resource_t

struct kdnn_batch_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        using cpu_batch_normalization_bwd_pd_t::
                cpu_batch_normalization_bwd_pd_t;

        pd_t(const pd_t& p) : cpu_batch_normalization_bwd_pd_t(p) {
            if (p.kdnn_batch_normalization_prim_) {
                this->kdnn_batch_normalization_prim_.reset(new KDNN::BatchNormalizationLayerBWD{*(p.kdnn_batch_normalization_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_batch_normalization_bwd_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(src_md_);
            const memory_desc_wrapper diff_dst_d(diff_dst_md_);
            const memory_desc_wrapper diff_src_d(diff_src_md_);
            const memory_desc_wrapper stats_d(stat_md_);
            const memory_desc_wrapper scaleshift_d(scaleshift_md_);
            const memory_desc_wrapper diff_ss_d(diff_scaleshift_md_);

            if (!set_default_formats_common()) {
                return status::unimplemented;
            }

            if (fuse_norm_add_relu()) {
                return status::unimplemented;
            }

            if (fuse_norm_relu()) {
                init_default_ws(8);
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            auto&& bnorm_bwd = kdnn_utils::convert_to_kdnn_batch_normalization_bwd(src_d, diff_dst_d, stats_d,
                scaleshift_d, diff_src_d, diff_ss_d, use_global_stats(), use_scale(), use_shift(), C(), fuse_norm_relu());
            if (!bnorm_bwd.first) {
                return status::unimplemented;
            } else {
                kdnn_batch_normalization_prim_.reset(bnorm_bwd.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::BatchNormalizationLayerBWD> kdnn_batch_normalization_prim_;
    }; // pd_t

    kdnn_batch_normalization_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        mapper.add(this, std::make_unique<kdnn_batch_normalization_bwd_resource_t>(pd()->kdnn_batch_normalization_prim_));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward(const exec_ctx_t &ctx) const {
        auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        auto mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        auto variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);

        auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        const float *scale = nullptr;
        float *diff_scale = nullptr;
        float *diff_shift = nullptr;
        if (pd()->use_scale()) {
            scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
            diff_scale = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SCALE);
        }
        if (pd()->use_shift()) {
            diff_shift = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SHIFT);
        }

        const std::uint8_t *ws = nullptr;
        if (pd()->fuse_norm_relu()) {
            ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);
        }

        const float eps = pd()->desc()->batch_norm_epsilon;

        return execute_backward(ctx, src, diff_dst, mean, variance, scale, diff_src,
            diff_scale, diff_shift, eps, ws);
    }

    status_t execute_backward(const exec_ctx_t &ctx, const void *src,
            const void *diff_dst, const float *mean, const float *variance,
            const float *scale, void *diff_src, float *diff_scale,
            float *diff_shift, const float eps, const std::uint8_t *ws) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::BatchNormalizationLayerBWD &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_batch_normalization_bwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, diff_dst, mean, variance, scale, diff_src,
                diff_scale, diff_shift, eps, ws);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_batch_normalization_bwd_t


} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_BATCH_NORMALIZATION_HPP
