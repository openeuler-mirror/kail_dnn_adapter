/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: kdnn resampling header
 * Author: KPL
 * Create: 2024-07-04
 * Notes: NA
 */

#ifndef CPU_AARCH64_KDNN_RESAMPLING_HPP
#define CPU_AARCH64_KDNN_RESAMPLING_HPP

#include "kdnn.hpp"

#include "cpu/cpu_resampling_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include <iostream>
using namespace std;
namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_resampling_fwd_resource_t : public resource_t {
    kdnn_resampling_fwd_resource_t(const std::unique_ptr<KDNN::ResamplingLayerFWD> &kdnn_resampling_prim) noexcept
        : kdnn_resampling_obj_(new KDNN::ResamplingLayerFWD{*(kdnn_resampling_prim.get())}) {}

     KDNN::ResamplingLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_resampling_obj_; }

     DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_resampling_fwd_resource_t);
private:
    std::unique_ptr<KDNN::ResamplingLayerFWD> kdnn_resampling_obj_;
}; // kdnn_resampling_fwd_resource_t

struct kdnn_resampling_fwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_resampling_fwd_pd_t(p) {
            if (p.kdnn_resampling_prim_) {
                this->kdnn_resampling_prim_.reset(new KDNN::ResamplingLayerFWD{*(p.kdnn_resampling_prim_.get())});
            }
        }
        DECLARE_COMMON_PD_T("kdnn", kdnn_resampling_fwd_t);
        status_t init(engine_t *engine) {
            auto && first_non_any_md = kdnn_utils::get_first_non_any_format_kind(src_md_, dst_md_);
            auto&& layout = kdnn_utils::get_layout(first_non_any_md);
            if (src_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(src_md_, layout));
            }
            if (dst_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(dst_md_, layout));
            }

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            bool ok = is_fwd() && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;
            auto&& resampling_fwd = kdnn_utils::convert_to_kdnn_resampling(src_d, dst_d, desc_.alg_kind);
            if (!resampling_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_resampling_prim_.reset(resampling_fwd.second);
                return status::success;
            }
        }

        // We use `unique_ptr` because `resampling_layer_info` doesn't have default constructor
        std::unique_ptr<KDNN::ResamplingLayerFWD> kdnn_resampling_prim_;

        friend struct kdnn_post_ops_t;
    }; // pd_t

    kdnn_resampling_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_resampling_fwd_resource_t>(pd()->kdnn_resampling_prim_));

        return status::success;
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(
            const exec_ctx_t &ctx, const void *src, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::ResamplingLayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_resampling_fwd_resource_t>(this))->get_kdnn_obj();
        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    friend struct kdnn_post_ops_t;
}; // kdnn_resampling_fwd_t

struct kdnn_resampling_bwd_resource_t : public resource_t {
    kdnn_resampling_bwd_resource_t(const std::unique_ptr<KDNN::ResamplingLayerBWD> &kdnn_resampling_prim) noexcept
        : kdnn_resampling_obj_(new KDNN::ResamplingLayerBWD{*(kdnn_resampling_prim.get())}) {}

    KDNN::ResamplingLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_resampling_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_resampling_bwd_resource_t);
private:
    std::unique_ptr<KDNN::ResamplingLayerBWD> kdnn_resampling_obj_;
}; // kdnn_resampling_bwd_resource_t

struct kdnn_resampling_bwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_bwd_pd_t {
        using cpu_resampling_bwd_pd_t::cpu_resampling_bwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t &p) : cpu_resampling_bwd_pd_t(p) {
            if (p.kdnn_resampling_prim_) {
                this->kdnn_resampling_prim_.reset(new KDNN::ResamplingLayerBWD{*(p.kdnn_resampling_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_resampling_bwd_t);

        status_t init(engine_t *engine) {
            auto&& first_non_any_md = kdnn_utils::get_first_non_any_format_kind(diff_dst_md_, diff_src_md_);
            auto&& layout = kdnn_utils::get_layout(first_non_any_md);
            if (diff_dst_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(diff_dst_md_, layout));
            }
            if (diff_src_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(diff_src_md_, layout));
            }

            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());

            bool ok = (!is_fwd()) && (!has_zero_dim_memory()) && (attr()->has_default_values());
            if (!ok) { return status::unimplemented; }
            auto&& resampling_bwd =
                kdnn_utils::convert_to_kdnn_resampling_bwd(diff_dst_d, diff_src_d, desc_.alg_kind);
            if (!resampling_bwd.first) {
                return status::unimplemented;
            } else {
                kdnn_resampling_prim_.reset(resampling_bwd.second);
                return status::success;
            }
        }

        // We use `unique_ptr` because `resampling_layer_info` doesn't have default constructor
        std::unique_ptr<KDNN::ResamplingLayerBWD> kdnn_resampling_prim_;
        friend struct kdnn_post_ops_t;
    }; // pd_t

    kdnn_resampling_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

    status_t create_resource( engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) { return status::success; }
        mapper.add(this, std::make_unique<kdnn_resampling_bwd_resource_t>(pd()->kdnn_resampling_prim_));
        return status::success;
    }

private:
    // execute_backward has to be const thus mutability of mtx
    mutable std::mutex mtx_;

    status_t execute_backward(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
        return execute_backward(ctx, diff_dst, diff_src);
    }

    // Execute backward with arbitrary diff_dst and diff_src
    status_t execute_backward(const exec_ctx_t &ctx, const void *diff_dst, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock{mtx_};

        KDNN::ResamplingLayerBWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_resampling_bwd_resource_t>(this))->get_kdnn_obj();
        
        try {
            kdnn_obj.Run(diff_dst, diff_src);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    friend struct kdnn_post_ops_t;
}; // kdnn_resampling_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl


#endif
