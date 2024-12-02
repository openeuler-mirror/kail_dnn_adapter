#ifndef CPU_AARCH64_KDNN_SOFTMAX_HPP
#define CPU_AARCH64_KDNN_SOFTMAX_HPP

#include "kdnn.hpp"

#include "cpu/cpu_softmax_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_softmax_fwd_resource_t : public resource_t {
    kdnn_softmax_fwd_resource_t(const std::unique_ptr<KDNN::SoftmaxLayerFWD> &kdnn_softmax_prim) noexcept
        : kdnn_softmax_obj_(new KDNN::SoftmaxLayerFWD{*(kdnn_softmax_prim.get())}) {}

    KDNN::SoftmaxLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_softmax_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_softmax_fwd_resource_t);

private:
    std::unique_ptr<KDNN::SoftmaxLayerFWD> kdnn_softmax_obj_;
}; // kdnn_softmax_fwd_resource_t

struct kdnn_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_softmax_fwd_pd_t(p) {
            if (p.kdnn_softmax_prim_) {
                this->kdnn_softmax_prim_.reset(new KDNN::SoftmaxLayerFWD{*(p.kdnn_softmax_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_softmax_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = is_fwd() && attr()->has_default_values() && (set_default_formats() == status::success);
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            auto&& softmax_fwd = kdnn_utils::convert_to_kdnn_softmax(src_d, dst_d, axis(), is_logsoftmax());
            if (!softmax_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_softmax_prim_.reset(softmax_fwd.second);
                return status::success;
            }
        }
        std::unique_ptr<KDNN::SoftmaxLayerFWD> kdnn_softmax_prim_;
    };

    kdnn_softmax_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_softmax_fwd_resource_t>(pd()->kdnn_softmax_prim_));

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
 
        KDNN::SoftmaxLayerFWD &kdnn_obj = (ctx.get_resource_mapper()->get<kdnn_softmax_fwd_resource_t>(this))->get_kdnn_obj();
        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // kdnn_softmax_fwd_t

struct kdnn_softmax_bwd_resource_t : public resource_t {
    kdnn_softmax_bwd_resource_t(const std::unique_ptr<KDNN::SoftmaxLayerBWD> &kdnn_softmax_prim) noexcept
        : kdnn_softmax_obj_(new KDNN::SoftmaxLayerBWD{*(kdnn_softmax_prim.get())}) {}

    KDNN::SoftmaxLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_softmax_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_softmax_bwd_resource_t);

private:
    std::unique_ptr<KDNN::SoftmaxLayerBWD> kdnn_softmax_obj_;
}; // kdnn_softmax_bwd_resource_t

struct kdnn_softmax_bwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_softmax_bwd_pd_t(p) {
            if (p.kdnn_softmax_prim_) {
                this->kdnn_softmax_prim_.reset(new KDNN::SoftmaxLayerBWD{*(p.kdnn_softmax_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_softmax_bwd_t);

        status_t init(engine_t *engine) {
            bool ok = !is_fwd() && attr()->has_default_values() && (set_default_formats() == status::success);
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            auto&& softmax_bwd = kdnn_utils::convert_to_kdnn_softmax(dst_d, diff_dst_d, diff_src_d, axis(), is_logsoftmax());
            if (!softmax_bwd.first) {
                return status::unimplemented;
            } else {
                kdnn_softmax_prim_.reset(softmax_bwd.second);
                return status::success;
            }
        }
        std::unique_ptr<KDNN::SoftmaxLayerBWD> kdnn_softmax_prim_;
    };

    kdnn_softmax_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_softmax_bwd_resource_t>(pd()->kdnn_softmax_prim_));

        return status::success;
    }

private:
    // execute_backward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_backward(const exec_ctx_t &ctx) const {

        const void *dst = CTX_IN_MEM(const void *, DNNL_ARG_DST);
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        return execute_backward(ctx, dst, diff_dst, diff_src);
    }

    // Execute backward with arbitrary src and dst
    status_t execute_backward(
            const exec_ctx_t &ctx, const void *dst, const void *diff_dst, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::SoftmaxLayerBWD &kdnn_obj = (ctx.get_resource_mapper()->get<kdnn_softmax_bwd_resource_t>(this))->get_kdnn_obj();
        try {
            kdnn_obj.Run(dst, diff_dst, diff_src);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // kdnn_softmax_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_ELTWISE_HPP
