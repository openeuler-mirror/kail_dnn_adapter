#ifndef CPU_AARCH64_KDNN_SHUFFLE_HPP
#define CPU_AARCH64_KDNN_SHUFFLE_HPP
 
#include <memory>
 
#include "kdnn.hpp"
 
#include "cpu/cpu_shuffle_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
 
namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
 
struct kdnn_shuffle_resource_t : public resource_t {
    kdnn_shuffle_resource_t(const std::unique_ptr<KDNN::ShuffleLayer> &kdnn_shuffle_prim) noexcept
        : kdnn_shuffle_obj_(new KDNN::ShuffleLayer{*(kdnn_shuffle_prim.get())}) {}
 
    KDNN::ShuffleLayer &get_kdnn_obj() const noexcept { return *kdnn_shuffle_obj_; }
 
    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_shuffle_resource_t);
 
private:
    std::unique_ptr<KDNN::ShuffleLayer> kdnn_shuffle_obj_;
}; // kdnn_shuffle_resource_t
 
struct kdnn_shuffle_t : public primitive_t {
    struct pd_t : public cpu_shuffle_pd_t {
        using cpu_shuffle_pd_t::cpu_shuffle_pd_t;
 
        pd_t(const pd_t& p) : cpu_shuffle_pd_t(p) {
            if (p.kdnn_shuffle_prim_) {
                this->kdnn_shuffle_prim_.reset(new KDNN::ShuffleLayer{*(p.kdnn_shuffle_prim_.get())});
            }
        }
 
        DECLARE_COMMON_PD_T("kdnn", kdnn_shuffle_t);
 
        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(
                    is_fwd() ? src_md() : diff_src_md());
            const memory_desc_wrapper dst_d(
                    is_fwd() ? dst_md() : diff_dst_md());
 
            bool ok = src_d.data_type() == dst_d.data_type()
                    && platform::has_data_type_support(src_d.data_type())
                    && attr()->has_default_values()
                    && set_default_formats_common() && src_d == dst_d;
            if (!ok) return status::unimplemented;
 
            auto groupSize = is_fwd() ? desc_.group_size : src_d.dims()[axis()] / desc_.group_size;
            auto&& shuffle = kdnn_utils::convert_to_kdnn_shuffle(src_d, dst_d, axis(), groupSize);
            if (!shuffle.first) {
                return status::unimplemented;
            } else {
                kdnn_shuffle_prim_.reset(shuffle.second);
                return status::success;
            }
        }
 
        std::unique_ptr<KDNN::ShuffleLayer> kdnn_shuffle_prim_;
 
        friend struct kdnn_post_ops_t;
    }; // pd_t
 
    kdnn_shuffle_t(const pd_t *kpd) : primitive_t(kpd) {}
 
    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
 
        mapper.add(this, std::make_unique<kdnn_shuffle_resource_t>(pd()->kdnn_shuffle_prim_));
 
        return status::success;
    }
 
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }
 
private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;
 
    status_t execute_forward(const exec_ctx_t &ctx) const {
        auto src =
            pd()->is_fwd() ? CTX_IN_MEM(const void *, DNNL_ARG_SRC) : CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        auto dst =
            pd()->is_fwd() ? CTX_OUT_MEM(void *, DNNL_ARG_DST) : CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
 
        return execute_forward(ctx, src, dst);
    }
 
    // Execute forward with arbitrary src
    status_t execute_forward(const exec_ctx_t &ctx, const void *src, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::ShuffleLayer &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_shuffle_resource_t>(this))->get_kdnn_obj();
 
        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }
 
        return status::success;
    }
 
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
 
    friend struct kdnn_post_ops_t;
}; // kdnn_shuffle_t
 
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
 
#endif // CPU_AARCH64_KDNN_SHUFFLE_HPP
