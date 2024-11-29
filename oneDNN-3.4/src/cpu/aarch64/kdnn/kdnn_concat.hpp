#ifndef CPU_AARCH64_KDNN_CONCAT_HPP
#define CPU_AARCH64_KDNN_CONCAT_HPP

#include "kdnn.hpp"

#include "cpu/cpu_concat_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_concat_resource_t : public resource_t {
    kdnn_concat_resource_t(const std::unique_ptr<KDNN::ConcatLayer> &kdnn_concat_prim) noexcept
        : kdnn_concat_obj_(new KDNN::ConcatLayer{*(kdnn_concat_prim.get())}) {}

    KDNN::ConcatLayer &get_kdnn_obj() const noexcept { return *kdnn_concat_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_concat_resource_t);

private:
    std::unique_ptr<KDNN::ConcatLayer> kdnn_concat_obj_;
}; // kdnn_concat_resource_t

struct kdnn_concat_t : public primitive_t {
    struct pd_t : public cpu_concat_pd_t {
        using cpu_concat_pd_t::cpu_concat_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_concat_pd_t(p) {
            if (p.kdnn_concat_prim_) {
                this->kdnn_concat_prim_.reset(new KDNN::ConcatLayer{*(p.kdnn_concat_prim_.get())});
            }
        }

        DECLARE_CONCAT_PD_T("kdnn", kdnn_concat_t);
        
        status_t init(engine_t *engine) {
            bool ok = (set_default_params() == status::success) && attr()->has_default_values();
                if (!ok) return status::unimplemented;

            const memory_desc_wrapper dst_d(dst_md());

            const int n = n_inputs();
            std::vector<memory_desc_wrapper> src_vec_d;
            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper src_i_d(src_md(i));
                src_vec_d.push_back(src_i_d);
            }

            const int concat_dim = this->concat_dim();
            auto&& concat = kdnn_utils::convert_to_kdnn_concat(src_vec_d, concat_dim, dst_d);
            if (!concat.first) {
                return status::unimplemented;
            } else {
                kdnn_concat_prim_.reset(concat.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::ConcatLayer> kdnn_concat_prim_;
    }; // pd_t

    kdnn_concat_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_concat_resource_t>(pd()->kdnn_concat_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        std::vector<const void *> srcVec;
        for (int a = 0; a < pd()->n_inputs(); ++a) {
            srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_MULTIPLE_SRC + a));
        }
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, srcVec.data(), dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void **src, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KDNN::ConcatLayer &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_concat_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_concat_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_CONCAT_HPP
