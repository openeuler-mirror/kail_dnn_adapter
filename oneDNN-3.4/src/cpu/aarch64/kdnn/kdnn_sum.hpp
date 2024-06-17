#ifndef CPU_AARCH64_KDNN_SUM_HPP
#define CPU_AARCH64_KDNN_SUM_HPP

#include "kdnn.hpp"

#include "cpu/cpu_sum_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_sum_resource_t : public resource_t {
    kdnn_sum_resource_t(const std::unique_ptr<KDNN::SumLayer> &kdnn_sum_prim) noexcept
        : kdnn_sum_obj_(new KDNN::SumLayer{*(kdnn_sum_prim.get())}) {}

    KDNN::SumLayer &get_kdnn_obj() const noexcept { return *kdnn_sum_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_sum_resource_t);

private:
    std::unique_ptr<KDNN::SumLayer> kdnn_sum_obj_;
}; // kdnn_sum_resource_t

struct kdnn_sum_t : public primitive_t {
    struct pd_t : public cpu_sum_pd_t {
        using cpu_sum_pd_t::cpu_sum_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_sum_pd_t(p) {
            if (p.kdnn_sum_prim_) {
                this->kdnn_sum_prim_.reset(new KDNN::SumLayer{*(p.kdnn_sum_prim_.get())});
            }
        }

        DECLARE_SUM_PD_T("kdnn", kdnn_sum_t);
        
        status_t init(engine_t *engine) {
            bool ok = true && set_default_params() == status::success;
                if (!ok) return status::unimplemented;

            const memory_desc_wrapper dst_d(dst_md());
            if (!kdnn_utils::is_data_type_supported_by_kdnn(dst_d.data_type())) {
                return status::unimplemented;
            }
            
            if (dst_d.ndims() < 1 || dst_d.ndims() > 5) {
                return status::unimplemented;
            }

            if (!kdnn_utils::is_data_layout_supported_by_kdnn(dst_d)) {
                return status::unimplemented;
            }

            const int n = n_inputs();
            std::vector<memory_desc_wrapper> src_vec_d;
            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper src_i_d(src_md(i));
                bool ok = kdnn_utils::is_data_type_supported_by_kdnn(src_i_d.data_type())
                     && src_i_d.ndims() >= 1 && src_i_d.ndims() <= 5
                     && kdnn_utils::is_data_layout_supported_by_kdnn(src_i_d);
                if (!ok) return status::unimplemented;
                src_vec_d.push_back(src_i_d);
            }

            const float *scl = scales();
            if (!kdnn_utils::may_convert_to_kdnn_sum(src_vec_d, scl, dst_d)) {
                return status::unimplemented;
            } else {
                kdnn_sum_prim_.reset(kdnn_utils::convert_to_kdnn_sum(src_vec_d, scl, dst_d));
            }

            return status::success;
        }

        std::unique_ptr<KDNN::SumLayer> kdnn_sum_prim_;
    }; // pd_t

    kdnn_sum_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_sum_resource_t>(pd()->kdnn_sum_prim_));

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
 
        KDNN::SumLayer &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_sum_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, dst);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_sum_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_SUM_HPP
