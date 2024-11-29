#ifndef CPU_AARCH64_KDNN_REORDER_HPP
#define CPU_AARCH64_KDNN_REORDER_HPP

#include <memory>

#include "kdnn.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_reorder_resource_t : public resource_t {
    kdnn_reorder_resource_t(const std::unique_ptr<KDNN::ReorderLayer> &kdnn_reorder_prim) noexcept
        : kdnn_reorder_obj_(new KDNN::ReorderLayer{*(kdnn_reorder_prim.get())}) {}

    KDNN::ReorderLayer &get_kdnn_obj() const noexcept { return *kdnn_reorder_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_reorder_resource_t);

private:
    std::unique_ptr<KDNN::ReorderLayer> kdnn_reorder_obj_;
}; // kdnn_reorder_resource_t

struct kdnn_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        pd_t(const pd_t& p) : cpu_reorder_pd_t(p) {
            if (p.kdnn_reorder_prim_) {
                this->kdnn_reorder_prim_.reset(new KDNN::ReorderLayer{*(p.kdnn_reorder_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
            const primitive_attr_t *attr, engine_t *src_engine,
            const memory_desc_t *src_md, engine_t *dst_engine,
            const memory_desc_t *dst_md) {

            auto dst_d = memory_desc_wrapper(dst_md);
            const memory_desc_wrapper input_d(src_md);
            bool args_ok = attr->has_default_values()
                && !input_d.has_runtime_dims_or_strides()
                && !input_d.has_zero_dim()
                && (0 == dst_d.extra().compensation_mask)
                && (0 == dst_d.extra().asymm_compensation_mask);
            if (!args_ok) return status::invalid_arguments;

            int scales_mask = -1;
            bool is_set = false;
            CHECK(attr->scales_.get(DNNL_ARG_DST, &scales_mask, &is_set));
            if (is_set && scales_mask > 0) {
                return status::unimplemented;
            }

            const auto &post_ops = attr->post_ops_;
            if (post_ops.len() != 0) {
                return status::unimplemented;
            }

            auto _pd = make_unique_pd<pd_t>(attr, src_engine->kind(), src_md,
                dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                return status::unimplemented;
            }

            const memory_desc_wrapper output_d(dst_md);
            auto&& reorder = kdnn_utils::convert_to_kdnn_reorder(input_d, output_d);
            if (!reorder.first) {
                return status::unimplemented;
            } else {
                _pd->kdnn_reorder_prim_.reset(reorder.second);
            }

            CHECK(_pd->init_scratchpad_md());
            return safe_ptr_assign(*reorder_pd, _pd.release());
        };

        std::unique_ptr<KDNN::ReorderLayer> kdnn_reorder_prim_;
    }; // pd_t

    kdnn_reorder_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_reorder_resource_t>(pd()->kdnn_reorder_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto src = CTX_IN_MEM(const void *, DNNL_ARG_FROM);
        auto dst = CTX_OUT_MEM(void *, DNNL_ARG_TO);

        return execute(ctx, src, dst);
    }

private:
    mutable std::mutex mtx;

    status_t execute(const exec_ctx_t &ctx, const void *src, void *dst) const {
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::ReorderLayer &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_reorder_resource_t>(this))->get_kdnn_obj();

        try {
            const memory_desc_wrapper output_d(pd()->dst_md());
            kdnn_obj.Run(src, static_cast<char *>(dst) +
                output_d.offset0() * types::data_type_size(output_d.data_type()));
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // kdnn_reorder_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_REORDER_HPP
