#ifndef KDNN_JIT_SVE_CONVOLUTION_HPP
#define KDNN_JIT_SVE_CONVOLUTION_HPP

#include <memory>
#include <iostream>

#include "kdnn.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_jit_sve_conv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa = isa_undef>
struct kdnn_jit_sve_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T("kdnn_jit", kdnn_jit_sve_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            bool ok = true && mayiuse(isa) && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, wei_type, dst_type, dst_type,
                            data_type::undef)
                    && attr()->has_default_values()
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status = kdnn_jit_sve_conv_fwd_kernel<isa>::init_conf(jcp_,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, *attr(),
                    dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            kdnn_jit_sve_conv_fwd_kernel<isa>::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        kdnn_jit_conv_conf_t jcp_;

    }; // pd_t

    kdnn_jit_sve_convolution_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new kdnn_jit_sve_conv_fwd_kernel<isa>(pd()->jcp_, *pd()->attr())));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_forward_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_forward_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_forward_3d(ctx);
        else
            assert(false);

        if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);

        return status::success;
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    void prepare_padded_bias(const float *&bias,
            const memory_tracking::grantor_t &scratchpad) const;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<kdnn_jit_sve_conv_fwd_kernel<isa>> kernel_;

}; // kdnn_jit_sve_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

