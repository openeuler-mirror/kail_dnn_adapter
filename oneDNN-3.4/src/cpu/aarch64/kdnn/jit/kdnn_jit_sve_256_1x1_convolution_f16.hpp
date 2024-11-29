/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2021-2023 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_JIT_SVE_256_1X1_CONVOLUTION_F16_HPP
#define CPU_AARCH64_JIT_SVE_256_1X1_CONVOLUTION_F16_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_hashing.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/cpu_reducer.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_sve_256_1x1_conv_kernel_f16.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_uni_1x1_conv_utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type>
struct kdnn_jit_sve_256_1x1_convolution_fwd_f16_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_()
            , src_stride_(1) {}
        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            if (copy(other) != status::success) is_initialized_ = false;
        }

        DECLARE_COMMON_PD_T("kdnn_jit_1x1:",
                kdnn_jit_sve_256_1x1_convolution_fwd_f16_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace format_tag;

            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, wei_type, dst_type, dst_type,
                            data_type::undef)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, dst_type)
                    && !has_zero_dim_memory() && set_default_formats()
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            assert(ndims() >= 3);
            src_stride_ = conv_d->strides[ndims() - 3]; //stride w, for h=w, need to get w stride
            const memory_desc_wrapper src_desc_w(src_d);
            const memory_desc_wrapper dst_desc_w(dst_md());

            const auto dat_tag_ncx = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            if(src_desc_w.matches_one_of_tag(dat_tag_ncx) == dat_tag_ncx && 
               dst_desc_w.matches_one_of_tag(dat_tag_ncx) == dat_tag_ncx) {//for layout is ncx, need change layout to nxc
                dnnl_memory_desc_t conv_src_md, conv_dst_md;
                auto dnnl_tag = src_md()->ndims == 4 ? dnnl_nhwc : dnnl_nwc;
                dnnl_memory_desc_create_with_tag(&conv_src_md, src_md()->ndims,
                        src_md()->dims, dnnl_f16, dnnl_tag);
                dnnl_memory_desc_create_with_tag(&conv_dst_md, dst_md()->ndims,
                        dst_md()->dims, dnnl_f16, dnnl_tag);
                const memory_desc_t* p_changed_src = conv_src_md;
                const memory_desc_t* p_changed_dst = conv_dst_md;
                
                kdnn_rtus_prepare(this, conv_d, p_changed_src, p_changed_dst);
                const memory_desc_t* weights_d = weights_md();

                CHECK(kdnn_jit_sve_256_1x1_conv_kernel_f16::init_conf(jcp_, *conv_d, const_cast<memory_desc_t&>(*p_changed_src),
                    const_cast<memory_desc_t&>(*weights_d), const_cast<memory_desc_t&>(*p_changed_dst), *attr(), dnnl_get_max_threads(),
                    rtus_.reduce_src_));
            }
            else {
                kdnn_rtus_prepare(this, conv_d, src_d, dst_md());
                const memory_desc_t* dst_d = dst_md();
                const memory_desc_t* weights_d = weights_md();

                CHECK(kdnn_jit_sve_256_1x1_conv_kernel_f16::init_conf(jcp_, *conv_d, const_cast<memory_desc_t&>(*src_d),
                    const_cast<memory_desc_t&>(*weights_d), const_cast<memory_desc_t&>(*dst_d), *attr(), dnnl_get_max_threads(),
                    rtus_.reduce_src_));
            }
            auto scratchpad = scratchpad_registry().registrar();
            kdnn_jit_sve_256_1x1_conv_kernel_f16::init_scratchpad(scratchpad, (const kdnn_jit_1x1_conv_conf_t &)jcp_);

            kdnn_rtus_prepare_space_info(this, scratchpad, jcp_.nthr);
            return status::success;
        }

        const memory_desc_t *dst_md(
                int index = 0, bool user_input = false) const override {
                return cpu_convolution_fwd_pd_t::dst_md(index, user_input);
        }

        const memory_desc_t *arg_md(
                int arg, bool user_input = false) const override {
            return convolution_fwd_pd_t::arg_md(arg, user_input);
        }

        arg_usage_t arg_usage(int arg) const override {
            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS))
                return arg_usage_t::input;

            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS)
                    && attr_post_op_dw_inputs() > 1)
                return arg_usage_t::input;

            return convolution_fwd_pd_t::arg_usage(arg);
        }

        kdnn_jit_1x1_conv_conf_t jcp_;
        kdnn_reduce_to_unit_stride_t rtus_;
        int src_stride_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper dst_d(&dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_nCx16c
                    = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            const auto curr_src_tag
                    = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
            const auto curr_dst_tag
                    = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);
            auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
                    : utils::pick(ndims() - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        status_t copy(const pd_t &other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            src_stride_ = other.src_stride_;
            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend status_t kdnn_init_rtus_driver(conv_t *self);

    kdnn_jit_sve_256_1x1_convolution_fwd_f16_t(const pd_t *apd) 
        : primitive_t(apd)
        , is_src_need_change_layout_(false)
        , is_dst_need_change_layout_(false) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t init(engine_t *engine) override {
        using namespace format_tag;
        CHECK(safe_ptr_assign(kernel_,
                new kdnn_jit_sve_256_1x1_conv_kernel_f16(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        CHECK(kernel_->create_kernel());

        CHECK(kdnn_init_rtus_driver<sve_256>(this));

        const memory_desc_wrapper src_d(pd()->src_md()), dst_d(pd()->dst_md());
        const auto dat_tag_ncx = utils::pick(pd()->ndims() -3, ncw, nchw, ncdhw);
        if(src_d.matches_one_of_tag(dat_tag_ncx) == dat_tag_ncx) is_src_need_change_layout_ = true;
        if(dst_d.matches_one_of_tag(dat_tag_ncx) == dat_tag_ncx) is_dst_need_change_layout_ = true;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const dst_data_t *bias, const wei_data_t *weights_dw,
            const dst_data_t *bias_dw, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad,
            const void *post_ops_binary_rhs_arg_vec,
            const void *post_ops_binary_rhs_arg_vec_dw) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<kdnn_jit_sve_256_1x1_conv_kernel_f16> kernel_;
    std::unique_ptr<kdnn_rtus_driver_t<sve_256>> rtus_driver_;
    std::atomic<bool> is_src_need_change_layout_, is_dst_need_change_layout_;
};

using kdnn_jit_sve_256_1x1_convolution_fwd_f16
        = kdnn_jit_sve_256_1x1_convolution_fwd_f16_t<data_type::f16>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
