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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_generator.hpp"

#include "cpu/aarch64/kdnn/jit/kdnn_jit_sve_256_1x1_convolution_f16.hpp"



namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define data_blk_off(f, n, c, d, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) \
                  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                                  : (f).blk_off(n, c, d, h, w)))
/* convolution forward */

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void kdnn_jit_sve_256_1x1_convolution_fwd_f16_t<src_type, wei_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {
    const auto &jcp = kernel_->jcp;
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const wei_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    auto bias_dw = CTX_IN_MEM(
            const dst_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(pd()->jcp_.post_ops, ctx);

    auto scratchpad = ctx.get_scratchpad_grantor();

    if (pd()->wants_padded_bias()) {
        auto padded_bias
                = scratchpad.template get<dst_data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    auto dat_tag_nxc = jcp.ndims == 3 ? memory::format_tag::nwc : memory::format_tag::nhwc;
    auto dat_tag_ncx = jcp.ndims == 3 ? memory::format_tag::ncw : memory::format_tag::nchw;
    const int N = jcp.mb, IH = jcp.ih * pd()->src_stride_, IW = jcp.iw * pd()->src_stride_, IC = jcp.ic, OC = jcp.oc, OH = jcp.oh, OW = jcp.ow;
    const memory::dims conv_src_2d_sizes = {N, IC, IW}, conv_src_3d_sizes = {N, IC, IH, IW};
    const memory::dims conv_dst_2d_sizes = {N, OC, OW}, conv_dst_3d_sizes = {N, OC, OH, OW};
    engine eng(engine::kind::cpu, 0);
    stream s(eng);
    
    if(is_src_need_change_layout_) {// if src tag is ncx, need to change layout to nxc
        void* p_src_data = nullptr;
        auto reorder_src_func = [&](const memory::dims& conv_src_sizes){ // do reoder for src
            auto src_mem = memory({conv_src_sizes, memory::data_type::f16, dat_tag_ncx}, eng);
            auto src_mem_changed = memory({conv_src_sizes, memory::data_type::f16, dat_tag_nxc}, eng);
            src_mem.set_data_handle((void*)const_cast<src_data_t*>(src));
    
            auto reorder_src = reorder(src_mem, src_mem_changed);
            reorder_src.execute(
                s, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, src_mem_changed}});
            s.wait(); // wait for the reorder to complete
            p_src_data = src_mem_changed.get_data_handle();
            assert(p_src_data != nullptr);
            if(is_dst_need_change_layout_) {//for input = ncx & output = ncx
                auto conv_dst_sizes = jcp.ndims == 3 ? conv_dst_2d_sizes : conv_dst_3d_sizes;
                auto dst_mem_changed = memory({conv_dst_sizes, memory::data_type::f16, dat_tag_nxc}, eng);
                void* p_dst_data = dst_mem_changed.get_data_handle();
                assert(p_dst_data != nullptr);
                parallel(jcp.nthr, [&](const int ithr, const int nthr) {
                execute_forward_thr(ithr, nthr, (const src_data_t *)p_src_data, weights, bias, weights_dw, bias_dw,
                    (src_data_t*)p_dst_data, scratchpad, post_ops_binary_rhs_arg_vec.data(),
                    nullptr/*post_ops_binary_rhs_arg_vec_dw.data()*/);
                });
                auto dst_mem = memory({conv_dst_sizes, memory::data_type::f16, dat_tag_ncx}, eng);
                dst_mem.set_data_handle((void*)dst);
                auto reorder_dst = reorder(dst_mem_changed, dst_mem);
                reorder_dst.execute(
                    s, {{DNNL_ARG_FROM, dst_mem_changed}, {DNNL_ARG_TO, dst_mem}});
                s.wait(); // wait for the reorder to complete
            }
            else {//for input = ncx, but output != ncx
                parallel(jcp.nthr, [&](const int ithr, const int nthr) {
                execute_forward_thr(ithr, nthr, (const src_data_t *)p_src_data, weights, bias, weights_dw, bias_dw,
                    dst, scratchpad, post_ops_binary_rhs_arg_vec.data(),
                    nullptr/*post_ops_binary_rhs_arg_vec_dw.data()*/);
                });
            }
        };
        jcp.ndims == 3 ? reorder_src_func(conv_src_2d_sizes) : reorder_src_func(conv_src_3d_sizes);
    }
    else {
        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad, post_ops_binary_rhs_arg_vec.data(),
                nullptr/*post_ops_binary_rhs_arg_vec_dw.data()*/);
        });
    }

    if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void kdnn_jit_sve_256_1x1_convolution_fwd_f16_t<src_type, wei_type,
        dst_type>::execute_forward_thr(const int ithr, const int nthr,
        const src_data_t *src, const wei_data_t *weights,
        const dst_data_t *bias, const wei_data_t *weights_dw,
        const dst_data_t *bias_dw, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad,
        const void *post_ops_binary_rhs_arg_vec,
        const void *post_ops_binary_rhs_arg_vec_dw) const {
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper dw_weights_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS));
    const memory_desc_wrapper dw_bias_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<src_data_t>(key_conv_rtus_space)
            : nullptr;
    dnnl_memory_desc_t conv_src_md, conv_dst_md;
    if(is_src_need_change_layout_) {
        auto src_dnnl_tag = pd()->src_md()->ndims == 4 ? dnnl_nhwc : dnnl_nwc;
        dnnl_memory_desc_create_with_tag(&conv_src_md, pd()->src_md()->ndims,
            pd()->src_md()->dims, dnnl_f16, src_dnnl_tag);  //only surpport 2d or 3d
    }
    else {
        conv_src_md = const_cast<dnnl_memory_desc_t>(pd()->src_md());
    }
    
    if(is_dst_need_change_layout_) {
        auto dst_dnnl_tag = pd()->src_md()->ndims == 4 ? dnnl_nhwc : dnnl_nwc;
        dnnl_memory_desc_create_with_tag(&conv_dst_md, pd()->dst_md()->ndims,
            pd()->dst_md()->dims, dnnl_f16, dst_dnnl_tag); //only surpport 2d or 3d
    }
    else {
        conv_dst_md = const_cast<dnnl_memory_desc_t>(pd()->dst_md());
    }

    const memory_desc_wrapper src_d(conv_src_md);
    const memory_desc_wrapper dst_d(conv_dst_md);
    const int ndims = conv_src_md->ndims;
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };
    auto p = kdnn_jit_1x1_conv_call_s();

    auto rp = kdnn_rtus_driver_t<sve_256>::call_params_t();
    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;

    // override some constants for fused dw_conv
    const int os_block = jcp.with_dw_conv ? jcp.ow : jcp.bcast_block;
    const int nb_bcast = jcp.with_dw_conv ? jcp.oh : jcp.nb_bcast;
    const int nb_bcast_blocking = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking;
    const int nb_bcast_blocking_max
            = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking_max;
    const int nb_load_blocking = jcp.nb_load_blocking;
    const int nb_load_blocking_max = jcp.with_dw_conv
            ? jcp.nb_load_blocking
            : jcp.nb_load_blocking_max;
    const bool is_dst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    auto init_bcast = [&](int iwork, int bcast_end, int &n, int &g,
                              int &bcast_step, int &od, int &oh, int &ow,
                              int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, nb_bcast);
        bcast_step = step(
                nb_bcast_blocking, nb_bcast - osb, nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        od = os / (jcp.oh * jcp.ow);
        int os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;

        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int ocb_end, int &load_step) {
        load_step = step(nb_load_blocking, ocb_end - ocb, nb_load_blocking_max);
        const auto max_oc
                = nstl::min(ocb_end * jcp.oc_block, jcp.oc_without_padding);
        p.load_dim = this_block_size(
                ocb * jcp.oc_block, max_oc, load_step * jcp.oc_block);
    };

    auto init_reduce = [&](int icb) {
        const int nb_ic_blocking_step
                = nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
        p.first_last_flag = 0 | (icb == 0 ? kdnn_FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking_step >= nb_ic ? kdnn_FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking_step * jcp.ic_block);
        rp.icb = p.reduce_dim;
    };

    auto ker_1x1 = [&](int ocb, int ocb_start, int icb, int n, int g, int od,
                           int oh, int ow, int id, int ih, int iw) {
        const int oc_off_idx = is_dst_layout_nxc
                ? g * jcp.oc + ocb * jcp.oc_block
                : g * nb_oc + ocb;
        const size_t dst_off = data_blk_off(dst_d, n, oc_off_idx, od, oh, ow);

        p.output_data = &dst[dst_off];

        p.bias_data = bias
                ? &bias[oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block)]
                : nullptr;

        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

        const int ic_off_idx = is_src_layout_nxc
                ? g * jcp.ic + icb * jcp.ic_block
                : g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                    + (is_src_layout_nxc ? ic_off_idx
                                         : jcp.is * ic_off_idx * jcp.ic_block);
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);
                (*rtus_driver_)(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);

        p.oc_l_off = oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block);
        p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
        p.dst_orig = dst;

        (*kernel_)(&p);
    };
    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

        if (jcp.loop_order == kdnn_loop_rlb) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    int iwork = bcast_start;
                    while (iwork < bcast_end) {
                        int n {0}, g {0}, bcast_step {0}, od {0}, oh {0},
                                ow {0}, id {0}, ih {0}, iw {0};
                        init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh,
                                ow, id, ih, iw);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        iwork += bcast_step;
                    }
                    ocb += load_step;
                }
            }
        } else if (jcp.loop_order == kdnn_loop_lbr) {
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, ocb_end, load_step);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                            id {0}, ih {0}, iw {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == kdnn_loop_rbl) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                            id {0}, ih {0}, iw {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    int ocb = ocb_start;
                    while (ocb < ocb_end) {
                        int load_step;
                        init_load(ocb, ocb_end, load_step);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        ocb += load_step;
                    }
                    iwork += bcast_step;
                }
            }
        } else if (jcp.loop_order == kdnn_loop_blr) {
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                        id {0}, ih {0}, iw {0};
                init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow, id,
                        ih, iw);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
        } else {
            assert(!"unsupported loop order");
        }
    };
    
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
    int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
            ocb_start, ocb_end, jcp.load_grp_count);

    conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
}
template struct kdnn_jit_sve_256_1x1_convolution_fwd_f16_t<data_type::f16>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
