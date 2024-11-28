#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"

#include "cpu/aarch64/kdnn/jit/kdnn_jit_f16_sve_conv_kernel.hpp"
#include "cpu/platform.hpp"

#define KDNN_GET_OFF(field) static_cast<int32_t>(offsetof(kdnn_jit_conv_call_s, field))
#define KDNN_A64FX_L2_EFFECTIVE_CAPACITY ((666 - 128) * 1024)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

namespace {

constexpr auto small_spatial = 14;
unsigned int L2_cache_size = platform::get_per_core_cache_size(2);

// calculates filter size taking into account dilation
inline int calculate_extended_filter_size(int filter_size, int dilation) {
    return (filter_size - 1) * (dilation + 1) + 1;
}

inline int calculate_end_padding(int start_padding, int dst_size, int src_size,
        int spatial_stride, int dilated_filter_size) {
    return (dst_size - 1) * spatial_stride + dilated_filter_size
            - (src_size + start_padding);
}

inline void pick_loop_order(kdnn_jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(
            jcp.prop_kind, forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // The w in the loop order is currently ignored by 3D BWD_D
    jcp.loop_order = (w <= small_spatial && h <= small_spatial) ? kdnn_loop_cwgn
                                                                : kdnn_loop_gncw;
    if (utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc)
            && jcp.ngroups > 1 && jcp.oc < 16)
        jcp.loop_order = kdnn_loop_nhwcg;
}

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value) {
    if (mdw.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(md, tag_value));
        tag = tag_value;
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

inline bool is_1stconv(const kdnn_jit_conv_conf_t &jcp) {
    if (mayiuse(sve_256))
        return (jcp.ic < 16 && jcp.ngroups == 1);
    else
        return one_of(jcp.ic, 1, 3);
}

inline bool is_ow_threading_on(const kdnn_jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}

inline bool is_owb_prefetching(const kdnn_jit_conv_conf_t &jcp) {
    return false;
}

} // namespace

template <cpu_isa_t isa>
void kdnn_jit_f16_sve_conv_fwd_kernel<isa>::prepare_output(int ur_w) {

    auto zreg_out_h = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegH(idx);
    };

    int prev_out_ofs = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            fmov(zreg_out_h(j, k));
            if (!is_owb_prefetching(jcp)) {
                size_t aux_output_offset = get_output_offset(j, k);
                std::string op = "LD";
                if (j == 0) {
                    prefetch(op, 2, reg_out_prf, aux_output_offset);
                    add_imm(reg_tmp_addr, reg_out_prf, aux_output_offset,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_tmp_addr, reg_tmp_addr,
                            aux_output_offset - prev_out_ofs, reg_tmp_imm);
                    prefetch(op, 2, reg_tmp_addr, 0);
                }
                prev_out_ofs = aux_output_offset;
            }
        }
}

template <cpu_isa_t isa>
void kdnn_jit_f16_sve_conv_fwd_kernel<isa>::store_output(int ur_w) {

    Label no_update_label, store_label, eltwise_label;

    auto _test = [&](const int cond) { return tst(reg_channel, cond); };

    auto zreg_tmp = [=](int idx) { return ZReg(idx); };
    auto zreg_tmp_h = [=](int idx) { return ZRegH(idx); };

    auto zreg_out = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZReg(idx);
    };
    auto zreg_out_h = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegH(idx);
    };

    ldr(reg_channel, ptr(kdnn_abi_param1, KDNN_GET_OFF(flags)));

    if (jcp.with_bias) { ldr(reg_bias, ptr(kdnn_abi_param1, KDNN_GET_OFF(bias))); }

    if (!jcp.with_sum) {
        auto _jmp = [&](const Label &l) { return b(NE, l); };

        _test(kdnn_FLAG_IC_FIRST);
        _jmp(no_update_label);
    }

    int reg_ofs = jcp.ur_w * jcp.nb_oc_blocking;
    int num_regs = 32 - reg_ofs;
    int prev_out_ofs = -1;

    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = get_output_offset(j, k);
            int idx = reg_ofs + ((j + k * ur_w) % num_regs);
            if (j == 0) {
                add_imm(reg_out_ofs, reg_out, aux_output_offset, reg_tmp_imm);
                prev_out_ofs = aux_output_offset;
                ld1h(zreg_tmp(idx).h, P_ALL_ONE / T_z, ptr(reg_out_ofs));
            } else {
                add_imm(reg_out_ofs, reg_out_ofs,
                        aux_output_offset - prev_out_ofs, reg_tmp_imm);
                prev_out_ofs = aux_output_offset;
                ld1h(zreg_tmp(idx).h, P_ALL_ONE / T_z, ptr(reg_out_ofs));
            }
        }
        for (int j = 0; j < ur_w; j++) {
            int idx = reg_ofs + ((j + k * ur_w) % num_regs);
            fadd(zreg_out_h(j, k), zreg_out_h(j, k), zreg_tmp_h(idx));
        }
    }

    if (!jcp.with_sum) {
        b(eltwise_label);
    } else {
        auto _jmp = [&](const Label &l) { return b(EQ, l); };

        // *Note 1
        _test(kdnn_FLAG_IC_FIRST);
        _jmp(eltwise_label);
    }

    auto bias_load = [=](int bias_offset, int idx) {
        int ofs = bias_offset;

        if ((KDNN_F16_VL_OFS(ofs) < LDRMAX) && (KDNN_F16_VL_OFS(ofs) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            add_imm(X_DEFAULT_ADDR, reg_bias, ofs, X_TMP_0);
            ld1h(zreg_tmp(idx).h, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else {
            add_imm(reg_tmp_addr, reg_bias, ofs, reg_tmp_imm);
            ld1h(zreg_tmp(idx).h, P_ALL_ONE / T_z, ptr(reg_tmp_addr));
        }
    };

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            int idx = reg_ofs + (k % num_regs);
            bias_load(bias_offset, idx);
            for (int j = 0; j < ur_w; j++) {
                fadd(zreg_out_h(j, k), zreg_out_h(j, k), zreg_tmp_h(idx));
            }
            int ofs = bias_offset + 256; // cache line size ?
            std::string op = "LD";
            prefetch(op, 2, reg_bias, ofs);
        }
    }

    L(eltwise_label);
   
    auto out_str = [=](int j, int k, int aux_output_offset, int prev_out_ofs) {
        int ofs = aux_output_offset;

        if (prev_out_ofs == -1)
            add_imm(reg_tmp_addr, reg_out, ofs, reg_tmp_imm);
        else
            add_imm(reg_tmp_addr, reg_tmp_addr, ofs - prev_out_ofs,
                reg_tmp_imm);
        st1h(zreg_out(j, k).h, P_ALL_ONE / T_z, ptr(reg_tmp_addr));
        prev_out_ofs = aux_output_offset;
        return prev_out_ofs;
    };

    L(store_label);
    prev_out_ofs = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = (size_t)typesize
                    * ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;

            prev_out_ofs = out_str(j, k, aux_output_offset,
                    prev_out_ofs); // <- reg_tmp_addr
        }
    }
}

template <cpu_isa_t isa>
void kdnn_jit_f16_sve_conv_fwd_kernel<isa>::compute_loop_fma_core(
        int ur_w, int pad_l, int pad_r) {
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;

    Label kh_label, kd_label;

    std::vector<Label> ic_tail_jmp(kw);
    int shift_kernel_ptr
            = jcp.typesize_in * jcp.kw * jcp.oc_block * jcp.ic_block;
    int inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                       : (!jcp.is_1stconv ? ic_block : 1);

    int shift_input_ptr
            = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mul;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        mov(reg_out_org, reg_out);
        ldr(reg_ki, ptr(kdnn_abi_param1, KDNN_GET_OFF(kd_padding)));
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            mov(aux_reg_ker_d_org, aux_reg_ker_d);
        } else {
            mov(aux_reg_ker_d, aux_reg_ker_d_org);
        }
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        ldr(reg_kj, ptr(kdnn_abi_param1, KDNN_GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    auto zreg_inp_h = [=](int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return ZRegH(idx);
    };
    auto zreg_out_h = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegH(idx);
    };
    auto zreg_wei = [=](int idx) {
        assert(idx < 32);
        return ZReg(idx);
    };
    auto zreg_wei_h = [=](int idx) {
        assert(idx < 32);
        return ZRegH(idx);
    };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_input_offset,
                              int prev_ofs) {
        int ofs;
        if ((prev_ofs != -1) && ((aux_input_offset - prev_ofs) > 0)) {
            ofs = aux_input_offset - prev_ofs;
            add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                reg_tmp_imm);
        } else {
            ofs = aux_input_offset;
            add_imm(reg_prev_bcast_addr, aux_reg_inp, ofs, reg_tmp_imm);
        }

        ld1rh(zreg_inp_h(jj, nb_oc_block), P_ALL_ONE,
            ptr(reg_prev_bcast_addr));
        prev_ofs = aux_input_offset;

        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int reg_idx, int prev_ofs) {
        int ofs;
        if ((prev_ofs != -1) && ((aux_kernel_offset - prev_ofs) > 0)) {
            ofs = aux_kernel_offset - prev_ofs;
            add_imm(reg_prev_wei_addr, reg_prev_wei_addr, ofs,
                reg_tmp_imm);
        } else {
            ofs = aux_kernel_offset;
            add_imm(reg_prev_wei_addr, aux_reg_ker, ofs, reg_tmp_imm);
        }

        ld1h(zreg_wei(reg_idx).h, P_ALL_ONE / T_z,
            ptr(reg_prev_wei_addr));
        prev_ofs = aux_kernel_offset;
        return prev_ofs;
    };

    align(16);
    L(kh_label);
    {
        int prev_bcast_ofs = -1;
        int prev_wei_ofs = -1;
        for (int ki = 0; ki < kw; ki++) {

            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);

            int wei_reg_ofs = nb_oc_block * jcp.ur_w;
            wei_reg_ofs += ur_w >= 16 ? 1 : jj_end;
            int num_regs4wei = 32 - wei_reg_ofs;
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail) {
                        break;
                    } else if (ic == ic_tail) {
                        cmp_imm(reg_channel, ic_tail, reg_tmp_imm);
                        b(EQ, ic_tail_jmp[ki]);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int reg_idx = wei_reg_ofs + ii;
                    if (reg_idx >= 32) break;
                    int aux_kernel_offset = jcp.typesize_in
                            * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd
                                            * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    wei_count++;
                    if (jj_end - jj_start > 0) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + (ii % num_regs4wei),
                                prev_wei_ofs);
                    }
                }

                if ((jcp.kernel_kind == kdnn_expl_bcast) && (ur_w < 16)) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        size_t aux_input_offset
                                = get_input_offset(ki, ic, jj, pad_l);
                        prev_bcast_ofs = bcast_load(jj, nb_oc_block,
                                aux_input_offset, prev_bcast_ofs);
                    }
                }

                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                            * ((ii + wei_count) * jcp.nb_ic * jcp.kh * jcp.kw
                                            * jcp.kd * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == kdnn_expl_bcast) {
                            if (ur_w >= 16) {
                                size_t aux_input_offset
                                        = get_input_offset(ki, ic, jj, pad_l);
                                prev_bcast_ofs = bcast_load(0, nb_oc_block,
                                        aux_input_offset, prev_bcast_ofs);

                                fmla(zreg_out_h(jj, ii), P_ALL_ONE,
                                        zreg_inp_h(0, nb_oc_block),
                                        zreg_wei_h(wei_reg_ofs
                                                + (ii % num_regs4wei)));

                            } else {
                                fmla(zreg_out_h(jj, ii), P_ALL_ONE,
                                        zreg_inp_h(jj, nb_oc_block),
                                        zreg_wei_h(wei_reg_ofs
                                                + (ii % num_regs4wei)));
                            }
                        } else {
                            assert(NULL);
                        }

                    if ((jj_end - jj_start > 0)
                            && ((wei_count + ii) < nb_oc_block)) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + ((ii + wei_count) % num_regs4wei),
                                prev_wei_ofs);
                    }
                }
            }
            L(ic_tail_jmp[ki]);
        }

        add_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp, aux_reg_inp, shift_input_ptr, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1); //dec(reg_kj);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d, aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul,
                reg_tmp_imm);
        const int ker_shift
                = typesize * jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
        add_imm(aux_reg_ker_d, aux_reg_ker_d, ker_shift, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1); //dec(reg_ki);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        if (icb_loop_in_compute_function) mov(aux_reg_ker_d, aux_reg_ker_d_org);
        mov(reg_out, reg_out_org);
    }
}

template <cpu_isa_t isa>
void kdnn_jit_f16_sve_conv_fwd_kernel<isa>::compute_loop(
        int ur_w, int pad_l, int pad_r) {

    if (jcp.ndims == 5) mov(reg_oi_org, reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                        < nstl::max(jcp.f_pad, jcp.back_pad)) {
            ldr(reg_kj, ptr(kdnn_abi_param1, KDNN_GET_OFF(kd_padding)));
            cmp(reg_kj, 0);
            b(LE, skip_compute_loop);
        }
    }
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        ldr(reg_kj, ptr(kdnn_abi_param1, KDNN_GET_OFF(kh_padding)));
        cmp(reg_kj, 0);
        b(LE, skip_compute_loop);
    }

    Label ic_loop;
    const bool generate_icb_loop = jcp.nb_ic > 1 && is_src_layout_nxc();
    if (generate_icb_loop) {
        mov(reg_inp_org, reg_inp);
        mov(reg_ker_org, reg_ker);

        ldr(reg_channel, ptr(param, KDNN_GET_OFF(reduce_work)));
        L(ic_loop);
    }

    if (jcp.ver == kdnn_ver_fma)
        if (jcp.is_1stconv && jcp.kernel_kind != kdnn_expl_bcast)
            assert(!"STOP:jcp.is_1stconv && jcp.kernel_kind != expl_bcast");
        else if (jcp.kernel_kind == kdnn_embd_bcast && jcp.nb_oc_blocking == 1)
            assert(!"STOP:jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking "
                    "== 1");
        else {
            compute_loop_fma_core(ur_w, pad_l, pad_r);
        }
    else
        assert(!"unknown convolution version");

    if (generate_icb_loop) {
        assert(is_src_layout_nxc());
        const int inp_shift = jcp.ic_block * jcp.typesize_in;
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        const int ker_shift = jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                * jcp.oc_block * jcp.typesize_in;
        add_imm(reg_ker, reg_ker, ker_shift, reg_tmp_imm);
        sub_imm(reg_channel, reg_channel, jcp.ic_block, reg_tmp_imm);
        b(GT, ic_loop);
        mov(reg_ker, reg_ker_org);
        mov(reg_inp, reg_inp_org);
    }

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) mov(reg_oi, reg_oi_org);
}

template <cpu_isa_t isa>
void kdnn_jit_f16_sve_conv_fwd_kernel<isa>::generate() {
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    int inp_mult = is_src_layout_nxc() ? jcp.ngroups * jcp.ic
                                       : (jcp.is_1stconv ? 1 : jcp.ic_block);
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * ur_w * stride_w * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;
    int out_shift = jcp.typesize_out * ur_w
            * (is_dst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block);

    const int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float16_t);

    preamble();

    //TO DO : renaming predicate register (P_ALL_ONE)
    if (simd_w_ != cpu_sveLen / sizeof(float16_t))
        set_preg(P_ALL_ONE.h, simd_w_, X_TMP_0, X_TMP_1);

    ldr(reg_inp, ptr(kdnn_abi_param1, KDNN_GET_OFF(src)));
    ldr(reg_out, ptr(kdnn_abi_param1, KDNN_GET_OFF(dst)));
    ldr(reg_ker, ptr(kdnn_abi_param1, KDNN_GET_OFF(filt)));
    ldr(reg_kh, ptr(kdnn_abi_param1, KDNN_GET_OFF(kh_padding)));
    if (jcp.ndims == 5) mov(aux_reg_ker_d_org, reg_ker);

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    if (!is_ow_threading_on(jcp)) { // nb_ow <= 1
        // nb_ow is # of output width blocks ??

        // ow is being processed as a whole - with left and right paddings
        // n_oi is # of output width blocks ??
        if (r_pad1 > 0) n_oi--;

        if (ow == ur_w) {
            ldr(reg_out_prf, ptr(kdnn_abi_param1, KDNN_GET_OFF(dst_prf)));
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            mov(reg_out_prf, reg_out);
            if (n_oi == 0) {
                add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                compute_loop(ur_w, l_pad, r_pad1);
                add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
                add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            } else {
                mov(reg_oi, 0);
                if (l_pad > 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w, l_pad, 0);
                    add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                    add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); // increment
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        add_imm(reg_out_prf, reg_out_prf, out_shift,
                                reg_tmp_imm);
                        compute_loop(ur_w, 0, 0);

                        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
                        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                        add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); //inc(reg_oi);
                        cmp_imm(reg_oi, n_oi, reg_tmp_imm);

                        b(LT, ow_loop_label);
                    }
                }
                if (r_pad1 > 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w, 0, r_pad1);
                    add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                }
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow - 1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        ldr(reg_owb, ptr(kdnn_abi_param1, KDNN_GET_OFF(owb)));
        cmp(reg_owb, 0); // is that the first ow-block ?
        b(GT, middle_ow_blocks_label);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        mov(reg_out_prf, reg_out);

        if (l_pad > 0) {
            add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
            compute_loop(ur_w, l_pad, 0);
            add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
            add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
            sub(reg_oi, reg_oi, 1); // decrement
            cmp(reg_oi, 0);
        }
        b(oi_loop_label);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add_imm(reg_inp, reg_inp, inp_shift_pad_second_block, reg_tmp_imm);
        }

        // set number of iteration for oi-loop
        cmp_imm(reg_owb, jcp.nb_ow - 1, reg_tmp_imm); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        b(EQ, oi_loop_label);
        cmp_imm(reg_owb, jcp.nb_ow - 2, reg_tmp_imm); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        b(EQ, oi_loop_label);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
        cmp(reg_oi, 0);
        b(LE, oi_loop_end_label);

        add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
        compute_loop(ur_w, 0, 0);
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
        sub(reg_oi, reg_oi, 1); // dec(reg_oi);
        cmp(reg_oi, 0);
        b(oi_loop_start_label);
        L(oi_loop_end_label);

        ldr(reg_owb, ptr(kdnn_abi_param1, KDNN_GET_OFF(owb)));

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            b(EQ, last_oi_label);
        } else {
            b(EQ, end_label);
        }
        cmp_imm(reg_owb, jcp.nb_ow - 2, reg_tmp_imm); // next to last ow-block ?
        b(LT, end_label);
        if (next_last_ow_block_padded) {
            b(EQ, last_oi_label);
        } else {
            b(EQ, end_label);
        }
        // that is last block
        if (!last_ow_block_padded) { b(tail_label); }

        // last oi block with right padding
        L(last_oi_label);
        add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
        compute_loop(ur_w, 0, r_pad1);
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);

        ldr(reg_owb, ptr(kdnn_abi_param1, KDNN_GET_OFF(owb)));
        cmp_imm(reg_owb, jcp.nb_ow - 1, reg_tmp_imm); // last ow_block?
        b(LT, end_label);

        L(tail_label);
        if (ur_w_tail != 0) {
            add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
            compute_loop(ur_w_tail, 0, r_pad);
        }
        L(end_label);
    }
    postamble();

}

/*struct instantiation*/
template struct kdnn_jit_f16_sve_conv_fwd_kernel<sve_256>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

