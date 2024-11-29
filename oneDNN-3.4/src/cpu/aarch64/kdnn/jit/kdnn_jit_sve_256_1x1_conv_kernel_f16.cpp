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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_sve_256_1x1_conv_kernel_f16.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_uni_1x1_conv_utils.hpp"
#define KDNN_GET_OFF(field) static_cast<int32_t>(offsetof(kdnn_jit_1x1_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

kdnn_jit_sve_256_1x1_conv_kernel_f16::kdnn_jit_sve_256_1x1_conv_kernel_f16(
        const kdnn_jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
    }
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::bcast_loop(int load_loop_blk) {

    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_output_data, reg_output_data);
    ldr(reg_bcast_loop_iter, ptr(X_SP, reg_bcast_loop_work_offt));

    Label bcast_loop;
    Label bcast_loop_tail;
    Label large_tail;

    cmp_imm(reg_bcast_loop_iter, jcp.bcast_block, reg_tmp_imm);
    b(LT, bcast_loop_tail);

    L(bcast_loop);
    {
        assert(jcp.ur != 0);
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        //do reduce loop
        for (int i = 0; i < num_substeps; i++) {
            if (i + 1 == num_substeps) L(large_tail);
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_substep, reg_tmp_imm);
                add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_substep, reg_tmp_imm);
            } else {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep,
                        reg_tmp_imm);
                add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_output_substep,
                        reg_tmp_imm);
            }
            subs_imm(reg_bcast_loop_iter, reg_bcast_loop_iter, jcp.ur,
                    reg_tmp_imm);
        }
        cmp_imm(reg_bcast_loop_iter, jcp.bcast_block, reg_tmp_imm);
        b(GE, bcast_loop);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        if (jcp.ur_tail >= jcp.ur) {
            cmp_imm(reg_bcast_loop_iter, jcp.ur, reg_tmp_imm);
            b(GE, large_tail);
        }
        if (jcp.ur_tail % jcp.ur) {
            cmp(reg_bcast_loop_iter, 0);
            b(LE, bcast_loop_tail_out);
            reduce_loop(load_loop_blk, jcp.ur_tail % jcp.ur, 0, true);
            L(bcast_loop_tail_out);
        }
    }
}

Xbyak_aarch64::XReg kdnn_jit_sve_256_1x1_conv_kernel_f16::output_ptr(
        const bool kdnn_is_out_layout_nxc, const int i_load, const int i_ur,
        Xbyak_aarch64::XReg addr) {
    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {
        int i_load_shift = kdnn_is_out_layout_nxc
                ? jcp.load_block
                : (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) * jcp.load_block;
        int i_ur_shift = kdnn_is_out_layout_nxc ? jcp.load_dim : jcp.load_block;
        int offset = (i_load * i_load_shift + i_ur * i_ur_shift)
                * jcp.typesize_out;
        EVEX_compress_addr(addr, X_TMP_0, aux_reg_output_data, offset);
    } else {
        int offset = jcp.typesize_out * jcp.load_block * i_ur;
        mov(X_TMP_0, i_load);
        mul(X_TMP_0, reg_output_stride, X_TMP_0);
        add_imm(X_TMP_1, X_TMP_0, offset, X_TMP_2);
        add(addr, aux_reg_output_data, X_TMP_1);
    }
    return addr;
}

static int vreg_accum_idx(
        const int load_loop_blk, const int i_load, const int i_ur) {
    return (i_ur * load_loop_blk + i_load);
}

template <typename F>
static void iterate(const int load_loop_blk, const int ur, const bool mask_tail,
        const F &fun) {
    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
        const bool mask_flag = mask_tail && i_load + 1 == load_loop_blk;
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            fun(mask_flag, i_load, i_ur);
    }
}
template <typename F>
static void iterate(const int load_loop_blk, const int ur, const F &fun) {
    iterate(load_loop_blk, ur, false, fun);
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::apply_postops(
        const bool kdnn_is_out_layout_nxc, const int load_loop_blk, const int ur) {
    injector_utils::vmm_index_set_t vmm_idxs;
    if (jcp.with_binary) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        const auto mask_tail = jcp.oc_without_padding % jcp.load_block;
        iterate(load_loop_blk, ur, mask_tail,
                [&](const bool mask_flag, const int i_load, const int i_ur) {
                    const auto vmm_idx
                            = vreg_accum_idx(load_loop_blk, i_load, i_ur);
                    vmm_idxs.emplace(vmm_idx);

                    rhs_arg_params.vmm_idx_to_out_reg.emplace(
                            vmm_idx, aux_reg_output_data);
                    rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(vmm_idx,
                            get_output_offset(kdnn_is_out_layout_nxc, i_load, i_ur));
                    if (mask_flag)
                        rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
                });

        ldr(abi_param1, ptr(X_SP, reg_abi_param1_backup));

    } else {
        iterate(load_loop_blk, ur,
                [&](const bool, const int i_load, const int i_ur) {
                    vmm_idxs.emplace(
                            vreg_accum_idx(load_loop_blk, i_load, i_ur));
                });
    }
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {

    const bool out_layout_nxc = kdnn_is_out_layout_nxc(jcp);
    const bool load_layout_nxc = kdnn_is_load_layout_nxc(jcp);
    const bool bcast_layout_nxc = kdnn_is_bcast_layout_nxc(jcp);
    const int reduce_dim_tail = jcp.reduce_dim % jcp.reduce_block;
    const int load_dim_tail = jcp.load_dim % jcp.load_block;

    auto vreg_load
            = [=](int i_load) { return ZReg(ur * load_loop_blk + i_load); };

    auto vreg_input
            = [=](int i_ur) { return ZReg((ur + 1) * load_loop_blk + i_ur); }; //load input, 
    auto vreg_input_accord_blk
            = [=](int i_ur) { return ZReg(ur * (load_loop_blk + 1) + load_loop_blk + i_ur); };   //load input 2
    
    auto vreg_accum = [=](int i_load, int i_ur) {
        return ZReg(vreg_accum_idx(load_loop_blk, i_load, i_ur));
    };

    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(X_DEFAULT_ADDR, X_TMP_0, reg_bias_data,
                jcp.typesize_out * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast,
                             const Xbyak_aarch64::XReg addr,
                             const Xbyak_aarch64::XReg tmp) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            const int reduce_mul = bcast_layout_nxc ? jcp.reduce_dim
                                                    : jcp.reduce_loop_unroll;
            offt = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * reduce_mul
                    : i_ur * reduce_mul + i_reduce;
        } else {
            int rmul = bcast_layout_nxc ? jcp.ic : jcp.ic_block;
            offt = i_reduce * rmul + i_ur;
        }
        return EVEX_compress_addr(
                addr, tmp, aux_reg_bcast_data, jcp.typesize_in * offt, bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load,
                            const Xbyak_aarch64::XReg addr,
                            const Xbyak_aarch64::XReg tmp) {
        assert(jcp.reduce_loop_unroll != 0);
        int offt;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        int lmul = jcp.load_block
                * (load_layout_nxc ? 1
                                   : utils::rnd_up(
                                           jcp.reduce_dim, jcp.reduce_block));
        int rmul = load_layout_nxc ? jcp.load_dim : jcp.load_block;
        offt = i_load * lmul + u0 * rmul;
        return EVEX_compress_addr(addr, tmp, aux_reg_load_data,
                u1 * jcp.reduce_loop_load_step + jcp.typesize_in * offt);
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            tst(reg_reduce_pos_flag, kdnn_FLAG_REDUCE_FIRST);
            b(EQ, init_zero);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
            {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto vreg_acc = vreg_accum(i_load, i_ur);
                    bool tmpRet = i_load + 1 == load_loop_blk && load_dim_tail;
                    if (tmpRet)
                    {
                        ld1h(vreg_acc.h, k_load_dim_mask / T_z,
                                ptr(bias_ptr(i_load)));
                    }
                    else
                    {
                        ldr(vreg_acc, ptr(bias_ptr(i_load)));
                    }
                }
            }
            b(init_done);
        }

        L(init_zero);

        /* Zero clear */
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                eor(r.d, r.d, r.d);
            }
        L(init_done);
    };

    auto store = [=]() {
        Label store_noadd;
        if (!jcp.with_sum) {
            tst(reg_reduce_pos_flag, kdnn_FLAG_REDUCE_FIRST);
            b(NE, store_noadd);
        }

        for (int i_ur = 0; i_ur < ur; ++i_ur)
        {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum(i_load, i_ur).h;
                bool tmpRet = i_load + 1 == load_loop_blk && load_dim_tail;
                if (tmpRet)
                    ld1h(zreg_tmp.h, k_load_dim_mask / T_z,
                            ptr(output_ptr(out_layout_nxc, i_load, i_ur,
                                    X_DEFAULT_ADDR)));
                else
                    ldr(zreg_tmp,
                            ptr(output_ptr(out_layout_nxc, i_load, i_ur,
                                    X_DEFAULT_ADDR)));
                fadd(r, r, zreg_tmp.h);
            }
        }        

        L(store_noadd);
        if (jcp.with_eltwise || jcp.with_binary) {
            Label store_nopostops;
            tst(reg_reduce_pos_flag, kdnn_FLAG_REDUCE_LAST);
            b(EQ, store_nopostops);

            apply_postops(out_layout_nxc, load_loop_blk, ur);

            L(store_nopostops);
        }

        auto store_output = [=](bool output_is_aligned) {
            const auto mask_flag = load_dim_tail;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    auto vreg_acc = vreg_accum(i_load, i_ur);
                    // for nxc_layout-bwd_w, weights are still padded and the
                    // output_ptr here can be uninitialized scratchpad.
                    // To ensure final output (after reduction) is zero-padded,
                    // here we zero-pad output by omitting the mask.
                    if (jcp.prop_kind != backward_weights
                            && (i_load + 1 == load_loop_blk && mask_flag)) {
                        st1h(vreg_acc.h, k_load_dim_mask / T_z,
                                ptr(output_ptr(out_layout_nxc, i_load, i_ur,
                                        X_DEFAULT_ADDR)));
                    } else {
                        str(vreg_acc,
                                ptr(output_ptr(out_layout_nxc, i_load, i_ur,
                                        X_DEFAULT_ADDR)));
                    }
                }
            }
        };

        Label unaligned_store, end_store;
        tst(aux_reg_output_data, cpu_isa_traits<sve_256>::vlen - 1);
        b(NE, unaligned_store);
        store_output(true);
        b(end_store);
        L(unaligned_store);
        { store_output(false); }
        L(end_store);
    };

    auto bcast_load = [=](const int i_reduce)
    {
        for (int i_ur = 0; i_ur < ur; ++i_ur)
        {
            auto vreg = vreg_input(i_ur);
            if (jcp.expl_bcast && load_loop_blk >= 1) {
                ld1rh(vreg.h, P_ALL_ONE,
                        ptr(bcast_ptr(i_reduce, i_ur, true,
                                X_DEFAULT_ADDR, X_TMP_0)));
            }
        }
    };

    auto bcast_load_accord_blk = [=](const int i_reduce)
    {
        for (int i_ur = 0; i_ur < ur; ++i_ur)
        {
            auto vreg = vreg_input_accord_blk(i_ur);
            if (jcp.expl_bcast && load_loop_blk >= 1) {
                ld1rh(vreg.h, P_ALL_ONE,
                        ptr(bcast_ptr(i_reduce, i_ur, true,
                                X_DEFAULT_ADDR, X_TMP_0)));
            }
        }
    };

    auto bcast_load_single = [=](const int iReduce, int iUr, int inPos)
    {
        auto vreg = vreg_input(inPos);
        if (jcp.expl_bcast && load_loop_blk >= 1) {
            ld1rh(vreg.h, P_ALL_ONE,
                    ptr(bcast_ptr(iReduce, iUr, true,
                            X_DEFAULT_ADDR, X_TMP_0)));
        }
    };

    auto kernel_load = [=](int reduce_pos)
    {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            auto vreg = vreg_load(i_load);
            if (i_load + 1 == load_loop_blk && load_dim_tail)
                ld1h(vreg.h, k_load_dim_mask / T_z,
                        ptr(load_ptr(reduce_pos, i_load, X_DEFAULT_ADDR,
                                X_TMP_0)));
            else
                ldr(vreg,
                        ptr(load_ptr(reduce_pos, i_load, X_DEFAULT_ADDR,
                                X_TMP_0)));
        }
    };

    auto fma_sub = [=](bool is_input_accord_blk)
    {
        for (int i_ur = 0; i_ur < ur; ++i_ur)
        {
            auto tmp_reg = is_input_accord_blk == false ? vreg_input(i_ur) : vreg_input_accord_blk(i_ur);
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto vreg_acc = vreg_accum(i_load, i_ur);
                if (i_load + 1 == load_loop_blk && load_dim_tail) {
                    fmla(vreg_acc.h, k_load_dim_mask / T_m,
                            vreg_load(i_load).h, tmp_reg.h);
                } else if (jcp.expl_bcast && load_loop_blk >= 1) {
                    fmla(vreg_acc.h, P_ALL_ONE / T_m, vreg_load(i_load).h,
                            tmp_reg.h);
                } else {
                    fmla(vreg_acc.h, P_ALL_ONE / T_m, vreg_load(i_load).h,
                            tmp_reg.h);
                }
            }
        }
    };

    //fma calculation logic for fixed temporary register
    auto fma_tmp_reg = [=](int ur_pos, bool is_last_reg_tmp){
        auto tmp_reg = is_last_reg_tmp == true ? zreg_tmp : zreg_tmp1;
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            auto vreg_acc = vreg_accum(i_load, ur_pos);
            if (i_load + 1 == load_loop_blk && load_dim_tail) {
                fmla(vreg_acc.h, k_load_dim_mask / T_m,
                        vreg_load(i_load).h, tmp_reg.h);
            } else if (jcp.expl_bcast && load_loop_blk >= 1) {
                fmla(vreg_acc.h, P_ALL_ONE / T_m, vreg_load(i_load).h,
                        tmp_reg.h);
            } else {
                fmla(vreg_acc.h, P_ALL_ONE / T_m, vreg_load(i_load).h,
                        tmp_reg.h);
            }
        }
    };

    //fma calculation logic for regs resue
    auto fma_reg_reuse = [=](int ur_pos, const int all_input_regs_size){
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            auto vreg_acc = vreg_accum(i_load, ur_pos);
            if (i_load + 1 == load_loop_blk && load_dim_tail) {
                fmla(vreg_acc.h, k_load_dim_mask / T_m,
                        vreg_load(i_load).h, vreg_input(ur_pos % all_input_regs_size).h);
            } else if (jcp.expl_bcast && load_loop_blk >= 1) {
                fmla(vreg_acc.h, P_ALL_ONE / T_m, vreg_load(i_load).h,
                        vreg_input(ur_pos % all_input_regs_size).h);
            } else {
                fmla(vreg_acc.h, P_ALL_ONE / T_m, vreg_load(i_load).h,
                        vreg_input(ur_pos % all_input_regs_size).h);
            }
        }
    };

    // for fma reg reuse
    auto deal_fma_reg_reuse = [=](int reduce_pos){
        const int input_regs_size = 4;
        int i_ur = 0;
        for(; i_ur < input_regs_size; ++i_ur)
        {
            bcast_load_single(reduce_pos, i_ur, i_ur % input_regs_size);
        }
        int ur_varia_v = i_ur;
        fma_reg_reuse(i_ur - ur_varia_v, input_regs_size); ur_varia_v--;
        fma_reg_reuse(i_ur - ur_varia_v, input_regs_size); ur_varia_v--;

        bcast_load_single(reduce_pos, i_ur, i_ur % input_regs_size); i_ur++; ur_varia_v++;
        bcast_load_single(reduce_pos, i_ur, i_ur % input_regs_size); i_ur++; ur_varia_v++;
        
        fma_reg_reuse(i_ur - ur_varia_v, input_regs_size); ur_varia_v--;
        fma_reg_reuse(i_ur - ur_varia_v, input_regs_size); ur_varia_v--;
        fma_reg_reuse(i_ur - ur_varia_v, input_regs_size); ur_varia_v--;
        fma_reg_reuse(i_ur - ur_varia_v, input_regs_size); ur_varia_v--;
    };

    //general fma calculation logic for all ur
    auto fma_becommon = [=](int reduce_pos){
        if (jcp.expl_bcast && load_loop_blk >= 1) {
                ld1rh(zreg_tmp.h, P_ALL_ONE,
                        ptr(bcast_ptr(reduce_pos, 0, true,
                                X_DEFAULT_ADDR, X_TMP_0)));
        }                
        for (int i_ur = 1; i_ur < ur; ++i_ur) {
            if (jcp.expl_bcast && load_loop_blk >= 1) {
                ld1rh(zreg_tmp1.h, P_ALL_ONE,
                        ptr(bcast_ptr(reduce_pos, i_ur, true,
                                X_DEFAULT_ADDR, X_TMP_0)));
            }
            fma_tmp_reg(i_ur - 1, true);
            i_ur++;
            if(i_ur >= ur) break;
            if (jcp.expl_bcast && load_loop_blk >= 1) {
                ld1rh(zreg_tmp.h, P_ALL_ONE,
                        ptr(bcast_ptr(reduce_pos, i_ur, true,
                                X_DEFAULT_ADDR, X_TMP_0)));

            }
            fma_tmp_reg(i_ur - 1, false);
        }
        ur % 2 == 0 ? fma_tmp_reg(ur - 1, false) : fma_tmp_reg(ur - 1, true);
    };  

    //finaly optimized fma calculation logic
    auto fma_block_optimize = [=](bool last_block) {
        const int i_reduce_end = reduce_dim_tail && last_block
                ? reduce_dim_tail
                : jcp.reduce_loop_unroll;
        if(load_loop_blk < 2 && jcp.ur >= 10)          // ur = 10, bs = 1
        {
            bcast_load(0);                             //load first reduce input  
            for (int i_reduce = 1; i_reduce < i_reduce_end; i_reduce++) {
                kernel_load(i_reduce - 1);
                bcast_load_accord_blk(i_reduce);     
                fma_sub(false); i_reduce++;
                if(i_reduce >= i_reduce_end) break;
                bcast_load(i_reduce);
                kernel_load(i_reduce - 1);
                fma_sub(true);
            }
            kernel_load(i_reduce_end - 1);
            i_reduce_end % 2 == 0 ? fma_sub(true) : fma_sub(false);
        }
        else if((load_loop_blk < 4 && jcp.ur == 7) ||  // ur = 7, bs = 3, oc = 64， ic = 64
                 (load_loop_blk < 3 && jcp.ur >= 9))   // ur = 9, bs = 2
        {
            for (int i_reduce = 0; i_reduce < i_reduce_end; i_reduce++) {
                kernel_load(i_reduce);
                bcast_load(i_reduce);
                fma_sub(false);
            }
        }
        else if(load_loop_blk < 5 && jcp.ur == 6)     // ur = 6, bs = 4, 4 * 6 + 4 + 4
        {
            for (int i_reduce = 0; i_reduce < i_reduce_end; i_reduce++) {
                kernel_load(i_reduce);
                ur == 6 ? deal_fma_reg_reuse(i_reduce) : fma_becommon(i_reduce);
            }
        }
        else                                          // ur = 5, bs = 5, or ur = 4, bs = 6, or other cases
        {
            for (int i_reduce = 0; i_reduce < i_reduce_end; i_reduce++) {
                kernel_load(i_reduce);
                fma_becommon(i_reduce);
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    prepare_kernel(load_loop_blk);

    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    subs_imm(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll,
            reg_tmp_imm);
    b(LE, reduce_loop_tail);

    L(reduce_loop);
    {
        fma_block_optimize(false);
        add_imm(aux_reg_bcast_data, aux_reg_bcast_data,
                jcp.reduce_loop_bcast_step, reg_tmp_imm);
        add_imm(aux_reg_load_data, aux_reg_load_data, jcp.reduce_loop_load_step,
                reg_tmp_imm);
        subs_imm(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll,
                reg_tmp_imm);
        b(GT, reduce_loop);
    }

    L(reduce_loop_tail);
    fma_block_optimize(true);

    store();
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::generate() {
    preamble();

    sub_imm(X_SP, X_SP, stack_space_needed, X_TMP_0);
    if (jcp.with_binary) {
        const auto zeroed_reg = x15;
        eor(zeroed_reg, zeroed_reg, zeroed_reg);
        str(zeroed_reg, ptr(X_SP, reg_binary_post_op_acc_off));
        str(param1, ptr(X_SP, reg_abi_param1_backup));
    }

    /* Pointers indicate weight, input, and output data */
    ldr(reg_bcast_data, ptr(abi_param1, KDNN_GET_OFF(bcast_data))); // Input
    ldr(reg_load_data, ptr(abi_param1, KDNN_GET_OFF(load_data))); // Weight
    ldr(reg_output_data, ptr(abi_param1, KDNN_GET_OFF(output_data))); // Output

    /* Pointer indicates bias data if the layer has bias option */
    if (jcp.with_bias) ldr(reg_bias_data, ptr(abi_param1, KDNN_GET_OFF(bias_data)));

    /* Get workloads of each loop */
    ldr(reg_load_loop_work, ptr(abi_param1, KDNN_GET_OFF(load_dim)));
    ldr(reg_bcast_loop_work, ptr(abi_param1, KDNN_GET_OFF(bcast_dim)));
    str(reg_bcast_loop_work, ptr(X_SP, reg_bcast_loop_work_offt));
    ldr(reg_reduce_loop_work, ptr(abi_param1, KDNN_GET_OFF(reduce_dim)));

    /* A flag for controlling reduce loop */
    ldr(reg_reduce_pos_flag, ptr(abi_param1, KDNN_GET_OFF(first_last_flag)));
    if (jcp.prop_kind == backward_weights)
        ldr(reg_output_stride, ptr(param1, KDNN_GET_OFF(output_stride)));

    const int load_dim_tail
            = (one_of(jcp.prop_kind, forward_training, forward_inference)
                              ? jcp.oc_without_padding
                              : jcp.load_dim)
            % jcp.load_block;
    if (load_dim_tail) {
        const WReg w_tmp(reg_load_dim_tail_mask.getIdx());
        mov_imm(w_tmp, (1 << load_dim_tail) - 1);
        str(zreg_tmp1, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));
        index(zreg_tmp.h, 0, 1);
        mov(zreg_tmp1.h, 1);
        lsl(zreg_tmp1.h, P_ALL_ONE / T_m, zreg_tmp.h);
        dup(zreg_tmp.h, w_tmp);
        and_(zreg_tmp.d, zreg_tmp.d, zreg_tmp1.d);
        cmpne(k_load_dim_tail_mask.h, P_ALL_ONE, zreg_tmp.h, 0);
        ldr(zreg_tmp1, ptr(X_TRANSLATOR_STACK, -1, MUL_VL));
    }

    auto load_loop_body = [=](int load_loop_blk) {
        if (load_dim_tail) {
            eor(k_load_dim_mask.b, P_ALL_ONE / T_z, k_load_dim_mask.b,
                    k_load_dim_mask.b);
            not_(k_load_dim_mask.b, P_ALL_ONE / T_z, k_load_dim_mask.b);
        }
        subs_imm(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step, reg_tmp_imm);
        if (load_dim_tail) {
            Label no_update_mask;
            b(GE, no_update_mask);
            mov(k_load_dim_mask.b, k_load_dim_tail_mask.b);
            L(no_update_mask);
        }
        bcast_loop(load_loop_blk);
        add_imm(reg_load_data, reg_load_data,
                load_loop_blk * jcp.load_loop_load_step, reg_tmp_imm);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                add_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out,
                        reg_tmp_imm);
                add_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (kdnn_is_out_layout_nxc(jcp)
                                                ? 1
                                                : (jcp.with_dw_conv
                                                                ? jcp.ow
                                                                : jcp.bcast_dim)),
                        reg_tmp_imm);
                if (jcp.with_binary) {
                    const auto oc_off_oprnd = aux_reg_load_data;
                    ldr(oc_off_oprnd, ptr(X_SP, reg_binary_post_op_acc_off));
                    add_imm(oc_off_oprnd, oc_off_oprnd,
                            jcp.load_block * load_loop_blk, X_TMP_0);
                    str(oc_off_oprnd, ptr(X_SP, reg_binary_post_op_acc_off));
                }
                break;
            case backward_data:
                add_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (kdnn_is_out_layout_nxc(jcp) ? 1 : jcp.bcast_dim),
                        reg_tmp_imm);
                break;
            case backward_weights:
                for (int i_load = 0; i_load < load_loop_blk; i_load++)
                    add(reg_output_data, reg_output_data, reg_output_stride);
                break;
            default: assert(!"invalid prop_kind");
        }
    };

    const int simd_w = cpu_isa_traits<sve_256>::vlen / sizeof(float16_t);

    Label load_loop_blk[7];

    // with an implicit load_loop_block          {6, 5, 4, 3, 2,  1}
    static const int ur_cases_fma_embd_bcast[] = {2, 4, 5, 8, 14, 32};
    static const int ur_cases_fma_expl_bcast[] = {4, 5, 6, 9, 14, 32}; //modify 2 to 4, is for ur=4, the best use regs

    const int size_ur_cases_fma = jcp.expl_bcast
            ? sizeof(ur_cases_fma_expl_bcast)
            : sizeof(ur_cases_fma_embd_bcast);

    const int *ur_cases_fma = jcp.expl_bcast ? ur_cases_fma_expl_bcast
                                             : ur_cases_fma_embd_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = size_ur_cases_fma / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            cmp_imm(reg_load_loop_work, simd_w * (label_idx + 1), reg_tmp_imm);
            b(LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    b(LE, load_loop_blk[num_ur_cases]);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp_imm(reg_load_loop_work, 2 * label_idx * simd_w,
                            reg_tmp_imm);
                    b(EQ, load_loop_blk[label_idx - 1]);
                }
                cmp_imm(reg_load_loop_work, label_idx * simd_w, reg_tmp_imm);
                b(GT, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx >= 0; --idx) {
                cmp_imm(reg_load_loop_work, simd_w * (idx + 1), reg_tmp_imm);
                b(GE, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp_imm(reg_load_loop_work, simd_w, reg_tmp_imm);
                b(LE, load_loop_blk[0]);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add_imm(X_SP, X_SP, stack_space_needed, X_TMP_0);

    postamble();
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::prepare_output(const int load_loop_blk, int ur_w)
{
    auto zreg_out_h = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur;
        assert(idx < ker_reg_base_idx);
        return ZRegH(idx);
    };
    const bool out_layout_nxc = kdnn_is_out_layout_nxc(jcp);

    int prev_out_ofs = -1;
    for (int k = 0; k < load_loop_blk; k++)
    {
        for (int j = 0; j < ur_w; j++) {
            fmov(zreg_out_h(j, k));
            if (!is_owb_prefetching(jcp)) {
                size_t aux_output_offset = get_output_offset(out_layout_nxc, k, j);
                std::string op = "LD";
                if (j == 0) {
                    prefetch(op, 2, aux_reg_output_data, aux_output_offset);
                    add_imm(reg_prev_out_addr, aux_reg_output_data, aux_output_offset,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_out_addr, reg_prev_out_addr,
                            aux_output_offset - prev_out_ofs, reg_tmp_imm);
                    prefetch(op, 2, reg_prev_out_addr, 0);
                }
                prev_out_ofs = aux_output_offset;
            }
        }
    }  
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::prepare_input(int ur_w)
{
    const bool bcast_layout_nxc = kdnn_is_bcast_layout_nxc(jcp);
    int prev_in_ofs = -1;
    for(int j = 0; j < ur_w; j++)
    {
        if (!is_owb_prefetching(jcp)) {
            size_t aux_input_offset = get_input_offset(bcast_layout_nxc, j);
            std::string op = "LD";
            if(j == 0)
            {
                prefetch(op, 2, aux_reg_bcast_data, aux_input_offset);
                add_imm(reg_prev_bcast_addr, aux_reg_bcast_data, aux_input_offset, 
                        reg_tmp_imm);
            }
            else
            {
                add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr,
                        aux_input_offset - prev_in_ofs, reg_tmp_imm);
                prefetch(op, 2, reg_prev_bcast_addr, 0);
            }
            prev_in_ofs = aux_input_offset;
        }
    }
}

void kdnn_jit_sve_256_1x1_conv_kernel_f16::prepare_kernel(const int load_loop_blk)
{
    const bool load_layout_nxc = kdnn_is_load_layout_nxc(jcp);
    int prev_load_ofs = -1;
    for(int k = 0; k < load_loop_blk; k++)
    {
        mov(aux_reg_load_data, reg_load_data);
        if (!is_owb_prefetching(jcp)) {
            size_t aux_load_offset = get_load_offset(load_layout_nxc);
            std::string op = "LD";
            if(k == 0)
            {
                prefetch(op, 3, aux_reg_load_data, aux_load_offset);
                add_imm(reg_prev_bcast_addr, aux_reg_load_data, aux_load_offset, 
                        reg_tmp_imm);
            }
            else
            {
                add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr,
                        aux_load_offset - prev_load_ofs, reg_tmp_imm);
                prefetch(op, 3, reg_prev_bcast_addr, 0);
            }
            prev_load_ofs = aux_load_offset;
        }
    }
}

