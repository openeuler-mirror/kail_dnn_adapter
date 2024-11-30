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

#ifndef CPU_AARCH64_JIT_SVE_256_1x1_CONV_KERNEL_F16_HPP
#define CPU_AARCH64_JIT_SVE_256_1x1_CONV_KERNEL_F16_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_generator.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_op_imm_check.hpp"
#include "cpu/aarch64/kdnn/jit/kdnn_jit_primitive_conf.hpp"
#include "common/float16.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

/* Get vector offsets, ofs / VL(VL: 256bits = 32Bytes) */
#define KDNN_F16_1x1_VL_OFS(ofs) ((ofs) >> 4)

struct kdnn_jit_sve_256_1x1_conv_kernel_f16 : public kdnn_jit_generator {
    kdnn_jit_sve_256_1x1_conv_kernel_f16(const kdnn_jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(kdnn_jit_sve_256_1x1_conv_kernel_f16)
    
    static status_t init_conf(kdnn_jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd, 
            const primitive_attr_t &attr, int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const kdnn_jit_1x1_conv_conf_t &jcp);

    kdnn_jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    using reg64_t = const XReg;
    enum {
        typesize = sizeof(float16_t),
        ker_reg_base_idx = 31,
    };
    /* Flags and loop variables */
    reg64_t reg_reduce_pos_flag = x1;
    reg64_t reduce_loop_iter = x2;
    reg64_t reg_bcast_loop_iter = x3;
    reg64_t reg_relu_ns = x20; // For forward
    reg64_t reg_output_stride = x20; // For backward

    /* Pointer */
    reg64_t reg_bcast_data = x5; // Input
    reg64_t reg_load_data = x6; // Weight
    reg64_t reg_output_data = x7; // Output
    reg64_t reg_bias_data = x8; // bias
    reg64_t aux1_reg_bcast_data = x9;
    reg64_t aux_reg_output_data = x10;
    reg64_t aux_reg_bcast_data = x11;
    reg64_t aux_reg_load_data = x12;
    reg64_t reg_prev_bcast_addr
            = x13; // Input: The reg keeps addr accessed by previous ldr inst
    reg64_t reg_prev_out_addr
            = x14; // Output: The reg keeps addr accessed by previous ldr or str inst

    /* Workload */
    reg64_t reg_load_loop_work = x15;
    reg64_t reg_reduce_loop_work = x16;
    reg64_t reg_bcast_loop_work = x17;

    /* Temporay registers */
    reg64_t reg_tmp_imm = x27; // tmp for add_imm
    reg64_t reg_tmp_ofs = x19; // tmp reg to calc bwd wei offset in out_load

    reg64_t reg_load_dim_tail_mask = aux_reg_load_data;

    std::unique_ptr<injector::jit_uni_postops_injector_t<sve_256>>
            postops_injector_;

    constexpr static int isa_simd_width_
            = cpu_isa_traits<sve_256>::vlen / sizeof(float16_t);

    ZReg vreg_bcast = ZReg(31);
    PReg k_load_dim_mask = p2;
    PReg k_load_dim_tail_mask = p3;
    ZReg zreg_tmp = ZReg(31);
    ZReg zreg_tmp1 = ZReg(30);

    constexpr static int reg64_size_ = sizeof(int64_t);
    constexpr static int reg_bcast_loop_work_offt = 0;
    constexpr static int reg_binary_post_op_acc_off = 1 * reg64_size_;
    constexpr static int reg_abi_param1_backup = 2 * reg64_size_;
    constexpr static int stack_space_needed = 5 * reg64_size_;

    template <typename T>
    Xbyak_aarch64::XReg EVEX_compress_addr(const Xbyak_aarch64::XReg &addr,
            const Xbyak_aarch64::XReg &x_tmp, Xbyak_aarch64::XReg base,
            T raw_offt, bool bcast = false) {

        assert(raw_offt <= INT_MAX);
        auto offt = static_cast<int>(raw_offt);

        add_imm(addr, base, offt, x_tmp);
        if (bcast) {
            // addr is the same as addr when bcast is false.
        }
        return addr;
    }

    void prefetch(
            const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load = false;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_aligned = ((ofs & 0xFF) == 0) ? true : false;
        if (cacheline_aligned == true) {
            Prfop op;
            switch (level) {
                case 1: op = (for_load == true) ? PLDL1KEEP : PSTL1KEEP; break;
                case 2: op = (for_load == true) ? PLDL2KEEP : PSTL2KEEP; break;
                case 3: op = (for_load == true) ? PLDL3KEEP : PSTL3KEEP; break;
                default: assert(!"invalid prfop"); break;
            }

            if (kdnn_prfm_imm_check(ofs)) {
                prfm(op, ptr(in, static_cast<int32_t>(ofs)));
            } else {
                add_imm(reg_prev_out_addr, in, ofs, reg_tmp_imm);
                prfm(op, ptr(reg_prev_out_addr));
            }
        } else {
            PrfopSve op_sve;
            switch (level) {
                case 1:
                    op_sve = (for_load == true) ? PLDL1KEEP_SVE : PSTL1KEEP_SVE;
                    break;
                case 2:
                    op_sve = (for_load == true) ? PLDL2KEEP_SVE : PSTL2KEEP_SVE;
                    break;
                case 3:
                    op_sve = (for_load == true) ? PLDL3KEEP_SVE : PSTL3KEEP_SVE;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            if ((KDNN_F16_1x1_VL_OFS(ofs) < PRFWMAX)
                    && (KDNN_F16_1x1_VL_OFS(ofs) >= (-1 * PRFWMAX))) {
                prfw(op_sve, P_ALL_ONE,
                        ptr(in, static_cast<int32_t>(KDNN_F16_1x1_VL_OFS(ofs))));
            } else {
                add_imm(reg_prev_out_addr, in, ofs, reg_tmp_imm);
                prfw(op_sve, P_ALL_ONE, ptr(reg_prev_out_addr));
            }
        }
    }

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate() override;
    static void balance(kdnn_jit_1x1_conv_conf_t &jcp);
    inline bool is_owb_prefetching(const kdnn_jit_1x1_conv_conf_t &jcp) {
        return false;
    }

    inline size_t get_output_offset(
            const bool is_out_layout_nxc, const int i_load, const int i_ur) {
        const size_t i_load_shift = is_out_layout_nxc
                ? jcp.load_block
                : (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) * jcp.load_block;
        const size_t i_ur_shift
                = is_out_layout_nxc ? jcp.load_dim : jcp.load_block;
        return jcp.typesize_out * (i_load * i_load_shift + i_ur * i_ur_shift);
    }

    inline size_t get_input_offset(
        const bool is_in_layout_nxc, const int i_ur)
    {
        const size_t i_ur_shift
                = is_in_layout_nxc ? jcp.load_dim : jcp.bcast_block;
        return jcp.typesize_in * (i_ur * i_ur_shift);
    }

    inline size_t get_load_offset(
        const bool is_load_layout_nxc)
    {
        const size_t i_load_shift = is_load_layout_nxc
                ? jcp.load_block
                : (jcp.with_dw_conv ? jcp.iw : jcp.ic); 
        return jcp.typesize_in * (i_load_shift * jcp.oc_block);
    }

    inline void prepare_output(const int load_loop_blk, int ur_w);
    inline void prepare_input(int ur_w);
    inline void prepare_kernel(const int load_loop_blk);

    Xbyak_aarch64::XReg output_ptr(const bool out_layout_nxc, const int i_load,
            const int i_ur, Xbyak_aarch64::XReg addr);
    void apply_postops(const bool is_out_layout_nxc, const int load_loop_blk,
            const int ur);
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
