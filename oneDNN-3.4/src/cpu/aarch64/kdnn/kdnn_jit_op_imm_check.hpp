#ifndef KDNN_JIT_OP_IMM_CHECK_HPP
#define KDNN_JIT_OP_IMM_CHECK_HPP

#include "cpu/aarch64/cpu_isa_traits.hpp"

#define LDRMAX 255
#define LDRMIN (-256)
#define STRMAX 255
#define STRMIN (-256)
#define LD1RWMAX 252
#define PRFMMAX 32760
#define PRFMMIN 0
#define PRFWMAX 31
#define PRFWMIN (-32)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// Check the immediate value for LDR(vector) instruction
//
//   The imm9 in the LDR instruction is the optional signed immediate vector
//   offset, in the range -256 to 255, defaulting to 0.
template <typename T, cpu_isa_t isa = sve_512>
bool kdnn_ldr_imm_check(T ofs) {
    int vlen, vlen_shift;
    vlen = cpu_isa_traits<isa>::vlen;
    vlen_shift = cpu_isa_traits<isa>::vlen_shift;
    int shifted_ofs = ofs >> vlen_shift;
    return ((shifted_ofs) <= LDRMAX) && (shifted_ofs >= LDRMIN)
            && ((ofs % vlen) == 0);
}

// Check the immediate value for SDR(vector) instruction
//
//   The imm9 in the STR instruction is the optional signed immediate vector
//   offset, in the range -256 to 255, defaulting to 0.
template <typename T, cpu_isa_t isa = sve_512>
bool kdnn_str_imm_check(T ofs) {
    int vlen, vlen_shift;
    vlen = cpu_isa_traits<isa>::vlen;
    vlen_shift = cpu_isa_traits<isa>::vlen_shift;
    int shifted_ofs = ofs >> vlen_shift;
    return ((shifted_ofs) <= STRMAX) && (shifted_ofs >= STRMIN)
            && ((ofs % vlen) == 0);
}

// Check the immediate value for LD1RW instruction
//
//   The imm6 in the LD1RW instruction is the optional unsigned immediate byte
//   offset, a multiple of 4 in the range 0 to 252, defaulting to 0.
template <typename T>
bool kdnn_ld1rw_imm_check(T ofs) {
    return ((ofs & 0x3) == 0) && (ofs <= LD1RWMAX) && (ofs >= 0);
}

// Check the immediate value for PRFM(immediate) instruction
//
//   The pimm in the PRFM instruction is the optional positive immediate byte
//   offset, a multiple of 8 in the range 0 to 32760, defaulting to 0.
template <typename T>
bool kdnn_prfm_imm_check(T ofs) {
    return (ofs <= PRFMMAX) && (ofs >= PRFMMIN) && ((ofs & 0x7) == 0);
}

// Check the immediate value for PRFW(scalar plus immediate) instruction
//
//   The imm6 in the PRFW instruction is the optional signed immediate vector
//   offset, in the range -32 to 31, defaulting to 0.
template <typename T, cpu_isa_t isa = sve_512>
bool kdnn_prfw_imm_check(T ofs) {
    int vlen, vlen_shift;
    vlen = cpu_isa_traits<isa>::vlen;
    vlen_shift = cpu_isa_traits<isa>::vlen_shift;
    int shifted_ofs = ofs >> vlen_shift;
    return (shifted_ofs <= PRFWMAX) && (shifted_ofs >= PRFWMIN)
            && ((ofs % vlen) == 0);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

