#include <utility>
#include <iomanip>
#include <iostream>
#include "utils/parallel.hpp"
#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "deconv/deconv.hpp"

namespace deconv {
typedef enum {
    DATA_SRC,
    DATA_WEIGHT,
    DATA_DST,
    DATA_BIAS
} PrintType;
void PrintTensor(const prb_t* prb, const dnn_mem_t& srcdata, PrintType ptype, size_t GSize, size_t DSize, size_t BSize,
                 size_t CSize, size_t HSize, size_t WSize);
}