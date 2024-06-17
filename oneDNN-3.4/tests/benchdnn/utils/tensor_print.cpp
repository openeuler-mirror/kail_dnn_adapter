#include <utility>
#include <iomanip>
#include <iostream>
#include "utils/tensor_print.hpp"

namespace deconv {

static inline int64_t SrcOff(const prb_t *prb, int64_t mb, int64_t g, int64_t ic,
        int64_t id, int64_t ih, int64_t iw) {
    return (((mb * prb->ic + g * prb->ic / prb->g + ic) * prb->id + id)
                           * prb->ih
                   + ih)
            * prb->iw
            + iw;
}

static inline int64_t WeiOff(const prb_t *prb, int64_t g, int64_t oc, int64_t ic,
        int64_t kd, int64_t kh, int64_t kw) {
    return ((((g * prb->oc / prb->g + oc) * prb->ic / prb->g + ic) * prb->kd
                    + kd) * prb->kh
                   + kh)
            * prb->kw
            + kw;
}

static inline int64_t BiasOff(const prb_t *prb, int64_t g, int64_t oc) {
    return g * prb->oc / prb->g + oc;
}

static inline int64_t DstOff(const prb_t *prb, int64_t mb, int64_t g, int64_t oc,
        int64_t od, int64_t oh, int64_t ow) {
    return (((mb * prb->oc + g * prb->oc / prb->g + oc) * prb->od + od)
                           * prb->oh
                   + oh)
            * prb->ow
            + ow;
}

void PrintTensor(const prb_t* prb, const dnn_mem_t& srcdata, PrintType ptype, size_t GSize, size_t DSize, size_t BSize,
                 size_t CSize, size_t HSize, size_t WSize) {
    switch (ptype) {
        case DATA_SRC: { std::cout << "------------- DATA_SRC------------" << std::endl;break; }
        case DATA_WEIGHT: { std::cout << "------------- DATA_WEIGHT------------" << std::endl;break; }
        case DATA_DST: { std::cout << "------------- DATA_DST------------" << std::endl;break; }
        case DATA_BIAS: { std::cout << "------------- DATA_BIAS------------" << std::endl;break; }
    }
    for (size_t pg = 0; pg < GSize; pg++) {
        for (size_t pd = 0; pd < DSize; pd++) {
            for (size_t pb = 0; pb < BSize; pb++) {
                for (size_t pc = 0; pc < CSize; pc++) {
                    std::cout << "g="<< pg<< " d=" << pd << " b=" << pb << " c=" <<pc << std::endl;
                    for (size_t ph = 0; ph < HSize; ph++) {
                        for (size_t pw = 0; pw < WSize; pw++) {
                            float res = 0.0;
                            if (DATA_SRC == ptype) {
                                size_t src_off = SrcOff(prb, pb, pg, pc, pd, ph, pw);
                                res = ((float *)srcdata)[src_off];
                            } else  if (DATA_WEIGHT == ptype) {
                                size_t wei_off = WeiOff(prb, pg, pc, pc, pd, ph, pw);
                                res =  ((float *)srcdata)[wei_off];
                            } else  if (DATA_DST == ptype) {
                                const size_t dst_off = DstOff(prb, pb, pg, pc, pd, ph, pw);
                                res =  ((float *)srcdata)[dst_off];
                            } else  if (DATA_BIAS == ptype) {
                                const size_t bia_off = BiasOff(prb, pg, pc);
                                res = ((float *)srcdata)[bia_off];
                            }
                            std::cout<< std::setw(10)<< res;
                        }
                        std::cout << std::endl;
                    }
                }
            }
        }
    }
}

}
