/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example inner_product.cpp
/// > Annotated version: @ref inner_product_example_cpp
///
/// @page inner_product_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute an
/// [Inner Product](@ref dev_guide_inner_product) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops;
/// - Creation of optimized memory format from the primitive descriptor.
///
/// @page inner_product_example_cpp Inner Product Primitive Example
/// @copydetails inner_product_example_cpp_short
///
/// @include inner_product.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void inner_product_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // input channels
            IH = 227, // tensor height
            IW = 227, // tensor width
            OC = 96; // output channels

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {OC, IC, IH, IW};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OC};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and bias tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src and dst. In this
    // example, NCHW layout is assumed.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::nc);

    auto src_mem = memory(src_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Create memory object for user's layout for weights. In this example, OIHW
    // is assumed.
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::oihw}, engine);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);

    // Create memory descriptor for weights with format_tag::any. This enables
    // the inner product primitive to choose the memory layout for an optimized
    // primitive implementation, and this format may differ from the one
    // provided by the user.
    auto inner_product_weights_md
            = memory::desc(weights_dims, dt::f32, tag::any);

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops inner_product_ops;
    inner_product_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr inner_product_attr;
    inner_product_attr.set_post_ops(inner_product_ops);

    // Create inner product primitive descriptor.
    auto inner_product_pd = inner_product_forward::primitive_desc(engine,
            prop_kind::forward_training, src_md, inner_product_weights_md,
            bias_md, dst_md, inner_product_attr);

    // For now, assume that the weights memory layout generated by the primitive
    // and the one provided by the user are identical.
    auto inner_product_weights_mem = user_weights_mem;

    // Reorder the data in case the weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (inner_product_pd.weights_desc() != user_weights_mem.get_desc()) {
        inner_product_weights_mem
                = memory(inner_product_pd.weights_desc(), engine);
        reorder(user_weights_mem, inner_product_weights_mem)
                .execute(engine_stream, user_weights_mem,
                        inner_product_weights_mem);
    }

    // Create the primitive.
    auto inner_product_prim = inner_product_forward(inner_product_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> inner_product_args;
    inner_product_args.insert({DNNL_ARG_SRC, src_mem});
    inner_product_args.insert({DNNL_ARG_WEIGHTS, inner_product_weights_mem});
    inner_product_args.insert({DNNL_ARG_BIAS, bias_mem});
    inner_product_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: inner-product with ReLU.
    inner_product_prim.execute(engine_stream, inner_product_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            inner_product_example, parse_engine_kind(argc, argv));
}
