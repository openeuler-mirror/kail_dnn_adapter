#ifndef CPU_AARCH64_KDNN_INNER_PRODUCT_HPP
#define CPU_AARCH64_KDNN_INNER_PRODUCT_HPP

#include "kdnn.hpp"

#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/aarch64/kdnn/kdnn_utils.hpp"
#include "cpu/aarch64/kdnn/kdnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_ip_fwd_resource_t : public resource_t {
    kdnn_ip_fwd_resource_t(const std::unique_ptr<KDNN::InnerProductLayerFWD> &kdnn_ip_prim) noexcept
        : kdnn_ip_obj_(new KDNN::InnerProductLayerFWD{*(kdnn_ip_prim.get())}) {}

    KDNN::InnerProductLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_ip_fwd_resource_t);

private:
    std::unique_ptr<KDNN::InnerProductLayerFWD> kdnn_ip_obj_;
}; // kdnn_ip_fwd_resource_t

struct kdnn_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_inner_product_fwd_pd_t(p) {
            if (p.kdnn_ip_prim_) {
                this->kdnn_ip_prim_.reset(new KDNN::InnerProductLayerFWD{*(p.kdnn_ip_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_tmp_dst_ = p.need_tmp_dst_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_inner_product_fwd_t);
        
        status_t init(engine_t *engine) {          
            const bool ok = is_fwd() && !has_zero_dim_memory()
                        && (set_default_params() == status::success) &&
                        attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops, data_type_t::dnnl_data_type_undef);

            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto src_tag = memory_desc_matches_one_of_tag(src_md_, ndhwc, ncdhw, nhwc, nchw, nwc, ncw, nc);
            auto wei_tag = memory_desc_matches_one_of_tag(weights_md_, any, undef, src_tag);
            auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, nc);

            if (utils::one_of(format_tag::undef, src_tag, dst_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            // Inner product is the same as the matmul n x (c[h[d]w]) * (i[h[d]w]) x o
            // (note that the src c and weights i both correspond to the input
            // channel). KDNN FullyConnectedLayer assumes the chw dimensions of
            // src and ihw dimensions of weights are collapsed, so we need to
            // make sure that they have the same layout. Given that weights are
            // more often fixed, (so reorders can be hoisted) it makes sense to
            // reorder the weights to fit the src.
            if (wei_tag == any || wei_tag == undef) {
                CHECK(memory_desc_init_by_tag(weights_md_, src_tag));
            }

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper bias_d(weights_md(1));
            auto&& ip_fwd = kdnn_utils::convert_to_kdnn_ip_fwd(src_d,
                weights_d, dst_d, bias_d);
            if (!ip_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_ip_prim_.reset(ip_fwd.second);
                CHECK(post_ops_.init(engine, attr_.post_ops_, dst_md_));
                if (post_ops_.has_sum()) {
                    need_tmp_dst_ = true;
                    dst_size_ = dst_d.nelems() * dst_d.data_type_size();
                } else {
                    need_tmp_dst_ = false;
                    dst_size_ = 0;
                }
                return status::success;
            }
        }

        bool need_tmp_dst_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;

        std::unique_ptr<KDNN::InnerProductLayerFWD> kdnn_ip_prim_;

    }; // pd_t

    kdnn_inner_product_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_ip_fwd_resource_t>(pd()->kdnn_ip_prim_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *bia = CTX_OUT_MEM(void *, DNNL_ARG_BIAS);
        void *dst;

        if (pd()->need_tmp_dst_) {
            dst = KDNN::Service::AlignedAlloc(pd()->dst_size_);
        } else {
            dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        }

        return execute_forward(ctx, src, wei, dst, bia);
    }

    // Execute forward with arbitrary src, wei, bias and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst, void *bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::InnerProductLayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_ip_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, wei, dst, bias);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        pd()->post_ops_.execute(ctx, dst);

        if (pd()->need_tmp_dst_) {
            KDNN::Service::Deallocate(dst);
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_inner_product_fwd_t

struct kdnn_ip_bwd_d_resource_t : public resource_t {
    kdnn_ip_bwd_d_resource_t(const std::unique_ptr<KDNN::InnerProductLayerBWDData> &kdnn_ip_prim) noexcept
        : kdnn_ip_obj_(new KDNN::InnerProductLayerBWDData{*(kdnn_ip_prim.get())}) {}

    KDNN::InnerProductLayerBWDData &get_kdnn_obj() const noexcept { return *kdnn_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_ip_bwd_d_resource_t);

private:
    std::unique_ptr<KDNN::InnerProductLayerBWDData> kdnn_ip_obj_;
}; // kdnn_ip_bwd_d_resource_t

struct kdnn_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::
                cpu_inner_product_bwd_data_pd_t;

        pd_t(const pd_t& p) : cpu_inner_product_bwd_data_pd_t(p) {
            if (p.kdnn_ip_prim_) {
                this->kdnn_ip_prim_.reset(new KDNN::InnerProductLayerBWDData{*(p.kdnn_ip_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {

            const bool ok = desc()->prop_kind == prop_kind::backward_data
                    && !has_zero_dim_memory()
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto diff_dst_tag = memory_desc_matches_one_of_tag(diff_dst_md_, nc);
            auto diff_src_tag = memory_desc_matches_one_of_tag(diff_src_md_, ndhwc, ncdhw, nhwc, nchw, nwc, ncw, nc);
            auto wei_tag = memory_desc_matches_one_of_tag(weights_md_, any, undef, diff_src_tag);

            if (utils::one_of(format_tag::undef, diff_dst_tag, diff_src_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            if (wei_tag == any || wei_tag == undef) {
                CHECK(memory_desc_init_by_tag(weights_md_, diff_src_tag));
            }

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper diff_src_d(diff_src_md(0));
            auto&& ip_bwdd = kdnn_utils::convert_to_kdnn_ip_bwd_d(diff_dst_d,
                weights_d, diff_src_d);
            if (!ip_bwdd.first) {
                return status::unimplemented;
            } else {
                kdnn_ip_prim_.reset(ip_bwdd.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::InnerProductLayerBWDData> kdnn_ip_prim_;

    }; // pd_t

    kdnn_inner_product_bwd_data_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_ip_bwd_d_resource_t>(pd()->kdnn_ip_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *weights  = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *diff_src       = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        return execute_forward(ctx, diff_dst, weights, diff_src);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *diff_dst,
            const void *weights, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::InnerProductLayerBWDData &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_ip_bwd_d_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, weights, diff_src);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_inner_product_bwd_data_t

struct kdnn_ip_bwd_w_resource_t : public resource_t {
    kdnn_ip_bwd_w_resource_t(const std::unique_ptr<KDNN::InnerProductLayerBWDWeights> &kdnn_ip_prim) noexcept
        : kdnn_ip_obj_(new KDNN::InnerProductLayerBWDWeights{*(kdnn_ip_prim.get())}) {}

    KDNN::InnerProductLayerBWDWeights &get_kdnn_obj() const noexcept { return *kdnn_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_ip_bwd_w_resource_t);

private:
    std::unique_ptr<KDNN::InnerProductLayerBWDWeights> kdnn_ip_obj_;
}; // kdnn_ip_bwd_w_resource_t

struct kdnn_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::
                cpu_inner_product_bwd_weights_pd_t;

        pd_t(const pd_t& p) : cpu_inner_product_bwd_weights_pd_t(p) {
            if (p.kdnn_ip_prim_) {
                this->kdnn_ip_prim_.reset(new KDNN::InnerProductLayerBWDWeights{*(p.kdnn_ip_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {

            const bool ok = desc()->prop_kind == prop_kind::backward_weights
                    && !has_zero_dim_memory()
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto diff_dst_tag = memory_desc_matches_one_of_tag(diff_dst_md_, nc);
            auto src_tag = memory_desc_matches_one_of_tag(src_md_, ndhwc, ncdhw, nhwc, nchw, nwc, ncw, nc);
            if (utils::one_of(format_tag::undef, diff_dst_tag, src_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            if (src_tag == cn) {
                CHECK(memory_desc_init_by_tag(diff_weights_md_, format_tag::oi));
                CHECK(memory_desc_init_by_tag(src_md_, format_tag::nc));
            } else {
                CHECK(memory_desc_init_by_tag(diff_weights_md_, src_tag));
            }

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper diff_weights_d(diff_weights_md(0));
            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper diff_bias_d(diff_weights_md(1));
            auto&& ip_bwdw = kdnn_utils::convert_to_kdnn_ip_bwd_w(diff_dst_d,
                src_d, diff_weights_d, diff_bias_d);
            if (!ip_bwdw.first) {
                return status::unimplemented;
            } else {
                kdnn_ip_prim_.reset(ip_bwdw.second);
                return status::success;
            }
        }

        std::unique_ptr<KDNN::InnerProductLayerBWDWeights> kdnn_ip_prim_;

    }; // pd_t

    kdnn_inner_product_bwd_weights_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_ip_bwd_w_resource_t>(pd()->kdnn_ip_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *src      = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *diff_weights   = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);
        void *diff_bias      = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        return execute_forward(ctx, diff_dst, src, diff_weights, diff_bias);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *diff_dst,
            const void *src, void *diff_weights, void *diff_bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KDNN::InnerProductLayerBWDWeights &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_ip_bwd_w_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, src, diff_weights, diff_bias);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_inner_product_bwd_weights_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KDNN_INNER_PRODUCT_HPP
