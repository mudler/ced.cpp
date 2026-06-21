#include "ced_runner.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>

namespace ced {

bool run_graph(int n_threads, const BuildFn& build,
               std::vector<std::vector<float>>& outs) {
    static ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "ced: cpu backend init failed\n");
        return false;
    }

    // Generous metadata context (no_alloc: data lives in gallocr buffers). The
    // 12-block ViT graph is a few hundred nodes; size for headroom.
    constexpr size_t kGraphSize = 8192;
    size_t mem = ggml_tensor_overhead() * (kGraphSize * 2) +
                 ggml_graph_overhead_custom(kGraphSize, false);
    struct ggml_init_params ip { mem, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return false;

    std::vector<GraphInput> inputs;
    std::vector<ggml_tensor*> capture = build(ctx, inputs);
    if (capture.empty()) {
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    // Mark captured tensors as outputs so the gallocr does NOT reuse their
    // buffers for downstream nodes (intermediates like enc_norm/logits feed
    // later ops and would otherwise read back freed/overwritten memory).
    for (ggml_tensor* t : capture) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    ggml_gallocr_t alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ced: gallocr alloc failed\n");
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Inputs are gallocr-allocated; upload host data now.
    for (const GraphInput& in : inputs)
        ggml_backend_tensor_set(in.t, in.data, 0, in.nbytes);

    ggml_backend_cpu_set_n_threads(backend, n_threads > 0 ? n_threads : 4);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ced: graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    outs.resize(capture.size());
    for (size_t i = 0; i < capture.size(); ++i) {
        size_t n = (size_t)ggml_nelements(capture[i]);
        outs[i].resize(n);
        ggml_backend_tensor_get(capture[i], outs[i].data(), 0, n * sizeof(float));
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

}  // namespace ced
