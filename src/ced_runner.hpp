#pragma once
#include <cstddef>
#include <functional>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace ced {

// An input leaf to be filled with host data AFTER the gallocr allocates the
// graph (gallocr decides the final address, so data is set post-alloc).
struct GraphInput {
    ggml_tensor* t = nullptr;
    const void*  data = nullptr;
    size_t       nbytes = 0;
};

// build() creates input leaves (registering each in `inputs`), builds the graph,
// and returns the tensors to capture. run_graph allocates the graph on a CPU
// backend, uploads the inputs, computes, and reads each captured tensor's f32
// data into `outs` (parallel to the returned vector).
using BuildFn =
    std::function<std::vector<ggml_tensor*>(ggml_context*, std::vector<GraphInput>&)>;

bool run_graph(int n_threads, const BuildFn& build,
               std::vector<std::vector<float>>& outs);

}  // namespace ced
