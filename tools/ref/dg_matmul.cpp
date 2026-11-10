// Isolate one CPU matrix product from a dg_ref trace, without loading the model.
// dg_matmul TYPE K N WEIGHT.bin INPUT.f32 ROWS OUTPUT.f32 THREADS [--no-repack]
// TYPE is the numeric GGML type. WEIGHT.bin is the original GGUF tensor bytes.
// Link against the same pinned ggml libraries as dg_ref, plus -lggml-cpu.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool read(const char * path, ggml_tensor * tensor) {
    std::vector<char> bytes(ggml_nbytes(tensor));
    FILE * file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "dg_matmul: cannot open %s\n", path);
        return false;
    }
    const bool size_ok = std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size() && std::fgetc(file) == EOF;
    std::fclose(file);
    if (!size_ok) {
        std::fprintf(stderr, "dg_matmul: %s must hold exactly %zu bytes\n", path, bytes.size());
        return false;
    }
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 9 && !(argc == 10 && std::strcmp(argv[9], "--no-repack") == 0)) {
        std::fprintf(stderr, "usage: dg_matmul TYPE K N WEIGHT.bin INPUT.f32 ROWS OUTPUT.f32 THREADS [--no-repack]\n");
        return 1;
    }
    const int type_id = std::atoi(argv[1]);
    const int64_t k = std::atoll(argv[2]), n = std::atoll(argv[3]), rows = std::atoll(argv[6]);
    const int threads = std::atoi(argv[8]);
    if (type_id < 0 || type_id >= GGML_TYPE_COUNT || k <= 0 || n <= 0 || rows <= 0 || threads <= 0) {
        std::fprintf(stderr, "dg_matmul: invalid type, dimensions or thread count\n");
        return 1;
    }
    const auto type = static_cast<ggml_type>(type_id);
    const int64_t block_size = ggml_blck_size(type);
    if (block_size <= 0) {
        std::fprintf(stderr, "dg_matmul: unsupported type id %d\n", type_id);
        return 1;
    }
    if (k % block_size) {
        std::fprintf(stderr, "dg_matmul: K does not match the %s block size\n", ggml_type_name(type));
        return 1;
    }
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        std::fprintf(stderr, "dg_matmul: CPU backend is unavailable\n");
        return 2;
    }
    auto backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        std::fprintf(stderr, "dg_matmul: CPU backend initialization failed\n");
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto buft = ggml_backend_cpu_buffer_type();
    if (argc != 10) {
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
        if (extra) {
            for (auto entry = extra(dev); *entry; ++entry) {
                if (std::strstr(ggml_backend_buft_name(*entry), "REPACK")) {
                    buft = *entry;
                    break;
                }
            }
        }
    }
    ggml_init_params wp = { ggml_tensor_overhead(), nullptr, true };
    auto wc = ggml_init(wp);
    auto weight = ggml_new_tensor_2d(wc, type, k, n);
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, buft);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!read(argv[4], weight)) {
        return 2;
    }
    ggml_init_params gp = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    auto gc = ggml_init(gp);
    auto input = ggml_new_tensor_2d(gc, GGML_TYPE_F32, k, rows);
    auto output = ggml_mul_mat(gc, weight, input);
    auto graph = ggml_new_graph(gc);
    ggml_build_forward_expand(graph, output);
    auto gb = ggml_backend_alloc_ctx_tensors(gc, backend);
    if (!read(argv[5], input)) {
        return 2;
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dg_matmul: matrix product failed\n");
        return 3;
    }
    std::vector<float> values(n * rows);
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    FILE * file = std::fopen(argv[7], "wb");
    if (!file) {
        std::fprintf(stderr, "dg_matmul: cannot open %s\n", argv[7]);
        return 2;
    }
    const bool written = std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size();
    const bool closed = std::fclose(file) == 0;
    std::fprintf(stderr, "weight buffer %s, %lld rows, %d threads\n", ggml_backend_buft_name(buft),
                 static_cast<long long>(rows), threads);
    ggml_backend_buffer_free(gb);
    ggml_backend_buffer_free(wb);
    ggml_free(gc);
    ggml_free(wc);
    ggml_backend_free(backend);
    return written && closed ? 0 : 2;
}
