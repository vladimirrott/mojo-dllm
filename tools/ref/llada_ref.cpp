// Reference oracle for mojo-dllm, built on llama.cpp's libllama.
//
//   llada_ref logits   MODEL TOKENS_CSV ROWS_CSV OUT.f32 THREADS
//       Non-causal forward over TOKENS; writes f32 logits for ROWS as [rows, vocab].
//
//   llada_ref generate-dream MODEL PROMPT_TOKENS_CSV GEN_LEN STEPS THREADS
//       Dream's diffusion_generate (Dream-org/Dream generation_utils.py),
//       greedy, alg="entropy", eps=1e-3, with the logit shift applied here:
//       llama.cpp returns raw rows and row p-1 scores position p.
//
//   llada_ref generate MODEL PROMPT_TOKENS_CSV GEN_LEN BLOCK_LEN STEPS THREADS
//       LLaDA's reference sampler (ML-GSAI/LLaDA generate.py, temperature 0,
//       remasking="low_confidence", cfg 0) driven by llama.cpp logits. Prints the
//       final canvas token ids as CSV on stdout.
//
// llama.cpp's own diffusion example is not used for token parity: on the pinned
// commit it never applies cfg/alg_temp, adds Gumbel noise to one row only, and
// samples from the distribution even at temperature 0 (see docs/correctness.md).

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_csv(const char * s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::atoi(item.c_str()));
        }
    }
    return out;
}

struct Ctx {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    int             n_vocab = 0;
};

static Ctx open(const char * path, int n_tokens, int threads) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    Ctx c;
    c.model = llama_model_load_from_file(path, mp);
    if (!c.model) {
        std::fprintf(stderr, "failed to load %s\n", path);
        std::exit(2);
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_tokens;
    cp.n_batch         = n_tokens;
    cp.n_ubatch        = n_tokens;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    c.ctx              = llama_init_from_model(c.model, cp);
    if (!c.ctx) {
        std::fprintf(stderr, "failed to create context\n");
        std::exit(2);
    }
    llama_set_causal_attn(c.ctx, false);
    c.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(c.model));
    return c;
}

// Logits for every position of `tokens`, row-major [n, vocab].
static const float * forward(Ctx & c, const std::vector<int> & tokens) {
    llama_batch b = llama_batch_init((int) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        b.token[i]     = tokens[i];
        b.pos[i]       = (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    b.n_tokens = (int) tokens.size();
    int rc     = llama_decode(c.ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        std::fprintf(stderr, "llama_decode failed: %d\n", rc);
        std::exit(3);
    }
    return llama_get_logits(c.ctx);
}

static int cmd_logits(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: llada_ref logits MODEL TOKENS ROWS OUT THREADS\n");
        return 1;
    }
    auto tokens = parse_csv(argv[3]);
    auto rows   = parse_csv(argv[4]);
    Ctx  c      = open(argv[2], (int) tokens.size(), std::atoi(argv[6]));
    const float * lg = forward(c, tokens);
    FILE * f = std::fopen(argv[5], "wb");
    for (int r : rows) {
        std::fwrite(lg + (size_t) r * c.n_vocab, sizeof(float), c.n_vocab, f);
    }
    std::fclose(f);
    std::fprintf(stderr, "wrote %zu rows x %d vocab\n", rows.size(), c.n_vocab);
    return 0;
}

static int cmd_generate(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: llada_ref generate MODEL PROMPT GEN_LEN BLOCK_LEN STEPS THREADS\n");
        return 1;
    }
    auto prompt    = parse_csv(argv[3]);
    int  gen_len   = std::atoi(argv[4]);
    int  block_len = std::atoi(argv[5]);
    int  steps     = std::atoi(argv[6]);
    if (gen_len % block_len != 0) {
        std::fprintf(stderr, "gen_len must be a multiple of block_len\n");
        return 1;
    }
    int num_blocks = gen_len / block_len;
    if (steps % num_blocks != 0) {
        std::fprintf(stderr, "steps must be a multiple of the block count\n");
        return 1;
    }
    int steps_per_block = steps / num_blocks;
    int n_prompt        = (int) prompt.size();
    std::vector<int> x(prompt);
    Ctx  c = open(argv[2], n_prompt + gen_len, std::atoi(argv[7]));
    char buf[32];
    if (llama_model_meta_val_str(c.model, "tokenizer.ggml.mask_token_id", buf, sizeof(buf)) < 0) {
        std::fprintf(stderr, "model has no tokenizer.ggml.mask_token_id\n");
        return 1;
    }
    const int MASK = std::atoi(buf);
    x.resize(n_prompt + gen_len, MASK);

    for (int nb = 0; nb < num_blocks; nb++) {
        int b0 = n_prompt + nb * block_len;
        int b1 = b0 + block_len;
        int m  = 0;
        for (int i = b0; i < b1; i++) {
            m += x[i] == MASK;
        }
        std::vector<int> transfer(steps_per_block, m / steps_per_block);
        for (int i = 0; i < m % steps_per_block; i++) {
            transfer[i] += 1;
        }
        for (int s = 0; s < steps_per_block; s++) {
            const float * lg = forward(c, x);
            std::vector<std::pair<double, int>> cand;
            std::vector<int> x0(x.size(), -1);
            for (int i = b0; i < b1; i++) {
                if (x[i] != MASK) {
                    continue;
                }
                const float * row = lg + (size_t) i * c.n_vocab;
                int best = 0;
                for (int v = 1; v < c.n_vocab; v++) {
                    if (row[v] > row[best]) {
                        best = v;
                    }
                }
                double mx = row[best], z = 0.0;
                for (int v = 0; v < c.n_vocab; v++) {
                    z += std::exp((double) row[v] - mx);
                }
                x0[i] = best;
                cand.push_back({ 1.0 / z, i });
            }
            int k = std::min(transfer[s], (int) cand.size());
            std::stable_sort(cand.begin(), cand.end(),
                             [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int j = 0; j < k; j++) {
                x[cand[j].second] = x0[cand[j].second];
            }
        }
    }
    for (size_t i = 0; i < x.size(); i++) {
        std::printf(i ? ",%d" : "%d", x[i]);
    }
    std::printf("\n");
    return 0;
}

static int cmd_generate_dream(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: llada_ref generate-dream MODEL PROMPT GEN_LEN STEPS THREADS\n");
        return 1;
    }
    auto   prompt   = parse_csv(argv[3]);
    int    gen_len  = std::atoi(argv[4]);
    int    steps    = std::atoi(argv[5]);
    int    n_prompt = (int) prompt.size();
    Ctx    c        = open(argv[2], n_prompt + gen_len, std::atoi(argv[6]));
    char   buf[32];
    if (llama_model_meta_val_str(c.model, "tokenizer.ggml.mask_token_id", buf, sizeof(buf)) < 0) {
        std::fprintf(stderr, "model has no tokenizer.ggml.mask_token_id\n");
        return 1;
    }
    const int    MASK = std::atoi(buf);
    const double eps  = 1e-3;
    std::vector<int> x(prompt);
    x.resize(n_prompt + gen_len, MASK);
    for (int i = 0; i < steps; i++) {
        std::vector<int> cand;
        for (int p = n_prompt; p < n_prompt + gen_len; p++) {
            if (x[p] == MASK) {
                cand.push_back(p);
            }
        }
        if (cand.empty()) {
            break;
        }
        const float * lg = forward(c, x);
        std::vector<std::pair<double, int>> conf;
        std::vector<int> x0(cand.size());
        for (size_t j = 0; j < cand.size(); j++) {
            const float * row = lg + (size_t) (cand[j] - 1) * c.n_vocab;
            int best = 0;
            for (int v = 1; v < c.n_vocab; v++) {
                if (row[v] > row[best]) {
                    best = v;
                }
            }
            double mx = row[best], z = 0.0;
            for (int v = 0; v < c.n_vocab; v++) {
                z += std::exp((double) row[v] - mx);
            }
            double logz = std::log(z), neg_h = 0.0;
            for (int v = 0; v < c.n_vocab; v++) {
                double lp = (double) row[v] - mx - logz;
                neg_h += std::exp(lp) * lp;
            }
            x0[j] = best;
            conf.push_back({ neg_h, (int) j });
        }
        int n = (int) cand.size();
        if (i < steps - 1) {
            double t = 1.0 - (double) i * (1.0 - eps) / steps;
            double s = 1.0 - (double) (i + 1) * (1.0 - eps) / steps;
            n        = (int) ((double) cand.size() * (1.0 - s / t));
        }
        std::stable_sort(conf.begin(), conf.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int j = 0; j < n; j++) {
            x[cand[conf[j].second]] = x0[conf[j].second];
        }
    }
    for (size_t i = 0; i < x.size(); i++) {
        std::printf(i ? ",%d" : "%d", x[i]);
    }
    std::printf("\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: llada_ref {logits|generate} ...\n");
        return 1;
    }
    if (std::strcmp(argv[1], "logits") == 0) {
        return cmd_logits(argc, argv);
    }
    if (std::strcmp(argv[1], "generate") == 0) {
        return cmd_generate(argc, argv);
    }
    if (std::strcmp(argv[1], "generate-dream") == 0) {
        return cmd_generate_dream(argc, argv);
    }
    std::fprintf(stderr, "unknown command %s\n", argv[1]);
    return 1;
}
