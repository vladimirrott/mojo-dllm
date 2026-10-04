"""Denoising loops: LLaDA's `generate.py` and Dream's `diffusion_generate`.

LLaDA (semi-autoregressive blocks, below) and Dream (one block, timestep
schedule, see `_generate_dream`) share the model call; `generate` picks the
loop from the model's architecture.

LLaDA (ML-GSAI/LLaDA `generate.py`):

    canvas = prompt ++ [MASK] * gen_length
    for each block of `block_length` generated positions, left to right:
        m = masked positions in the block
        spread m over steps/num_blocks steps:  m // s each, +1 for the first m % s
        each step:
            logits = model(canvas)                    # bidirectional, whole canvas
            x0[i]  = argmax(logits[i] (+ Gumbel noise if temperature > 0))
            conf[i] = softmax(logits[i])[x0[i]]       # "low_confidence" remasking
                      or U(0,1)                       # "random" remasking
            commit the k most confident masked positions of the block

Only the masked positions of the current block are candidates, so only their
logits are requested from the model (`full_logits` turns that off, to measure
what the saving is worth). Probabilities are computed in f64, as the reference
does.

Seeds drive this module's own PRNG (splitmix64 + xoshiro256**); a seed here
does not reproduce a PyTorch run. At temperature 0 with low-confidence
remasking the algorithm is deterministic and token-for-token comparable with
the reference.
"""

from std.math import exp, log
from std.time import perf_counter_ns

from mojo_dllm.models.transformer import DiffusionLM
from mojo_dllm.sys.mem import F32Ptr, Floats


struct GenConfig(Copyable, Movable):
    var gen_length: Int
    var block_length: Int
    var steps: Int
    var temperature: Float64
    var remasking: String
    var seed: Int
    var mask_id: Int
    var full_logits: Bool
    var eps: Float64
    """Dream's final timestep: timesteps run linspace(1, eps, steps + 1)."""

    def __init__(out self):
        self.gen_length = 128
        self.block_length = 32
        self.steps = 128
        self.temperature = 0.0
        self.remasking = "low_confidence"
        self.seed = 42
        self.mask_id = 126336
        self.full_logits = False
        self.eps = 1e-3


struct GenResult(Movable):
    var tokens: List[Int]
    var step_ms: List[Float64]
    var forward_passes: Int
    var seconds: Float64

    def __init__(out self):
        self.tokens = List[Int]()
        self.step_ms = List[Float64]()
        self.forward_passes = 0
        self.seconds = 0.0


struct Rng(Movable):
    """xoshiro256** seeded through splitmix64."""

    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    def __init__(out self, seed: Int):
        var x = UInt64(seed)
        self.s0 = Rng._splitmix(x)
        self.s1 = Rng._splitmix(x)
        self.s2 = Rng._splitmix(x)
        self.s3 = Rng._splitmix(x)

    @staticmethod
    def _splitmix(mut x: UInt64) -> UInt64:
        x += 0x9E3779B97F4A7C15
        var z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    @staticmethod
    def _rotl(x: UInt64, k: UInt64) -> UInt64:
        return (x << k) | (x >> (64 - k))

    def next_u64(mut self) -> UInt64:
        var result = Rng._rotl(self.s1 * 5, 7) * 9
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = Rng._rotl(self.s3, 45)
        return result

    def uniform(mut self) -> Float64:
        """Uniform in the open interval (0, 1)."""
        return (Float64(self.next_u64() >> 11) + 0.5) * (
            1.0 / 9007199254740992.0
        )


trait StepObserver:
    """Receives the canvas after every denoising step (for live display)."""

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        ...


struct NoObserver(StepObserver):
    def __init__(out self):
        pass

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        pass


def num_transfer_tokens(m: Int, steps: Int) -> List[Int]:
    var out = List[Int](length=steps, fill=m // steps)
    for i in range(m % steps):
        out[i] += 1
    return out^


def select_top(conf: List[Float64], k: Int) -> List[Int]:
    """Indices of the k largest values; ties go to the lower index."""
    var idx = List[Int](capacity=len(conf))
    for i in range(len(conf)):
        idx.append(i)
    # Insertion sort: candidates per step are at most one block (tens of items).
    for a in range(1, len(idx)):
        var v = idx[a]
        var b = a - 1
        while b >= 0 and (conf[idx[b]] < conf[v]):
            idx[b + 1] = idx[b]
            b -= 1
        idx[b + 1] = v
    var n = min(k, len(idx))
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(idx[i])
    return out^


def argmax_and_prob(row: F32Ptr, vocab: Int) -> Tuple[Int, Float64]:
    """First argmax and its softmax probability, accumulated in f64."""
    var best = 0
    var mx = row.unsafe_load(0)
    for v in range(1, vocab):
        var x = row.unsafe_load(v)
        if x > mx:
            mx = x
            best = v
    var z = Float64(0)
    var m64 = Float64(mx)
    for v in range(vocab):
        z += exp(Float64(row.unsafe_load(v)) - m64)
    return (best, 1.0 / z)


def _gumbel_argmax(
    row: F32Ptr, vocab: Int, temperature: Float64, mut rng: Rng
) -> Int:
    """argmax of exp(logits) / (-log U)^temperature, the reference's noise, in f64.

    Compared in log space, which ranks identically and does not overflow:
    logit - temperature * log(-log U).
    """
    var best = 0
    var best_v = Float64(-1e300)
    for v in range(vocab):
        var u = rng.uniform()
        var score = Float64(row.unsafe_load(v)) - temperature * log(-log(u))
        if score > best_v:
            best_v = score
            best = v
    return best


def generate(
    mut model: DiffusionLM, prompt: List[Int], cfg: GenConfig
) raises -> GenResult:
    var none = NoObserver()
    return generate_observed(model, prompt, cfg, none)


def generate_observed[
    O: StepObserver
](
    mut model: DiffusionLM, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    """Like `generate`, calling `obs.on_step` after every denoising step."""
    if model.cfg.arch == "dream":
        return _generate_dream(model, prompt, cfg, obs)
    return _generate_llada(model, prompt, cfg, obs)


def _generate_llada[
    O: StepObserver
](
    mut model: DiffusionLM, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    if cfg.gen_length <= 0 or cfg.block_length <= 0 or cfg.steps <= 0:
        raise Error(
            "generation length, block length and steps must be positive"
        )
    if cfg.gen_length % cfg.block_length != 0:
        raise Error(
            "generation length "
            + String(cfg.gen_length)
            + " is not a multiple of the block length "
            + String(cfg.block_length)
        )
    var num_blocks = cfg.gen_length // cfg.block_length
    if cfg.steps % num_blocks != 0:
        raise Error(
            "steps "
            + String(cfg.steps)
            + " is not a multiple of the number of blocks "
            + String(num_blocks)
        )
    if cfg.remasking != "low_confidence" and cfg.remasking != "random":
        raise Error("unknown remasking strategy: " + cfg.remasking)
    var steps_per_block = cfg.steps // num_blocks
    var P = len(prompt)
    var L = P + cfg.gen_length
    var vocab = model.cfg.vocab
    var res = GenResult()
    for t in prompt:
        res.tokens.append(t)
    for _ in range(cfg.gen_length):
        res.tokens.append(cfg.mask_id)
    var rng = Rng(cfg.seed)
    var logits = Floats((L if cfg.full_logits else cfg.block_length) * vocab)
    var t_start = perf_counter_ns()

    for b in range(num_blocks):
        var b0 = P + b * cfg.block_length
        var b1 = b0 + cfg.block_length
        var m = 0
        for i in range(b0, b1):
            if res.tokens[i] == cfg.mask_id:
                m += 1
        var schedule = num_transfer_tokens(m, steps_per_block)
        for s in range(steps_per_block):
            var t_step = perf_counter_ns()
            var cand = List[Int]()
            for i in range(b0, b1):
                if res.tokens[i] == cfg.mask_id:
                    cand.append(i)
            if len(cand) == 0:
                break
            var rows = List[Int]()
            if cfg.full_logits:
                for i in range(L):
                    rows.append(i)
            else:
                rows = cand.copy()
            model.forward(res.tokens, rows, logits.ptr())
            res.forward_passes += 1
            var x0 = List[Int](capacity=len(cand))
            var conf = List[Float64](capacity=len(cand))
            for j in range(len(cand)):
                var r = cand[j] if cfg.full_logits else j
                var row = logits.ptr().unsafe_offset(r * vocab)
                var ap = argmax_and_prob(row, vocab)
                var tok = ap[0]
                if cfg.temperature > 0:
                    tok = _gumbel_argmax(row, vocab, cfg.temperature, rng)
                x0.append(tok)
                if cfg.remasking == "random":
                    conf.append(rng.uniform())
                elif cfg.temperature > 0:
                    conf.append(_prob_of(row, vocab, tok))
                else:
                    conf.append(ap[1])
            var committed = List[Int]()
            for j in select_top(conf, schedule[s]):
                res.tokens[cand[j]] = x0[j]
                committed.append(cand[j])
            res.step_ms.append(Float64(perf_counter_ns() - t_step) / 1e6)
            obs.on_step(res.tokens, committed, res.forward_passes, cfg.steps)
    res.seconds = Float64(perf_counter_ns() - t_start) / 1e9
    return res^


def _prob_of(row: F32Ptr, vocab: Int, tok: Int) -> Float64:
    var mx = row.unsafe_load(0)
    for v in range(1, vocab):
        mx = max(mx, row.unsafe_load(v))
    var z = Float64(0)
    for v in range(vocab):
        z += exp(Float64(row.unsafe_load(v)) - Float64(mx))
    return exp(Float64(row.unsafe_load(tok)) - Float64(mx)) / z


def _confidence(
    row: F32Ptr, vocab: Int, algorithm: String
) -> Tuple[Int, Float64]:
    """Greedy token and Dream's confidence for it, in f64.

    entropy:      sum_v p_v log p_v (negative entropy; larger is surer)
    maskgit_plus: p(x0)
    topk_margin:  p(top1) - p(top2)
    """
    var best = 0
    var mx = row.unsafe_load(0)
    for v in range(1, vocab):
        var x = row.unsafe_load(v)
        if x > mx:
            mx = x
            best = v
    var m64 = Float64(mx)
    var z = Float64(0)
    var second = Float64(-1e300)
    for v in range(vocab):
        var x = Float64(row.unsafe_load(v))
        z += exp(x - m64)
        if v != best and x > second:
            second = x
    var p_best = 1.0 / z
    if algorithm == "maskgit_plus":
        return (best, p_best)
    if algorithm == "topk_margin":
        return (best, p_best - exp(second - m64) / z)
    var logz = log(z)
    var neg_h = Float64(0)
    for v in range(vocab):
        var lp = Float64(row.unsafe_load(v)) - m64 - logz
        neg_h += exp(lp) * lp
    return (best, neg_h)


def _generate_dream[
    O: StepObserver
](
    mut model: DiffusionLM, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    """Dream-org/Dream `diffusion_generate`, greedy (temperature 0, alg_temp 0).

    The whole generation region is one canvas. At step i of S, with
    t = 1 - i(1-eps)/S and s = 1 - (i+1)(1-eps)/S, Dream commits
    int(n_masked * (1 - s/t)) of the masked positions with the highest
    confidence, and every remaining one at the last step. Logits are shifted:
    the row before a position scores it.
    """
    if cfg.gen_length <= 0 or cfg.steps <= 0:
        raise Error("generation length and steps must be positive")
    var algorithm = cfg.remasking
    if algorithm == "low_confidence":
        algorithm = "entropy"
    if (
        algorithm != "entropy"
        and algorithm != "maskgit_plus"
        and algorithm != "topk_margin"
    ):
        raise Error(
            "unknown Dream algorithm: "
            + cfg.remasking
            + " (entropy, maskgit_plus, topk_margin)"
        )
    if cfg.temperature > 0:
        raise Error(
            "Dream sampling with temperature > 0 is not implemented yet; use"
            " --temperature 0"
        )
    var P = len(prompt)
    if P == 0:
        raise Error(
            "Dream needs at least one prompt token (logits are shifted)"
        )
    var vocab = model.cfg.vocab
    var res = GenResult()
    for t in prompt:
        res.tokens.append(t)
    for _ in range(cfg.gen_length):
        res.tokens.append(cfg.mask_id)
    var logits = Floats(cfg.gen_length * vocab)
    var t_start = perf_counter_ns()
    for i in range(cfg.steps):
        var t_step = perf_counter_ns()
        var cand = List[Int]()
        for p in range(P, P + cfg.gen_length):
            if res.tokens[p] == cfg.mask_id:
                cand.append(p)
        if len(cand) == 0:
            break
        var rows = List[Int]()
        for p in cand:
            rows.append(model.logit_row(p))
        model.forward(res.tokens, rows, logits.ptr())
        res.forward_passes += 1
        var x0 = List[Int](capacity=len(cand))
        var conf = List[Float64](capacity=len(cand))
        for j in range(len(cand)):
            var r = _confidence(
                logits.ptr().unsafe_offset(j * vocab), vocab, algorithm
            )
            x0.append(r[0])
            conf.append(r[1])
        var n_transfer = len(cand)
        if i < cfg.steps - 1:
            var t = 1.0 - Float64(i) * (1.0 - cfg.eps) / Float64(cfg.steps)
            var s = 1.0 - Float64(i + 1) * (1.0 - cfg.eps) / Float64(cfg.steps)
            n_transfer = Int(Float64(len(cand)) * (1.0 - s / t))
        var committed = List[Int]()
        if n_transfer > 0:
            for j in select_top(conf, n_transfer):
                res.tokens[cand[j]] = x0[j]
                committed.append(cand[j])
        res.step_ms.append(Float64(perf_counter_ns() - t_step) / 1e6)
        obs.on_step(res.tokens, committed, res.forward_passes, cfg.steps)
    res.seconds = Float64(perf_counter_ns() - t_start) / 1e9
    return res^
