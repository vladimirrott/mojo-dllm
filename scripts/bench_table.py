#!/usr/bin/env python3
"""Render the benchmark table from bench/results/*.json into README and docs.

usage: bench_table.py --write     rewrite the tables between the markers
       bench_table.py --check     exit 1 if a table differs from the evidence

No benchmark number in this repository is typed by hand. The tables live
between `<!-- bench:start -->` and `<!-- bench:end -->` and are produced from
the result files that bench/run_bench.py writes. A partial result file (a run
that was stopped) is refused, and so is a runtime that appears in two files.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "README.md", ROOT / "docs" / "benchmarks.md", ROOT / "docs" / "correctness.md"]
START, END = "<!-- bench:start -->", "<!-- bench:end -->"

ROWS = [
    ("mojo-dllm", "**mojo-dllm**", "Mojo"),
    ("mojo-dllm-full-logits", "mojo-dllm, logits for every position (ablation)", "Mojo"),
    ("llama.cpp", "llama.cpp `llama-diffusion-cli`", "C/C++"),
    ("diffuse-cpp", "diffuse-cpp, cache off", "C++"),
    ("diffuse-cpp-cache-entropy-exit", "diffuse-cpp, cache + `entropy_exit` †", "C++"),
]


def model_label(doc: dict) -> str:
    name = pathlib.Path(doc["config"]["models"]["gguf"]).name
    for suffix in (".gguf", ".Q4_K_M", "-Q4_K_M"):
        name = name.replace(suffix, "")
    return name


def load() -> dict[str, dict]:
    """Result files grouped by model: {label: {merged, meta, sources}}."""
    files = sorted((ROOT / "bench" / "results").glob("*.json"))
    if not files:
        raise SystemExit("bench_table: no result files under bench/results")
    groups: dict[str, dict] = {}
    for f in files:
        doc = json.loads(f.read_text())
        if doc.get("partial", True):
            raise SystemExit(f"bench_table: {f.name} is a partial run; finish or delete it")
        label = model_label(doc)
        g = groups.setdefault(label, dict(merged={}, meta={}, sources=[]))
        g["sources"].append(f.name)
        for rt, s in doc["summary"].items():
            if rt in g["merged"]:
                raise SystemExit(f"bench_table: {label} / {rt} appears in more than one result file")
            if s["runs_ok"] == 0:
                continue
            g["merged"][rt] = s
        meta = g["meta"]
        meta.setdefault("machine", doc["machine"])
        meta.setdefault("config", doc["config"])
        meta.setdefault("versions", {}).update(doc["versions"])
        meta.setdefault("reps", doc["reps"])
    return groups


def render() -> str:
    groups = load()
    order = sorted(groups, key=lambda k: (not k.startswith("LLaDA"), k))
    out: list[str] = []
    for label in order:
        g = groups[label]
        merged, meta, sources = g["merged"], g["meta"], g["sources"]
        c = meta["config"]
        m = meta["machine"]
        if out:
            out.append("")
        out += [
            f"**{label}**, Q4_K_M. {m['cpu']}, {c['threads']} threads, {m['ram_gb']} GB RAM, "
            f"`{m['power_profile']}` power profile. Each prompt: {c['gen_length']} generated tokens, "
            f"{c['steps']} denoising steps; median of {meta['reps']} runs x {len(c['prompts'])} prompts.",
            "",
            "| Runtime | ms / step | tokens / s | startup | peak RSS | language |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for key, row_label, lang in ROWS:
            s = merged.get(key)
            if s is None:
                continue
            out.append(
                f"| {row_label} | {s['ms_per_step_median']:,.0f} | {s['tokens_per_s_median']:.2f} | "
                f"{s['startup_s_median']:.1f} s | {s['peak_rss_gb_max']:.1f} GB | {lang} |"
            )
        v = meta["versions"]
        out += [
            "",
            f"Versions: mojo-dllm `{v.get('mojo-dllm')}`, {v.get('mojo')}, llama.cpp `{v.get('llama.cpp')}`"
            + (f", diffuse-cpp `{v['diffuse-cpp']}`" if any(k.startswith("diffuse") for k in merged) else "")
            + f". Evidence: {', '.join(f'`bench/results/{s}`' for s in sources)}.",
        ]
        if "diffuse-cpp-cache-entropy-exit" in merged:
            out += [
                "",
                "† Not the same work: the inter-step cache reuses stale K/V and `entropy_exit` can stop early. "
                "Shown because it is diffuse-cpp's recommended mode.",
            ]
    return "\n".join(out)


def parity() -> tuple[str, str]:
    """Logit and generation parity tables, one section per architecture (newest file each)."""
    files = sorted((ROOT / "bench" / "parity").glob("*.json"))
    if not files:
        raise SystemExit("bench_table: no parity evidence under bench/parity")
    newest: dict[str, tuple[pathlib.Path, dict]] = {}
    for f in files:
        doc = json.loads(f.read_text())
        newest[doc.get("arch", "llada")] = (f, doc)
    lg: list[str] = []
    gn: list[str] = []
    for arch in sorted(newest, key=lambda a: a != "llada"):
        f, doc = newest[arch]
        name = doc.get("model", arch)
        src = f"`bench/parity/{f.name}` (mojo-dllm `{doc['versions']['mojo-dllm']}`)"
        if lg:
            lg.append("")
            gn.append("")
        lg += [
            f"**{name}**",
            "",
            "| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS "
            "| same argmax | top-5 overlap |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in doc["logits"]:
            lg.append(
                f"| {c['seq_len']} tokens | {c['rows']} | {c['median_cosine']:.5f} | "
                f"{c['masked_median_cosine']:.5f} | {c['min_cosine']:.4f} (row {c['worst_rows'][0]}) | "
                f"{c['median_rel_rms']:.3f} | {c['argmax_agree']} / {c['rows']} | {c['mean_top5_overlap']:.2f} / 5 |"
            )
        lg += ["", f"Evidence: {src}."]
        gn += [
            f"**{name}**",
            "",
            "| Prompt | generated | steps | same tokens | first difference |",
            "|---|---:|---:|---:|---:|",
        ]
        for g in doc["generation"]:
            first = "none" if g["first_divergence"] is None else f"position {g['first_divergence']}"
            gn.append(
                f"| {g['prompt']} | {g['gen_length']} | {g['steps']} | "
                f"{g['matching_tokens']} / {g['gen_length']} | {first} |"
            )
        gn += ["", f"Evidence: {src}."]
    return "\n".join(lg), "\n".join(gn)


def quality() -> str:
    files = sorted((ROOT / "bench" / "quality").glob("*.json"))
    if not files:
        return ""
    out: list[str] = []
    for f in files:
        doc = json.loads(f.read_text())
        if out:
            out.append("")
        out += [
            f"**{doc['model']}**, {doc['gen_length']} generated tokens.",
            "",
            "| Prompt | steps | tokens / step | time | repeated words | output (first 120 characters) |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in doc["runs"]:
            snippet = r["text"][:120].replace("\n", " ").replace("|", "\\|")
            out.append(
                f"| {r['prompt']} | {r['steps']} | {r['tokens_per_step']:g} | {r['generate_s']:.0f} s | "
                f"{r['repeated_words']} | {snippet} |"
            )
        out += ["", f"Evidence: `bench/quality/{f.name}` (mojo-dllm `{doc['versions']['mojo-dllm']}`)."]
    return "\n".join(out)


def splice(text: str, table: str, start: str = START, end: str = END) -> str:
    if start not in text or end not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{table}\n{end}{tail}"


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    table = render()
    logit_table, gen_table = parity()
    quality_table = quality()
    bad = []
    for t in TARGETS:
        if not t.exists():
            continue
        cur = t.read_text()
        new = splice(cur, table)
        new = splice(new, logit_table, "<!-- parity:start -->", "<!-- parity:end -->")
        new = splice(new, gen_table, "<!-- genparity:start -->", "<!-- genparity:end -->")
        if quality_table:
            new = splice(new, quality_table, "<!-- quality:start -->", "<!-- quality:end -->")
        if mode == "--write":
            t.write_text(new)
            print(f"bench_table: wrote {t.relative_to(ROOT)}")
        elif new != cur:
            bad.append(str(t.relative_to(ROOT)))
    if bad:
        print("bench_table: out of date with bench/results: " + ", ".join(bad), file=sys.stderr)
        print("            run: python3 scripts/bench_table.py --write", file=sys.stderr)
        return 1
    if mode == "--check":
        print("bench_table: tables match the evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
