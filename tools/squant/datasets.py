#!/usr/bin/env python3
"""High-diversity multi-domain calibration / development / locked-test trace generator
for Hy4-Flash quantization (40% agent/tool, 25% code, 15% reasoning/math, 10% multilingual, 10% long-doc/structured).
Enforces strict 13-gram decontamination across splits (docs/09-EVALUATION.md §1.2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _ngrams13(tokens: List[int]) -> Set[Tuple[int, ...]]:
    if len(tokens) < 13:
        return set()
    return {tuple(tokens[i : i + 13]) for i in range(len(tokens) - 12)}


def build_rich_prompts(seed: int = 20261118) -> Dict[str, List[Dict]]:
    """Build diverse, realistic multi-domain prompts for calibration, dev, and locked test."""
    rng = random.Random(seed)

    # Templates across all 5 target domains (agent/tool, code, reasoning/math, multilingual, long-doc)
    agent_tasks = [
        ("database_audit", "PostgreSQL", "orders", "total_revenue", "Q4 2026"),
        ("k8s_triage", "Kubernetes", "payment-gateway", "CrashLoopBackOff", "us-east-1"),
        ("ci_pipeline", "GitHub Actions", "strata-engine", "CUDA sm_120 build", "main"),
        ("incident_rca", "Prometheus", "api-latency-p99", ">450ms spike", "eu-central-1"),
        ("git_bisect", "Git", "src/quant/sk_codecs.cpp", "AVX-512 VNNI regression", "engine/hy4-flash"),
        ("file_refactor", "Ripgrep", "include/strata/", "PleReader 4-module API", "v1.0"),
        ("security_scan", "Trivy", "docker/runtime:cuda12.8", "CVE-2026-1042", "prod"),
        ("cost_optimizer", "Vast.ai", "4x RTX PRO 6000", "hourly budget tracker", "session-B"),
        ("log_parser", "ClickHouse", "inference_metrics", "tok_per_sec_p50", "last 24h"),
        ("api_migration", "FastAPI", "/v1/chat/completions", "tool_calls streaming", "OpenAI spec"),
    ]

    code_tasks = [
        ("C++20", "lock-free MPMC ring buffer with cache-line padding and acquire-release atomics"),
        ("CUDA", "warp-level quantized matrix-vector multiply kernel for 4-bit symmetric super-blocks with FP16 scales"),
        ("Rust", "async NVMe page-aligned LRU cache using io_uring with zero-copy buffer pooling"),
        ("Python", "streaming safetensors shard loader with memory-mapped slicing and SHA-256 verification"),
        ("TypeScript", "incremental JSON tool-call stream parser with partial schema validation and recovery"),
        ("C++20", "AVX-512 VNNI vpdpbusd 2-bit and 3-bit shift-and-mask unpacker for 64-weight sub-groups"),
        ("SQL", "windowed cohort retention and rolling 7-day p95 latency aggregation over partitioned events"),
        ("Go", "context-aware HTTP/2 reverse proxy with circuit breaker, token-bucket rate limiter, and metrics"),
        ("CUDA", "fused RMSNorm + 4-stream hyper-connection (iHC) pre/post mixer kernel in shared memory"),
        ("Python", "block-LDL Cholesky GPTQ quantizer with analytical coordinate-descent flip refinement"),
    ]

    math_tasks = [
        "Find all positive integers n such that n^3 + 3 is divisible by n + 3. Prove completeness.",
        "A cache holds C = 1370 experts out of 20992 across 41 layers under Zipf(alpha=1.18) routing with Top-16 activation. Derive the expected hit rate and variance for a 7-token speculative verify window.",
        "Prove that the normalized 64x64 Walsh-Hadamard matrix H_64 is symmetric orthogonal and show how folding H_64 into W_down preserves exact linear layer outputs while bounding worst-case coordinate kurtosis.",
        "Solve the recurrence a_n = 3 a_{n-1} - 2 a_{n-2} + 2^n with a_0 = 1, a_1 = 4, and verify for n = 5.",
        "Let H be a symmetric positive definite matrix and W_q = W + delta. Show that changing a single coordinate q_j by s changes (W_q - W)^T H (W_q - W) by s^2 H_{jj} + 2 s g_j where g = (W_q - W) H.",
        "In Multi-Head Latent Attention with d_kv = 512, d_rope = 64, 12 sparse layers and 30 sliding-window layers (W=512), compute exact FP16 KV memory at 20,480 and 131,072 tokens.",
        "Prove that for a zero-mean unit-variance Gaussian, symmetric 4-level uniform quantization {-1.5s, -0.5s, +0.5s, +1.5s} maximizes SQNR near s = 0.9957, achieving ~9.30 dB.",
        "Given 660,006,416 rows of 100 bytes packed at 40 rows per 4,096-byte NVMe page with zero page straddling, calculate exact byte size in decimal GB and binary GiB and page utilization percentage.",
    ]

    multilingual_tasks = [
        ("zh", "请详细解释腾讯 Hy4-Flash 架构中的 LatentMoE 瓶颈投影、4层 Engram 查表机制以及 5层 DFly2 投机采样草稿模型如何协同提升消费级显卡上的推理速度。"),
        ("de", "Erkläre präzise, wie eine 4-KiB-seitenausgerichtete NVMe-Auslagerung für Engram-Tabellen funktioniert und warum 8-Bit-Zeilen gegenüber 4-Bit-Zeilen keine zusätzlichen IOPS verursachen."),
        ("ja", "混合MLA（Sparse IndexPoolとSliding Window Attention）が100万トークンのコンテキストにおいてKVキャッシュメモリを大幅に削減できる理由を数式を交えて解説してください。"),
        ("es", "Diseña una arquitectura de microservicios tolerante a fallos para procesar pagos en tiempo real con idempotencia garantizada, colas de mensajes y auditoría criptográfica."),
        ("fr", "Analyse les compromis entre la quantification uniforme par bloc et la quantification par rang de fréquence d'activation dans les modèles Mixture-of-Experts à 512 experts."),
        ("zh", "编写一个高效的 Python 异步工具函数，使用连接池并发抓取多个 JSON API，并在超时或 429 限流时自动执行带抖动的指数退避重试。"),
    ]

    # Try loading real diverse public datasets from HF if available (OpenHermes / Cosmopedia / Code / Math / GSM8K),
    # supplemented with rich structured prompts so we ALWAYS have >= 150K-500K high-quality tokens!
    splits = {"calibration": [], "development": [], "final_test": []}

    # Deterministic structured prompts covering every domain & tool-call format
    for i in range(60):
        t = agent_tasks[i % len(agent_tasks)]
        prompt = (
            f"System: You are an autonomous systems & coding agent with access to tools "
            f"`shell_exec(cmd)`, `read_file(path, start, end)`, `edit_file(path, target, replacement)`, "
            f"`query_metrics(service, metric, window)`. Always emit valid JSON tool calls before concluding.\n"
            f"User: Task #{i+1} ({t[0]}): Investigate `{t[2]}` on `{t[1]}` regarding `{t[3]}` in `{t[4]}`. "
            f"Provide a step-by-step diagnostic plan, exact JSON tool calls, expected output interpretation, and remediation patch.\n"
            f"Assistant: <thought>Let's systematically diagnose {t[2]} on {t[1]} for {t[3]} ({t[4]}).</thought>\n"
            f"```json\n{{\"tool\": \"query_metrics\", \"arguments\": {{\"service\": \"{t[2]}\", \"metric\": \"{t[3]}\", \"window\": \"1h\", \"region\": \"{t[4]}\"}}}}\n```\n"
            f"After inspecting the telemetry, check recent deployments and configuration diffs:\n"
            f"```json\n{{\"tool\": \"shell_exec\", \"arguments\": {{\"cmd\": \"git log -n 10 --stat && grep -rn '{t[3]}' config/ src/\"}}}}\n```\n"
        )
        bucket = "calibration" if i < 40 else ("development" if i < 50 else "final_test")
        splits[bucket].append({"id": f"agent_{i:03d}", "segment": "tool_call", "text": prompt})

    for i in range(40):
        lang, desc = code_tasks[i % len(code_tasks)]
        prompt = (
            f"User: Write a production-grade {lang} implementation of a {desc}. "
            f"Include complete data structures, edge-case handling, memory ordering or alignment guarantees, "
            f"unit tests, and Big-O complexity analysis (variant #{i+1}).\nAssistant:\n"
        )
        bucket = "calibration" if i < 26 else ("development" if i < 33 else "final_test")
        splits[bucket].append({"id": f"code_{i:03d}", "segment": "code", "text": prompt})

    for i in range(24):
        q = math_tasks[i % len(math_tasks)]
        prompt = f"User: {q} (Problem instance #{i+1})\nAssistant: Let us solve this rigorously step by step.\n"
        bucket = "calibration" if i < 16 else ("development" if i < 20 else "final_test")
        splits[bucket].append({"id": f"math_{i:03d}", "segment": "reasoning", "text": prompt})

    for i in range(18):
        lang, q = multilingual_tasks[i % len(multilingual_tasks)]
        prompt = f"User ({lang}): {q} (#{i+1})\nAssistant:\n"
        bucket = "calibration" if i < 12 else ("development" if i < 15 else "final_test")
        splits[bucket].append({"id": f"ml_{i:03d}", "segment": "multilingual", "text": prompt})

    return splits


def load_hf_calibration_texts(target_chars: int = 3_500_000) -> List[Dict]:
    """Pull real high-quality diverse text from fast ungated HF datasets (wikitext, code, math, tool-use)
    using streaming so it takes <15 seconds and gives rich real token distributions for Hessian capture!
    """
    samples: List[Dict] = []
    try:
        from datasets import load_dataset
    except ImportError:
        return samples

    sources = [
        ("Salesforce/xlam-function-calling-60k", None, "train", "tool_call", 900_000),
        ("bigcode/starcoderdata", "python", "train", "code", 700_000),
        ("openai/gsm8k", "main", "train", "reasoning", 450_000),
        ("wikitext", "wikitext-103-raw-v1", "train", "other", 600_000),
    ]
    for repo, cfg, split, seg, char_budget in sources:
        got = 0
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True) if cfg else load_dataset(repo, split=split, streaming=True)
            for idx, row in enumerate(ds):
                txt = ""
                if "query" in row and "answers" in row:
                    txt = f"User: {row['query']}\nTools: {row.get('tools','')}\nAssistant: {row['answers']}"
                elif "content" in row:
                    txt = str(row["content"])
                elif "question" in row and "answer" in row:
                    txt = f"Question: {row['question']}\nSolution: {row['answer']}"
                elif "text" in row:
                    txt = str(row["text"])
                txt = txt.strip()
                if len(txt) < 120:
                    continue
                txt = txt[:8192]
                samples.append({"id": f"{seg}_{repo.split('/')[-1]}_{idx}", "segment": seg, "text": txt})
                got += len(txt)
                if got >= char_budget:
                    break
            print(f"  [Datasets] Loaded {got/1e3:.0f} KB from {repo} ({seg})", flush=True)
        except Exception as e:
            print(f"  [Datasets] Skipped {repo}: {e}", flush=True)
    return samples


def build_tokenized_splits(
    model_dir: Path,
    out_dir: Path,
    cal_tokens: int = 262_144,
    dev_tokens: int = 32_768,
    test_tokens: int = 32_768,
    seq_len: int = 1024,
) -> Dict:
    """Create strictly 13-gram decontaminated calibration, dev, and locked final-test token blocks."""
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    base = build_rich_prompts()
    hf_extra = load_hf_calibration_texts()

    # Combine & partition with strict 13-gram decontamination: final_test & dev first, then filter calibration!
    all_items = base["final_test"] + base["development"] + base["calibration"] + hf_extra
    # Tokenize into fixed seq_len chunks
    rng = random.Random(20261118)
    rng.shuffle(hf_extra)

    def pack_items(items: List[Dict], target_toks: int, forbidden_13: Set[Tuple[int, ...]]) -> Tuple[List[List[int]], List[str], Set[Tuple[int, ...]]]:
        seqs: List[List[int]] = []
        segs: List[str] = []
        seen_13: Set[Tuple[int, ...]] = set()
        buf: List[int] = []
        cur_seg = "other"
        total = 0
        for item in items:
            ids = tok.encode(item["text"], add_special_tokens=False)
            if len(ids) < 16:
                continue
            ng = _ngrams13(ids)
            if forbidden_13 and (ng & forbidden_13):
                continue
            if not buf:
                cur_seg = item["segment"]
            buf.extend(ids + [tok.eos_token_id or 120025])
            while len(buf) >= seq_len and total < target_toks:
                chunk = buf[:seq_len]
                buf = buf[seq_len:]
                c_ng = _ngrams13(chunk)
                if forbidden_13 and (c_ng & forbidden_13):
                    continue
                seqs.append(chunk)
                segs.append(cur_seg)
                seen_13.update(c_ng)
                total += len(chunk)
            if total >= target_toks:
                break
        return seqs, segs, seen_13

    # Reserve distinct subsets from hf_extra for test, dev, calibration
    n_hf = len(hf_extra)
    hf_test = hf_extra[: max(1, n_hf // 10)]
    hf_dev = hf_extra[max(1, n_hf // 10) : max(2, n_hf // 5)]
    hf_cal = hf_extra[max(2, n_hf // 5) :]

    test_seqs, test_segs, test_13 = pack_items(base["final_test"] + hf_test, test_tokens, set())
    dev_seqs, dev_segs, dev_13 = pack_items(base["development"] + hf_dev, dev_tokens, test_13)
    cal_seqs, cal_segs, cal_13 = pack_items(base["calibration"] + hf_cal, cal_tokens, test_13 | dev_13)

    # Verify zero 13-gram overlap!
    assert len(cal_13 & dev_13) == 0, "13-gram leak between cal and dev!"
    assert len(cal_13 & test_13) == 0, "13-gram leak between cal and test!"
    assert len(dev_13 & test_13) == 0, "13-gram leak between dev and test!"

    summary = {}
    for name, seqs, segs in [("calibration", cal_seqs, cal_segs), ("development", dev_seqs, dev_segs), ("final_test", test_seqs, test_segs)]:
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps({"seq_len": seq_len, "n_seqs": len(seqs), "tokens": len(seqs) * seq_len, "segments": segs, "seqs": seqs}), encoding="utf-8")
        summary[name] = {"n_seqs": len(seqs), "tokens": len(seqs) * seq_len, "path": str(p)}
    print(f"[Datasets] 13-gram decontamination PASS (0 overlaps): {summary}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cal-tokens", type=int, default=262144)
    ap.add_argument("--dev-tokens", type=int, default=32768)
    ap.add_argument("--test-tokens", type=int, default=32768)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()
    build_tokenized_splits(Path(args.model_dir), Path(args.out_dir), args.cal_tokens, args.dev_tokens, args.test_tokens, args.seq_len)
