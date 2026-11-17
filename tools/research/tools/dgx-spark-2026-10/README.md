# DGX Spark experiments (27 October 2026)

What they were for: the acceptance run of the DGX Spark drop (0.2.7.1): does the engine hold a longer context, is long-context decode
limited by KV streaming, does the fused IQ prompt path give the same answers, how do several clients and multi-turn chats behave on a
unified-memory box. Each script starts a server from a run config, sends greedy requests and writes JSON rows.

| Script | Question |
| --- | --- |
| `dgx_smoke.py` | The drop's gate in one command: setup detection, one benchmark cell, the identity gate and a needle, each with a band. It calls `tools/dgx_bench.py`, which stays in `tools/` because [DGX_SPARK_LINUX.md](../../../../docs/DGX_SPARK_LINUX.md) documents it. |
| `dgx_quality.py`, `dgx_quality24.py`, `dgx_gate_ref.py` | Same greedy prompts through two configs (5, then 24 prompts), or against a recorded set of answers. |
| `dgx_needle.py`, `dgx_longctx.py`, `dgx_128k.py` | Hide a fact in a long prompt and ask for it back, at 12K, 60K and 100K tokens. |
| `dgx_kvresident.py`, `dgx_kvresident2.py` | Is decode at 47K to 56K limited by KV streaming? Compare `--kv-resident` settings. |
| `dgx_concurrent.py`, `dgx_turns.py`, `dgx_warm.py` | Several clients at once; a five-turn conversation; repeats inside one server for the warm state. |
| `dgx_compare.py` | Side-by-side medians of two result folders. |

Run from the repository root, for example `python research/tools/dgx-spark-2026-10/dgx_smoke.py`. The paths inside them were changed on the move so that the
repository root is found three folders up; nothing else was changed. The results these produced are in `bench/results/` (see its index).
