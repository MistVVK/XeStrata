<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# V100 PCIe 32 GB ×2: XeStrata 比較記録

2026-10-10、koshikawa-masato による実機測定。
通常版 Strata 0.1.41 と XeStrata xe0.1.40.2.1 を同一マシンで順番に測定しました。
この短い試験では XeStrata の生成速度は通常版を下回り、自動配置時の入力処理は上回りました。
本記録は診断用データであり、原因を確定した修正ではありません。

## 環境と設定

- Ryzen 9 9950X、128 GB RAM、Ubuntu 24.04.5、NVIDIA ドライバ 580.178.04。
- Tesla V100-PCIE-32GB ×2、各150 W制限、冷却ファン稼働。
- GPU 0: PCIe 3.0 x16、GPU 1: PCIe 3.0 x4。対称な接続ではありません。
- XeStrata の転送プローブ: 13.1 / 3.2 GB/s。
- モデル: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S、分割 GGUF 2ファイル、同じ pack、MTP rt、expert profile。
- Strata: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (0.1.41)。
- XeStrata: `94287ef038a70177a8412602da0ea0df4f91e8a4` (xe0.1.40.2.1)。
- XeStrata: contrib-llvm、Intel LLVM v7.1.1、CUDA 12.8、`sm_70` のソースビルド。
- 配布 deb はホストより新しい glibc を要求したため未使用。OS・ドライバは更新していません。

共通設定:

```text
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--max-context 32768 --kv int8 --batch 2
```

XeStrata の primary は `gpu_pci=0000:01:00.0`、secondary は `split_pci=[0000:0a:00.0]`。
`layer_split=auto`、`layer_split=24`、後者に `STRATA_SM70_TABLE=1` を追加した3条件を比較しました。
最適化フラグが個別カーネルで使用されたかはプロファイラでは確認していません。

診断ログ抜粋は [diagnostics.json](diagnostics.json) に保存しています。

## 結果

各3回の算術平均。生成・prefill は tok/s、TTFT は ms。

| Engine / placement | Japanese decode | Code decode | 1,551-token prefill | 1,551-token TTFT |
| --- | ---: | ---: | ---: | ---: |
| Strata 0.1.41 / auto (24+24) | 74.10 | 107.87 | 699.70 | 2,235.39 |
| XeStrata / auto (35+13) | 62.03 | 92.63 | 766.77 | 2,056.72 |
| XeStrata / 24+24 | 63.53 | 97.47 | 671.57 | 2,340.81 |
| XeStrata / 24+24 + SM70 table | 65.10 | 100.10 | 673.73 | 2,333.52 |

## 測定方法と再現

[runs.json](runs.json) は36回分の個別値、[comparison.csv](comparison.csv) は平均値です。
[requests.json](requests.json) に実際のリクエスト本文を順番どおり保存しました。

1. 対象バージョンを起動し、上記設定・同じモデル資材を指定します。
1. 他の推論リクエストを止めます。ビルド処理も終了後に測定しました。
1. 「短く挨拶してください。」を max_tokens=16 で1回送り、ウォームアップとして除外します。
1. requests.json の body を順番に `/v1/chat/completions` へ送信します。
1. SSE の最初の空でない content を受け取るまでを TTFT として計測します。
1. 各リクエストのエンジンログから prefill / decode の値を記録し、終了後0.2秒待ちます。

ウォームアップも temperature=0、seed=42、reasoning_effort=none、stream=true、
stream_options.include_usage=true、同じ model 値です。2並列枠を設定していますが、測定は1会話ずつです。
日本語・コード生成は256出力トークン、長い入力の要約は64出力トークンで打ち切っています。
コードは完成アプリの正しさを評価する試験ではありません。
TTFT はローカル HTTP ストリームのクライアント計測であり、純粋な prefill 時間ではありません。
エンジン統計の reused_tokens は全36回で0です。

## 診断と未検証事項

- 自動配置は K=35、実際の resident experts は21,038。
- 24+24 配置では24,576 resident experts、測定ログの decode cache hit rate は100%。
- 起動ログは `prompt matrix products on oneMath` と表示しています。
- `no XMX for FP16 and BF16` の表示もありますが、続けて joint_matrix / mma_gemm が記載されています。
  これだけで CPU フォールバックや個別カーネルの実行経路は断定できません。
- 8K・16K入力での prefill、詳細な GPU カーネルプロファイル、同時会話の性能は **unverified**。
  今回の1,551トークンの値を最大 prefill 性能と解釈しないでください。
- 別のバージョン・バックエンドを比較しており、モデル出力も完全一致しません。
  分散・クロック・温度を含む長時間の制御試験ではなく、過去のチャット UI 数値とも直接比較できません。
- モデルと profile のチェックサムは本記録に未収録です。
  他環境での再現では同名ファイルでも差が残る可能性があります。

本番は測定後に通常版へ復帰しました。設定・認証情報・私的な会話はこの記録に含めていません。
