<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 貢献する人へ

[English](CONTRIBUTING.md) | 日本語

報告とプルリクエストを歓迎します。送る前に、次を読んでください。

## いちばん大切なルール: 型番に依存しない

XeStrata は、開発機の GPU と CPU だけでなく、どの Intel Arc GPU（XMX のないものや内蔵 GPU も）、AVX2 のある x86-64 CPU、
contrib-llvm では NVIDIA と AMD の GPU でも動きます。そのため、次を守ってください。

- **型番・製品名・デバイス ID・ベンダー名で経路を選ばない。** 選ぶのは、ハードウェアが報告する能力からです。
  CPU なら CPUID の機能ビット（`cpu_avx512_ok` のように）、GPU なら行列演算の組み合わせ、サブグループの大きさ、GRF のモード、
  ローカルメモリ、メモリの大きさ、1 回に確保できる最大の大きさです。
- **大きさやしきい値を決め打ちしない。** タイルの形、チャンクの大きさ、キャッシュの大きさ、スレッドの数、スピンの回数などは、
  実行時にその報告から決めます。
- **どうしても測って決める値**は、1 枚のカードで決めません。Intel・NVIDIA・AMD の GPU で測り、小さいデバイスでも動く
  フォールバックを用意し、どこで測って何に依存するかをコメントに書きます。
- **一部のデバイスにない機能**（XMX、大きな GRF のモード、SYCL の拡張）は、使う前に有無を調べ、ない場合にも動く経路を用意します。
- **小さい構成でも確かめる。** 例: `XESTRATA_VRAM_LIMIT_MIB=8192 XESTRATA_MAX_ALLOC_MIB=4096 XESTRATA_NO_XMX=1` と少ない CPU スレッド。
  確かめていない構成は、プルリクエストに `unverified` と書きます。

このルールを守らない変更は、速くても入れません（[AGENTS.md](AGENTS.md#hardware-independence-the-first-rule)）。

## ライセンス

XeStrata のために書いたコードは LGPL 3 以降（LGPL-3.0-or-later）で、組み合わせた XeStrata 全体も同じライセンスで配布します（`COPYING.LESSER`、`COPYING`）。ほかのプロジェクトから来たコードは、下のとおりファイルごとに元のライセンスのままです。
元の Strata は MIT で、xe0.1.39 から XeStrata は LGPL にしました。

各ファイルは、中のコードがどこから来たかでライセンスが決まります。

| ファイル | ライセンス | 先頭の SPDX |
| --- | --- | --- |
| XeStrata のために書いたファイル | LGPL | XeStrata の著作権だけ、`LGPL-3.0-or-later` |
| Strata のファイルを XeStrata が変えたもの、ggml / llama.cpp から書き写したコードを含むもの | 元のコードは MIT のまま、XeStrata の変更とファイル全体は LGPL | 両方の著作権、`LGPL-3.0-or-later AND MIT` |
| Strata のファイルで XeStrata が変えていないもの | MIT のまま | Strata の著作権だけ、`MIT` |

- 新しいファイルには、XeStrata のヘッダーを付けてください。
- 変えていない Strata のファイルに手を入れると、そのファイルは混合になります。
  ヘッダーは `python tools/license_class.py --fix` が直します。
- この分け方は 2026-10-13 からで、xe0.1.43.1-1 より後のリリースに入ります。それ以前は、Strata と ggml から来たコードも
  まとめて LGPL と書いていました。
- 例外: XeStrata の oneMath への変更（フォーク [MistVVK/oneMath](https://github.com/MistVVK/oneMath)）は Apache-2.0、
  intel/llvm へのパッチ（`third_party/main/intel-llvm/patches/`）は Apache-2.0 WITH LLVM-exception です。
- 他のプロジェクトの素材は `third_party/` に、プロジェクトごとのフォルダーとライセンスの文書を付けて置きます。
  自由ソフトウェアでないものは `third_party/nonfree/` に置き、それがなくてもビルドと試験が通るようにします。

詳しくは [AGENTS.md](AGENTS.md#licenses) と `NOTICE` にあります。

## GPL 第 7 条の追加の許可

`NOTICE` は、GPL 第 3 版の第 7 条による追加の許可を与えています。
XeStrata を、GPU や CPU のメーカーが出す計算ライブラリ（oneMKL、cuBLAS など）やその実行時ライブラリ（CUDA の実行時と
ドライバー、oneAPI のコンパイラーの実行時など）とリンクしたものを、配布してよいという許可です。

contrib-llvm のビルドとそのパッケージは、自由ソフトウェアでないこれらのライブラリとリンクします。
この許可がないと、LGPL のままではそれらを配布できません。

XeStrata に入る変更は、この追加の許可を含めた条件で入ります。
さらに、変更を送る人は、次のことを認めてください。

- MistVVK が、XeStrata をほかのライブラリとリンクして配布するために追加の許可が要ると判断したときは、
  MistVVK は GPL 第 7 条による追加の許可を XeStrata に足すことができます。
- 足した許可は、その人が送ったコードにも及びます。

認められないとき、または送るコードにこれらの許可を付けられないとき（ほかのプロジェクトから写したコードなど）は、
原則として受け入れられません。

`bench/XeStrataBenchResults/` に足す計測の記録は、何ともリンクしないので、この節の対象外です。
記録は追加の許可のない LGPL-3.0-or-later で入り、上の同意も要りません。

## 取り込む変更の条件

誰が書いたものでも（メンテナー、プルリクエスト、upstream からの移植）、次の 2 つを満たす変更だけを入れます。

1. **intel/llvm で、free と contrib-llvm の両方式でビルドと試験をした**こと。
   ほかのコンパイラー（Intel oneAPI の icpx など）だけでビルドしたものや、ビルドしていないものは入れません。
   ビルドの方法は [docs/BUILD.ja.md](docs/BUILD.ja.md) と [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
1. **[型番に依存しない](#いちばん大切なルール-型番に依存しない)**こと。
   free の方式で、自由ソフトウェアでないものに頼りません。

満たさない変更はそのまま入れず、参考にして XeStrata のコードとして書き直します。

プルリクエストには次を書いてください。

- ビルドした方式（free、contrib-llvm）
- 動かした GPU と CPU
- 確かめていないことには `unverified`

コミットの前に `tools/lint/run.sh` を通し、変更が関わる既存のテストを動かしてください（[AGENTS.md](AGENTS.md#lints)）。

## 報告の仕方

[issue](https://github.com/MistVVK/XeStrata/issues) に書いてください。
どの報告でも、XeStrata の版、入れ方（ソース版か deb/rpm 版か）、ビルドの方式、GPU と CPU を書いてください。
ユーザー名、個人のパス、ホスト名、IP アドレス、キーは消してから貼ってください。

### 不具合

- 端末を操作できる LLM エージェントがあれば、[復旧と原因調査の指示書](docs/DIAGNOSE.ja.md)と症状を渡し、
  その[報告の形式](docs/DIAGNOSE.ja.md#報告の形式)の報告を添えてください。
- なければ、起きたこと、そのときのコマンドと設定、エンジンのログの終わりを添えてください。
  ログの場所は、使っているモデルの設定の `log` にあります。

### 速さ

- `xestrata --bench` を動かし、データフォルダの `records/bench-<日時>/summary.md` を貼ってください。
  記録はどこにも送られないので、貼る前に中身を確かめてください（[docs/BENCH.ja.md](docs/BENCH.ja.md#xestrata---bench)）。
- 計測の記録そのものを残したいときは、`bench/XeStrataBenchResults/` にフォルダーを足すプルリクエストにしてください（追加の許可のない LGPL-3.0-or-later で入ります）。
