<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Intel の GPU での XeStrata

[English](XE.md) | 日本語

この文書は、XeStrata のエンジンを Intel の GPU で動かすための実装と、それをどう確かめたかの記録です。
エンジンが GPU に求めること、移植した計算と検証の一覧をまとめます。
使い方は [README](../README.ja.md) と [詳しい説明](DETAILS.ja.md)、ビルドの方法、setup の振る舞い、要るパッケージは [BUILD](BUILD.ja.md) にあります。

XeStrata xe0.1.40.2 は、Strata のエンジンを Level Zero と SYCL で Intel の GPU に移したものです。版の数字は、取り込んだ upstream の版に合わせています。
移植の元は Strata 0.1.24（`3ce2523c2823687de5372be3af58534f56cbf286`）で、その後 0.1.39（`6f32ec07`）までの変更の一部を取り込んでいます
（[Strata 0.1.38 の取り込み](#strata-0138-の取り込み)）。
同じ SYCL のコードを、intel/llvm で NVIDIA の GPU 向けにもコンパイルします（contrib-llvm のビルド、[ビルド](BUILD.ja.md#ビルド)）。
upstream の CUDA のビルドは廃止してソースも消しました。
CUDA のソースは upstream の Strata（`3ce2523`）に残っていて、Xe の各ソースは移植元の CUDA のファイル名をコメントに書いています。

開発と測定は Intel Arc Pro B70（Xe2、32 GB）で行っています。
記録にある速さは、断りがなければこの GPU のものです。NVIDIA は RTX 4070 と RTX 3070 で確かめています。
エンジンは経路を GPU が報告する能力で選びます（AGENTS.md の最初の規則）。
プリフィルの行列積は、Xe2 の XMX には専用のカーネル（xmx_gemm）を、Arc A シリーズの XMX と NVIDIA の Tensor Core には
報告されたタイルの形で動く joint_matrix の積（mma_gemm）を使い、行列エンジンのない GPU は DP4a の経路を使います。

小さい GPU は、B70 を小さく見せて確かめます。
`STRATA_VRAM_LIMIT_MIB` はエンジンから見える VRAM を、`STRATA_MAX_ALLOC_MIB` は想定する最大の確保量を制限し、
`STRATA_NO_XMX=1` は XMX を使わない経路を選ばせます。
CPU 内蔵のグラフィックス（開発機では UHD 770）は、確認用の 2 つ目の実機として使います。

## Strata 0.1.40 の取り込み

upstream の Strata 0.1.39 から 0.1.40.2（`e8ca9afd`）までの変更を、Windows の実装を除いて Xe に移しています。
AMD（HIP）の変更は、XeStrata の AMD の経路（同じ SYCL のカーネル）に当てはまるものを移しました: AMD の APU の統合メモリ（setup と、エンジンの内蔵 GPU の空き）と、全機種向けだった融合した RMSNorm + RoPE です。
upstream の HIP 専用の速くするためのスイッチは、RX 9060 XT（gfx1200）で入り切りして測り、gfx11（RDNA3）と gfx103x（RDNA2）専用のものは試せる機器がないので移していません（[記録](../bench/results/2026-10-09-upstream-0140/README.md)）。
upstream の `sycl/`（Intel の GPU 向けの別の移植）はコードに入れず、同じ B70 で速さを比べる相手にしています。
NVIDIA の GPU では contrib-llvm のビルドで同じカーネルが動きます（[BUILD](BUILD.ja.md)）。

移したものの主なものは次のとおりです。

- 会話をファイルに保存して戻す（セッションファイル）、使い捨ての要求でチェックポイントを取らない `ckpt=0`、共有の先頭を固定する `pin=N` と `tools/research_run.py`。
- PLE の表は GGUF にある形式のまま読みます: IQ4_NL（既定の表）、Q4_0、Q5_0、Q5_1、Q8_0、FP8（E4M3 とスケール）、BF16。
  upstream の測定では、BF16 の表と比べた行ごとの平均誤差は Q8_0 0.53%、FP8 2.64%、Q5_1 3.78%、Q5_0 4.25%、IQ4_NL 7.60%、Q4_0 8.55% です。
- ネイティブのエキスパートの Q4_0、Q4_1、Q5_0、Q6_K の gate/up の GPU カーネル。CPU のエキスパートの AVX-2 の i-quant と Q2_0 のカーネル。
- 検証の窓とプロンプトの経路で起動の数を減らすカーネルの融合（#783 ほか）。
- `STRATA_PREFILL_CPU_SHARE=auto`（既定）: 短いチャンクで、ほとんど選ばれないエキスパートを空いている CPU のプールで計算します。
- `--kv-grow`: K/V は要求が届いたセルの分だけ VRAM を取り、残りをエキスパートのキャッシュが使います。
  デバイスの仮想メモリの対応づけの粒度が 2 MiB 以上なら既定で有効です（RTX 4070 は有効、B70 は 64 KiB で無効）。
- `--adapt-async`（常駐モードの既定）、`--batch-mtp`（`--batch` と `--mtp` のときの既定）、`--pipeline-windows`（opt-in、[MULTIGPU](MULTIGPU.ja.md)）、`--lookup-chain`（opt-in）。
- setup の追加（`--inspect`、`--source modelscope`、200K の文脈など）とサーバーの変更。

速くするための経路は、B70 か RTX 4070 で速くなり、もう一方で遅くならないものを移しました。
片方だけで速いものは、機器が報告する値で実行時に選ぶか、opt-in にしています（[記録](../bench/results/2026-10-09-upstream-0140/README.md)）。

取り込みは free（intel/llvm）、contrib-icpx（icpx）、contrib-llvm（intel/llvm と CUDA）でビルドし、どれも CTest の 71 件が通ります（`expert_multi_test` は AVX-512 のない CPU で飛ばします）。
AVX-512 の経路は Intel SDE（`-icx`）で `expert_multi_test` が通ります。
`expert_parity --selftest` と `pool_test --selftest` は正準形の pack がないので SDE では動かせず、`unverified` です（AVX2 では通ります）。
B70 では icpx と free のビルドが 8 つの答え（思考あり・なし）をすべて最後まで出し、6 つが一致しました。
プロンプト 1,148 tok/s、その後のデコード 98.6 tok/s、短い会話 77.3 tok/s（icpx）です。
8 GiB・4 GiB・XMX なし・CPU のワーカー 6 つの構成と、RTX 4070 でも答えを最後まで出しました。
UHD 770 では free と icpx のビルドが起動し、短い答えを返しました。
2 枚の Intel の GPU での `--pipeline-windows` は `unverified` です。

## Strata 0.1.38 の取り込み

upstream の Strata 0.1.38（`99f3dbd0b21d1401b3769e0c0d963913607f380b`）までの変更のうち、単一の GPU と Linux に関わるものを Xe に移しています。
upstream の履歴はマージとして残しています。
CUDA、HIP、Windows の実装は移していません。複数の GPU の実装は、層の分割だけを移しています（[MULTIGPU](MULTIGPU.ja.md)）。

移したものは次のとおりです。

- `--coupled-draft`: MTP の下書きを、対象モデルと同じサンプリングで引きます（既定は無効、`--no-coupled-draft` で明示的に無効）。
  下書きの語彙の対応表とペナルティの履歴も含みます。
  サーバーの設定では `coupled_draft` で指定でき、エンジンの引数があればそちらが優先されます。
- ファイルから読むエキスパートの層の `--resident-budget-gib`、`--resident-experts`、`--resident-cpu-experts`。
- `--shared-expert-arena PATH`: 完成したエキスパートの重みを Linux のプロセスの間で共有します。
  データを書き出してから、ファイルのロックの下で完成を公開します。
  読む側は pack の同一性を確かめてから使います。
- ネイティブの PLE キーは Q2_0、IQ3_XXS、IQ4_XS、Q8_0 を受け付けます。
  BF16 の PLE キーが pack にあれば、そちらを使います。
- ネイティブの RoPE は、要求のスケーリングの設定に従います。
- MTP のダウンロードは、固定したチェックポイントかどうかを確かめ、不正な範囲を拒みます。

速くするための経路は、B70 で測って速くならなかったので移していません（[記録](../bench/results/2026-10-04-upstream-0138/README.md)）。

| 経路 | B70 での結果 |
| --- | --- |
| プロンプトの量子化したエキスパートの積（`STRATA_PREFILL_MMQ`） | 既定の XMX の FP16 の経路の 0.34 倍 |
| 融合 MoE（`STRATA_PF_FUSED`） | 既定の経路の 0.25 倍 |
| GDN の key-head の共有（`STRATA_GDN_KEYHEAD`） | 既定の経路の 0.54 倍 |
| hyper-connection の分割読み出し（`STRATA_GR_V3`） | decode が約 18% 遅い |
| CPU のプールの P コアと E コアの配置（`--pool-affinity`） | decode が 1〜4% 遅い |
| K-quant の AVX2 の多トークンの積（`STRATA_KQ256`） | 差はばらつきの範囲 |
| AVX2 の IQ カーネルのソフトウェアのプリフェッチ | 差はばらつきの範囲 |
| ファイルの層のルーティングによる先読み（`STRATA_LOOKAHEAD`） | decode が約 2% 遅い |

量子化した積と融合 MoE は、XMX を使わずに decode と同じ DP4a の内積で計算します。
そのため、XMX のある GPU では既定の経路に勝てません。

取り込みは free（dpclang++）と nonfree（icpx）の両方でビルドし、どちらも CTest の 52 件が通ります。
既定の構成（IQ2_XS、nonfree）の出力は、取り込み前（`bae99bd`）と比べました。
18 トークンのチャットから 32 トークンを生成すると、2 回とも一致しました。
4,095 トークンのプロンプトから 16 トークンを生成すると、取り込み前は 3 回とも同じでした。
取り込み後は 3 回のうち 2 回が一致し、1 回は最初のトークンから違いました（[実行ごとに出力が変わる理由](#実行ごとに出力が変わる理由)）。
AVX-512 の経路は Intel SDE（`-icx`）で、`expert_multi_test` と `native_expert_parity --synthetic`（iq3_xxs/iq4_nl、q4_K/q5_1）が通ります。
`expert_parity` と `pool_test` は正準形の pack がないので飛ばしました。
統合のコミットでは、8 GiB の制限、XMX なし、CPU のワーカー 2 つの構成で、`--coupled-draft` の有無で同じ 16 トークンが出ることを確かめました。
UHD 770 では free のビルドが 2 トークンの MTP の窓で答えました。
nonfree のビルドの 4 トークンの窓は Intel のランタイムの中で止まり、`unverified` です。
すべての量子化の形式での実モデルの確認、プロセスをまたぐ共有アリーナ、常駐の入れ替えも `unverified` です。

## 実行時の約束

エンジンが GPU とホストに求めることと、守っている約束です。

- **BIOS**: 単体の GPU では、Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。
  そうしないと B70 の BAR が割り当てられず、xe ドライバーがつながりませんでした。
- **GPU の選び方**: GPU を 1 枚使います（層の分割では複数、[MULTIGPU](MULTIGPU.ja.md#gpu-の番号)）。Intel の GPU は Level Zero、NVIDIA の GPU は CUDA のバックエンドを通します（OpenCL は同じ Intel の GPU を重ねて見せるので使いません）。
  メーカー名やデバイス ID では選ばず、報告される能力で選びます。
  32 幅のサブグループ、FP16、デバイスとホストの USM が必要です。
  実行ファイルがその GPU 向けのコードを持たないとき（`STRATA_CUDA_ARCHS` なしのビルドと NVIDIA の GPU など）も断ります。
  `STRATA_GPU_PCI`（setup が書きます）は GPU を PCI アドレスで指定します。
  これがなければ、単体の GPU を CPU 内蔵のグラフィックス（Level Zero の integrated フラグ。他のバックエンドの GPU は単体とみなします）より先に選び、
  その中ではメモリの多いもの、同じならば計算ユニットの多いものを選びます（計算ユニットの大きさはメーカーで違うためです）。
  CPU への代替の経路はありません。
  必要なものがない GPU は、理由を示して断ります。
- **キュー**: 1 つのコンテキストが、順序つきの計算キューと転送キューを 1 つずつ持ちます。
  `copy_async` はイベントを返し、`compute_after` がキューをまたぐ依存を入れます。
  読み戻すときは、計算の完了イベントを `copy_async` に渡し、待ってからホストのバッファーを読みます。
  既存の parity のテストは、ホスト USM の別の確保を中継に使ってこの経路を通しています。
  `create_stream` は同じコンテキストに順序つきのキューを足します。CUDA が別のストリームに置いていた作業のためです。
  `destroy_stream` は 1 つを空にして解放し、`finish()` はすべてを空にします。
  カーネルが受け取る `void* stream` は `Runtime::stream` が解決します。
  エンジンのストリーム、計算キュー、または null（計算キュー。CUDA が同期していたところでは同期して完了）のどれかで、それ以外のポインターは断ります。
  別のストリームの作業の順序は、CUDA の非ブロッキングのストリームと同じく、イベントでだけ決まります。
- **確保**: デバイスの確保は 4096 バイト境界にそろえ、バンプアロケーターは容量を確かめ、折り返しません。
- **エキスパートの置き場（アリーナ）**: `PinnedArena` は、2 MB 境界にそろえた匿名のマッピング 1 つで、`MADV_HUGEPAGE` を付けます。
  ローダーが中身を詰める前に `prepare_for_device_copy` でデバイスへのコピー用に登録し、マッピングや登録に失敗したら例外を投げます。
  ホストの USM は使いません。
  Level Zero は `max_mem_alloc_size`（B70 で 32.5 GB）を超える 1 つの USM の確保を断り、Coder 以外のどのモデルのアリーナもそれより大きいためです。
  登録は、デバイスの最大の確保量を超えない、層の始まりで切れる断片に分けます。
  UHD 770（4 GiB）は、それより大きい登録をエラーなしで受け付け、間違ったバイトをコピーしました。
  `registered_bytes` は容量全体、`locked_bytes` は明示的に `mlock` したバイト数（0）です。
  `mlock` も hugetlbfs のプールも使わず、透過的なヒュージページで支えます。
  [B70 での測定](../bench/results/2026-09-30-b70/README.md)では、memlock の上限が 8 MiB のまま、47.46 GiB のアリーナを登録でき、`VmLck` は 0 でした。
  メモリが足りないときの振る舞いと、TLB で決まる CPU のアクセスの速さは `unverified` です。
- **デバイスの能力に合わせる**: カーネルはデバイスの報告に従います。
  1024 個のワークアイテムのワークグループは、それを受け付けるデバイスでだけ使います（UHD 770 は 512 まで）。
  FP64 はそれを持つデバイスでだけ使い、持たないデバイス（UHD 770 と Arc の A シリーズ）では FP32 で足し合わせます。
- **検証の窓での待ち合わせ**: 検証の窓（verify window）の GPU は、ホストのメモリのフラグを回し読みして CPU のエキスパートを待ちます。
  これには、動いているカーネルからホストの書き込みが見える必要があります。
  検証器は起動時にそれを確かめ（`doorbell_visible`）、見えない GPU（UHD 770）では、待つカーネルを使わず、ホストが区切りを順に起動します。
  結果は同じで、遅くなります。
  `STRATA_VERIFY_SEGMENTED=1` と `0` で、どちらかに固定できます。
  検証器の層のエキスパートがすべて VRAM にある窓は、エキスパートの割り当てを GPU が自分で決める別のグラフで動かし、ホストを待ちません（upstream cfd3b72。区切りも要りません）。
  B70 の Coder IQ1_M では 28.0 から 28.5 tok/s になり、logits は同じでした。
  `STRATA_VERIFY_RESIDENT_GRAPH=0` で使わないようにできます。
- **寿命**: コピー元、コピー先、読み手は、最後の使い手が終わるまで生かしておきます。
  コピーの完了イベントだけでは、カーネルがまだコピー先を読んでいる間の再利用は許されません。
  アリーナを壊すときは、USM を解放したり登録を解いたりする前に、持っているキューをすべて空にします。
  USM のポインターは確保したコンテキストで使い、その依存はアプリケーションが管理します
  （[Khronos の USM の説明](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst)）。
- **エラーと待ち**: SYCL の同期的な失敗は CLI まで伝えます。
  非同期のエラーは保持して、待つときに確かめます。
  待ちはドライバーの中でブロックし、見張りのスレッドが、120 秒続いた待ちをその名前とともに報告してプロセスを終えます。
  xe ドライバーは `job_timeout_ms`（ここでは 5 秒）でジョブをリセットするので、正常な待ちはその時間より十分早く戻ります。
  ハングや、安全に後始末できない失敗では、生きている GPU の確保の上を巻き戻さずにプロセスを終えます。
  ドライバーのジョブのタイムアウトは変えず、CTest は外側に 60 秒のタイムアウトを置きます。

parity のテストは、埋め込みとスケールの小さな記録と再生の確認を、実験的な [oneAPI の graph 拡張](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)で続けています。
これは、トークン全体のグラフや CPU と GPU のドアベルの手順を確かめるものではありません。

## 移植した計算

### 要素ごとの計算

Xe のライブラリは、`embedding_gather`、`gdn_gate`、`scale_inplace`、`f32_to_f16_bulk`、`silu_inplace`、`rms_norm_weighted` を提供します。
`elementwise.hpp` のほかの宣言は移植しておらず、ライブラリにはありません。
これらのカーネルと parity の参照は、高速な数学関数と積和の融合（contraction）を切ってコンパイルします。

icpx がコンパイルするものはすべて `-ffp-model=precise` を使います。
icpx の既定は fast で、upstream のホストのコードは g++ でビルドしていたためです。
デバイスのコードはすべて `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` でビルドします。
CUDA は既定で除算と平方根を IEEE の丸めで行いますが、SYCL のデバイスのコードは行いません。
`quantize_act_parity` は、Q8_K の `1/iscale` に出る 1 ulp の差でこれを見つけます。

| 計算 | 参照と、変えていない合格の基準 |
| --- | --- |
| 圧縮された埋め込みの行 | 独立したスカラーの展開、乗算と加算を別々に丸める、ビット単位の一致、符号つきの 0、ガード。108 の行のケースと、記録した実行 |
| GDN のゲート | CPU の double の softplus（`x > 20` の分岐つき）。相対 L1 が `1e-6` 以下、負のゲート |
| SiLU | CPU の double の式。相対 L1 が `1e-7` 以下。Xe は安定な FP32 の式を使う |
| スケーリング | CPU の FP32 の乗算。ビット単位の一致 |
| FP32 から FP16 | 共通の整数の変換。ビット単位の一致。CPU と GPU が変換を共有するので、この確認の独立性は `unverified` |
| RMS ノルム | CPU の FP64 の和と平均、イプシロン、任意の重み。相対 L1 が `1e-6` 未満、有限の出力、元の null の重みの確認 |

テストのケースは足さず、入力、CPU の期待値、許容誤差は元のままです。
NumPy で求めた FP16 の境界の確認も含む `quantize_act_parity` は、B70 で通ります。
GPU の境界を独立に確かめることと、非同期の失敗や寿命をわざと壊す試験は `unverified` です。
ケースを足すには承認が要ります。

### i-quant と MMVQ

`iq_kernels.hpp` はすべて移植しました（`src/kernels/xe/iq_kernels.cpp`）。
q8_1 の量子化、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q3_K、Q2_0 の展開、その MMVQ、
埋め込みの行、プロンプトの経路の gate と up の展開、まとめて計算するネイティブのエキスパートです。
ワープは 32 幅のサブグループにし、その和は CUDA のバタフライの順に取り、CUDA のバイトの組み込み関数は書き下しました。
まとめて計算するエキスパートの SwiGLU は、CUDA の `__expf` の代わりに精度の高い指数関数を使います。

`native_mmvq.hpp` もすべて移植しました（`src/kernels/xe/native_mmvq.cpp`）。
Q2_0、Q3_K、Q4_K、Q5_K、Q6_K、Q4_0、Q5_0、Q8_0、IQ4_NL、IQ4_XS の llama.cpp の MMVQ のアダプター（1〜8 列、正確な配置と upstream の配置）と、振り分けです。
振り分けは、CUDA と同じく、ほかの i-quant を `iq_mmvq` に回します。
1 列の場合も複数列のカーネルを CUDA の 1 列の配置で使い、別の 1 列用のカーネルは置きません。
CUDA のファイルでも、両者はビット単位で同じに保たれていました。
呼び出しには明示的なランタイムのストリームが要り、待たずに投入します。
CUDA の約束が null でないストリームを求めていたのと同じです。

[i-quant の測定](../bench/results/2026-09-30-xe-iq/README.md)では、10 の形式すべてが実際のモデルの行を gguf-py と同じに展開しました（Q2_0 は ggml から写した定義）。
MMVQ は `iq_parity` の基準 `2e-2` の中（4.2e-3〜5.8e-3）で、`iq_parity` の失敗は 0 です。
Q6_K、Q8_0、Q4_K、Q5_K、Q4_0、Q5_0 の MMVQ は、ランダムなブロックで FP64 から `2e-2` 以内です。
2〜8 列は、正確な配置で 1 列の呼び出しとビット単位で同じです。

### デコードのそのほかのカーネル

- `src/kernels/xe/rope.cpp`: `rope.hpp`、`native_rope.hpp`、`mrope.hpp` の M-RoPE の表（Xe のデバイスは 1 つなので、ポインターも 1 つ）
- `src/kernels/xe/router.cpp`: `router_top10.hpp` と `native_router.hpp`
- `src/kernels/xe/bf16_gemv.cpp`: `bf16_gemv.hpp`（ggml の mmvf のカーネルを含む）
- `quantize_act.cpp`、`qsa.cpp`、`kv_q8.cpp`、`kv_q4.cpp`、`qsa_decode_attn.cpp`、`kv_stream.cpp`: 同じ名前のヘッダー。
  QSA の入り口は、CUDA が終了していたところで `DeviceError` を投げます

KV のホスト側のプール（`KvHostPools`）はカーネルが読み書きするので、Xe ではホストの USM でなければなりません。
デバイスへのコピー用に登録しただけのメモリは、カーネルから参照できません。
`native_rope` と `native_router` は、CUDA が高速な数学関数を使っていたところで精度の高い `pow`、`cos`、`sin`、`exp` を使います。
これらの ggml-cuda の参照はツリーにないので、それとの一致は `unverified` です。
[デコードのカーネルの確認](../bench/results/2026-09-30-xe-kernels/README.md)では、`rope_parity`、`router_top10_parity`、`bf16_gemv_parity` が通り、どれも CTest に登録しています。

## B70 で確かめたモデル

どのモデルも同じ手順で確かめました（[手順](../bench/results/2026-09-30-xe-iq3xxs/README.md)）。

- すべての種類のエキスパートの組を含む層での `native_expert_parity` と、`dequant_bf16_test`
- 短いプロンプトと、固定したコミットの llama.cpp（CPU）に対する 96 トークンの貪欲生成の比較
- エキスパートの配置を固定した、3 つの推測の方式
- server（ストリーミング、取り消し、3 往復の会話）
- 26,293 トークンのプロンプト。KV のストリーミングのありとなし

ファイルは[テンソルの一覧](../bench/results/2026-09-30-model-inventory/README.md)で固定しています。

| モデル | 結果 | llama.cpp と比べて、最初に分かれるまでのトークン数（そこでの参照の上位 2 候補の差） | 推測の方式 |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | [動く](../bench/results/2026-09-30-xe-iq2xs/README.md) | 推測なしで 96 のうち 96 | 同じトークン |
| Qwen3.8-Flash-Next IQ3_XXS | [動く](../bench/results/2026-09-30-xe-iq3xxs/README.md) | 6（0.090） | 異なる（下を参照） |
| Qwen3.8-Flash-Next IQ3_S | [動く](../bench/results/2026-09-30-xe-iq3s/README.md) | 61（0.051） | 推測なし = suffix。MTP は同じ僅差のところで異なる |
| Qwen3.8-Flash-Next Q2_0 | [動く](../bench/results/2026-09-30-xe-q2_0/README.md)（ネイティブの pack。AVX-512 の正準形の pack はこの CPU では動かしていない） | 47（0.050） | 同じトークン |
| Qwen3.8-Flash-Next Coder IQ1_M | [動く](../bench/results/2026-09-30-xe-coder-iq1m/README.md) | 20（0.238） | 同じトークン |
| Swift 1.5 IQ2_XS | [動く](../bench/results/2026-09-30-xe-swift-iq2xs/README.md) | 7（0.151） | 推測なし = suffix。MTP は異なる |
| Swift 1.5 IQ3_XXS | [動く](../bench/results/2026-09-30-xe-swift-iq3xxs/README.md) | 62（0.058、3 候補） | 異なる（キャッシュの大きさも実行ごとに違った） |
| Swift 1.5 Q2_0 | [動く](../bench/results/2026-09-30-xe-swift-q2_0/README.md)（注） | 25（0.273） | 推測なし = suffix。MTP は異なる |
| OrcaRouter IQ3_XXS（互換の手順） | **動かしていない**: リポジトリが gated で、ダウンロードに Hugging Face のログインが要る | — | — |

注: Swift 1.5 Q2_0 は、層 13 のエキスパートのテンソルが 2 つのシャードにまたがります。
pack の索引は、テンソルごとにシャードを記録します。
2026-10-02 より前の pack は v4、それより新しいものは XeStrata 独自の xs1 と書きます。
xs1 は upstream の v&lt;N&gt; とは別に番号を振っているので、どちらも他方と取り違えられません。

llama.cpp と分かれるのは、どれも参照の上位 2 候補が僅差のところです。
そこから先は、比べても何も言えません。
ネイティブの経路は、プロンプトの最後のトークンから `--dump-logits` の行を書きます（プロンプトは出力ヘッドなしでまとめて読むため）。
上の記録はその機能より前に取ったので、トークンだけを比べています。

推測の方式で結果が分かれたのは、主に次の理由です。

- MTP の実行では、推測器が VRAM を使うので、キャッシュに入るエキスパートが減ります。
  そのぶん一部のエキスパートが、GPU ではなく CPU で計算されます。
- Qwen IQ3_XXS では、キャッシュが同じでも、suffix の実行と推測なしの実行が分かれます。
  CPU の複数トークン用のエキスパートのカーネル（IQ3_XXS の gate と up、IQ4_NL の down）が、ggml の 1 トークン用の内積と違う順に足すためです。
  CPU を ggml の内積に固定すると（`STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1`）、推測の方式によらずロジットがビット単位で同じになります。
  つまり、複数トークンの窓の GPU 側は差を生んでいません
  （[記録](../bench/results/2026-09-30-xe-iq3xxs/README.md#the-cause-with-the-logits-added-later-on-2026-09-30)）。
  この差は僅差の候補を入れ替えるだけなので、残しています。

### 実行ごとに出力が変わる理由

どのエキスパートを CPU で計算するかはキャッシュの大きさで決まり、それが出力の最後の数ビットを変えます。
`auto` は空いている VRAM からキャッシュの大きさを決めるので、起動ごとの VRAM の揺れでスロットが 1〜2 個動き、出力が変わることがあります
（[記録](../bench/results/2026-10-02-new-machine/README.md#outputs-that-change-from-run-to-run)）。
64 スロット単位に切り捨てれば同じ機械では同じキャッシュになりますが、RTX 4070 でデコードが 3〜4% 遅くなったので、速さを取って切り捨てていません。
同じ出力が要るとき（機械や設定をまたいだ比較も）は、`--expert-cache N` で固定します。

### 出力ヘッドの VRAM の確保が断られる

起動時に、出力ヘッドを GPU に上げる確保が断られることがあります。
そのとき空いている VRAM は約 27 GiB あり、いつもアリーナをデバイスへのコピー用に登録した直後の、最初のデバイスの確保です。
同じ設定でもう一度起動すると、普通に動きます。
新しい開発機では、約 120 回の起動のうち 3 回起きました
（[記録](../bench/results/2026-10-02-new-machine/README.md#the-output-heads-vram-refusal)）。
原因は `unverified` です。

### 小さい GPU での VRAM の予約

小さい GPU では（upstream #496）、既定の 700 MiB の予約のせいで、プロンプトの経路に 256 トークンの塊を貸すだけのスロットがキャッシュに残らないことがあります。
そのときエンジンは、それだけ残るところまで予約を下げます。
下限は 300 MiB で、`--vram-reserve-mib` で指定した予約より下げることはありません。
それでも足りなければ起動を止め、何 MiB 足りないか、何を減らせば空くかを示します。
`STRATA_VRAM_LIMIT_MIB` で確かめました。
4,608 MiB（IQ2_XS、32K、MTP）では予約が 482 MiB になり、必要な 368 スロットが残りました。
4,096 MiB では、330 MiB 足りずに止まりました。

## 外れたエキスパートの CPU と PCIe への分け方

検証の窓で VRAM にないエキスパートは、CPU が計算するか、GPU が PCIe 越しに読んで計算します。
どちらに何個回すかを、起動時に測った時間で決めます（[Project Maya](https://github.com/mw00/project-maya) の CPU lane の考え方。コードは書き起こし）。
以前は、リンクの速さだけから決めた割合（20 GB/s 以上で 0.55、それより遅ければ速さに比例）でした。

- **測るもの**: CPU のプールがエキスパート 1 個を計算する時間（64 個を順に回し、キャッシュに収まらないようにする）、
  窓の読み込みのカーネル（`fetch_blobs`）が PCIe 越しに 1 個読む時間、窓のエキスパートのカーネルが VRAM の 1 個を計算する時間、
  CPU がエキスパートを読むメモリの帯域。
- **取り合い**: CPU と PCIe は同じメモリを読みます。2 つの速さの合計がメモリの帯域を超えるとき、両方がその比で遅くなるとみなします。
  B70 では、PCIe の 1 個が単独の 0.066 ms から 0.13 ms になる見積もりで、デコードから逆算した値と合いました。
- **分け方**: 窓のグループごとに、GPU が VRAM 上のエキスパートを計算した後（読み込みを並べられる GPU では並べて）PCIe の分を読んで計算し、
  CPU が残りを計算するとして、遅い方が最も早く終わる数を選びます。
  CPU だけの場合より 10% 以上早く終わる見積もりのときだけ PCIe に回します。
- **領域**: Level Zero では、デバイスへのコピー用に登録したエキスパートの領域を GPU のカーネルが読めません。
  分け方が PCIe に回すときだけ、領域を層ごとにホスト USM へ移します（IQ3_XXS の 40 GB で約 7.5 秒）。`STRATA_ARENA_HOST_USM=1` は最初からホスト USM にします。
- **割合のまま**: `--pcie-frac`（`--calibrate` が保存した値を含む）を渡したとき、層の分割、`--pcie-mode dma` と `direct`、測れなかったときは、以前の割合を使います。

RTX 4070（PCIe 4.0 x4）では外れを全部 CPU に回し、IQ3_XXS 32K で 49.9 → 57〜58.6 tok/s、UD-Q4_K_XL 32K で 22.0 → 25.7 tok/s です。
B70 も全部 CPU（以前と同じ）で、RX 9060 XT は PCIe に回す数が 1 層 4.7 個から 1.0 個に減り、どちらも速さは変わりません
（[記録](../bench/results/2026-10-10-maya-split/README.md)）。
PCIe に回して速くなる GPU（速いリンクと遅い CPU の組み合わせ）では確かめていません（`unverified`）。

## 画像

画像のエンコーダー（`tools/vision`。llama.cpp の mtmd とモデルの mmproj）は、既定で GPU で動きます（`./setup.sh --vision yes` または `gpu`）。
GPU で動かせないときは CPU のエンコーダーに切り替わり、`--vision cpu` では CPU のエンコーダーだけを使います。
CPU のエンコーダーは、[画像の記録](../bench/results/2026-09-30-xe-vision-cpu/README.md)の画像を、2 つの形の格子で llama.cpp と同じに読みます。
動くモデルはすべて画像でも確かめました（[記録](../bench/results/2026-09-30-xe-vision-models/README.md)）。
2 つの形の格子、1 つの要求に 2 枚の画像、同じ画像をもう一度送ってそのキャッシュを使う会話です。

GPU のエンコーダーには 2 つのビルドがあります。
どちらも B70 を PCI アドレスで選びます（`strata-vision --gpu-pci`、設定の `vision.gpu_pci`）。
Vulkan では、最初の GPU が CPU 内蔵のグラフィックスやほかの GPU のこともあるためです。

- **Vulkan**（ggml-vulkan、`-DSTRATA_VISION_VULKAN=ON`）: free のビルドのものです。
  ドライバーは自由ソフトウェアの Mesa です。
  CPU のエンコーダー（8 スレッド）より 7〜9 倍速く、SYCL のものの約半分の速さです。
  埋め込みは CPU のエンコーダーと 2.4〜3.1%（相対 L2）違い、SYCL のものより差が小さく、毎回ビット単位で同じになります
  （[記録](../bench/results/2026-10-02-vision-vulkan/README.md)）。
- **SYCL**（ggml-sycl、`-DSTRATA_VISION_SYCL=ON`）: contrib-icpx（`--license contrib-icpx`）で、oneMKL が入っているときに使います。
  ggml-sycl は、自由ソフトウェアでない oneMKL をリンクします。
  すべてのノードが GPU で動き、CPU のエンコーダーより 13〜20 倍速く、答えは同じか、僅差のところで言い回しが変わる程度です。
  埋め込みは CPU のエンコーダーと 3〜6%（相対 L2）違い、その差に許容範囲を決めずに採用しました
  （[記録](../bench/results/2026-09-30-xe-vision-sycl/README.md)）。

ggml-sycl は、ビルドのときに oneDNN（`intel-oneapi-dnnl-devel`）が入っていればそれを使います。
oneDNN を使うと SYCL のエンコーダーは画像 1 枚あたり 1.2〜1.5 倍速くなりますが、同じ画像でも実行ごとに埋め込みがわずかに変わります。
そのため、選んだときだけ使います。
`./setup.sh --vision-onednn on` は設定に `"onednn": true` を書き、server の `--vision-onednn on|off` はその起動に限って上書きします。
server はエンコーダーのために `GGML_SYCL_ENABLE_DNN` と `GGML_SYCL_FA_ONEDNN` を設定し、off なら oneDNN なしのビルドと同じビットになります。

server は、どれかが準備できるまで、エンコーダーを次の順に起動します。

1. GPU のエンコーダー（選んでいれば oneDNN つき）
1. oneDNN なしの GPU のエンコーダー
1. CPU のエンコーダー（設定の `vision.fallback`）

どれが使われ、それより前のものがなぜ起動しなかったかは、コンソールと `/v1/status`（`vision.encoder`、`vision.fallback`）に出ます。
どれも起動しないとき、または使っていたエンコーダーが後で止まったときも、server は文章の応答を続け、`vision.error` に理由を書きます。

## IQ2_XS のテンソルの構成

[モデルの一覧](../bench/results/2026-09-30-b70/iq2-xs-baseline.json)は、Hugging Face のリビジョンと、各 GGUF のヘッダー全体の SHA-256 を記録しています。
[テンソルの CSV](../bench/results/2026-09-30-b70/iq2-xs-tensors.csv) は、各テンソルの実際の型、形、オフセット、バイト数を記録しています。
この一覧のために取ってきたのはヘッダーだけで、ヘッダーの先頭部分はモデルのシャードとしては使えません。
モデル全体での生成、プロンプトの経路、MTP とプロンプト照合の推測、server、KV のストリーミングつきの 26k トークンのプロンプトは、
[IQ2_XS の記録](../bench/results/2026-09-30-xe-iq2xs/README.md)にあります。
貪欲生成のトークンは CPU の llama.cpp と比べました。
ロジットの単位での比較と pack のハッシュは `unverified` です。

記録したリビジョンの `IQ2_XS` の配布物には、12 の格納形式で 1,224 のテンソルがあり、**IQ2_XS で格納したテンソルは 1 つもありません**。
この名前は、精度の混ざったモデル全体の量子化の目安を表しています。

| テンソルの役割 | 実際の格納形式 | 必要な計算と参照 |
| --- | --- | --- |
| ルーティングされるエキスパートの gate と up | 34 層で IQ2_S、11 層で IQ2_XXS、3 層で IQ1_M | 形式ごとの展開、活性化の量子化、選んだ行の GEMV とまとめた GEMM。固定した ggml の `to_float` と CPU の FP64 |
| ルーティングされるエキスパートの down | 48 層すべてで Q2_0 | ネイティブの Q2_0 の展開と内積、CPU の AVX2 の経路。同じ ggml のエキスパートの参照 |
| 量子化された密な重みと共有エキスパート | IQ3_S、IQ4_XS、IQ4_NL、Q6_K、Q8_0、Q2_0 | 型ごとの展開、密な GEMV と GEMM、SwiGLU。固定した ggml と FP64 の累積 |
| トークンの埋め込みと出力ヘッド | IQ4_XS | 行の展開、射影、サンプリング。固定した ggml の展開とホストの計算 |
| HC、ルーター、SSM、インデクサー、PLE の射影 | BF16 | BF16 の活性化の丸めと射影。既存の BF16、GR、ルーター、GDN、QSA の parity |
| ノルム、バイアス、再帰のパラメーター | F32。PLE の畳み込みは F16 | RMS、ゲート、畳み込み、再帰、索引。既存のホストの参照 |
| PLE の n-gram の表（2 つ目のシャード） | IQ4_NL | 整数の n-gram のハッシュ、選んだ行の読み出しと展開。既存の PLE のベクトルと実際の表の行 |

`src/kernels/native_expert_parity.cpp` のネイティブのエキスパートの参照は、ggml の展開とホストの FP64 の内積を使ってから、CPU と GPU の経路を比べます。
エキスパートごとの相対 L1 の基準 `3e-2` は、この比較のためのもので、カーネル全般の許容誤差ではありません。
GPU の q8_1 の活性化の約束と、CPU の ggml の `vec_dot_type` は、明示したまま保ちます。

## 検証の一覧

| テストと資産 | 状態と限界 |
| --- | --- |
| `strata-device --selftest`、`elementwise_parity` | 移植済み。B70 で通る。デバイスの自己診断は確保、境界、容量を確かめ、数値の読み戻しは確かめない |
| `gguf_reader_test`、`suffix_drafter_test`、`controller_test`、`draft_policy_test`、`conv_cache_test` | 今のツールチェーンで通る。エンジン全体の結合を示すものではない |
| `s_gemv_parity`、`s_gemv_q8k_parity` | 移植済み。CTest に登録。B70 で通る。`--bench` は `s_gemv_split`、`s2_gemv_quads`、`s2_gemv_fast` も素朴なカーネルと比べる |
| `dequant_s2_parity`、`s2_gemv_parity`、`s2_gemv_q8_parity` | 移植済み。CTest に登録。B70 で通る |
| `rope_parity`、`router_top10_parity`、`bf16_gemv_parity`、`quantize_act_parity`、`qsa_parity`、`kv_q8_parity`、`kv_q4_parity`、`kv_stream_parity`、`gdn_parity`、`gr_parity`、`cvec_parity` | 移植済み。CTest に登録。B70 で通る |
| `shared_expert_parity` | 移植済み。CTest に登録。B70 で通る（ネイティブのスカラーのゲートの SYCL graph での再生を含む） |
| `sampler_parity` | 移植済み。CTest に登録。B70 で通る |
| `qsa_prompt_attn_parity` | 既存の FP64 の参照と FP32 の基準を SYCL に移植（ベンチマークも兼ねるので、ビルドするが CTest には登録しない）。XMX のカーネルは int8 と FP16 の KV で通る（[記録](../bench/results/2026-10-02-prompt-attn-xmx/README.md)）。Tensor Core の版は INT8・FP16・K8V4 で通り、Q4_0 は飛ばす。行列エンジンがなければ、呼び出し側は `qsa_decode_attn_batch` を使い続ける |
| `native_moe`、`native_gdn`、`native_gdn_preprocess`、`native_ple_postops`、`native_qsa`、`native_qsa_score`、`native_qsa_indexer`、`native_flash_attn`、`qsa_select`、`fused_gdn`（ツリーに parity なし） | [固定したコミットの ggml-cpu に対するプローブ](../bench/results/2026-09-30-xe-native/README.md)が B70 で通る。CTest には登録しない |
| `native_expert_parity`、`dequant_bf16_test` | 移植済み。`STRATA_NATIVE_EXPERTS` でビルド。モデルのシャードが要るので登録しない。[B70 で確かめたモデル](#b70-で確かめたモデル)のすべてのシャードで通る |
| `iq_parity` | 移植済みでビルドする。登録しない。`tools/iq_fixture.py` が、固定したモデルのリビジョンの範囲読みから `logs/iq_fixture` を書く。10 の形式すべてが通る。Q6_K と Q8_0 は一覧にない |
| `ple_parity` | `ple_oracle_vectors.inc` をリポジトリに持つ。全体の確認には実際の表と参照の資産も要る。`unverified` |
| CPU のエキスパートとプールのテスト、`pool_stress` | この一覧には含めていない。正準形のエキスパートのテストには AVX-512 か pack、またはその両方が要る。AVX-512 の経路は AGENTS.md のとおり Intel SDE で動かす |
| `serve.test_server`、`serve.test_detok`、`serve.test_mcp` | 模擬のエンジンで通る |
| `tools/test_iq_pack.py`、`tools/test_shards.py`、`tools/test_calibrate.py` | `python -m unittest` で通る（2026-10-04） |
| `src/ngram/ple_reader_test.cpp`、`src/platform/memory_test.cpp` | ビルドし直した。`ple_reader_selftest` は通り、`platform_memory_test` は memlock の上限 8 MiB では失敗する。どちらも登録したアリーナの証拠にはならない |

## Xe で測り直していない upstream の設定

次の設定は、upstream の Strata が CUDA（多くは Ryzen 5 7600 と RTX 5070）で決めたものです。
XeStrata はそのまま使っていて、Xe での調整は一部しか済んでいません。

| 場所 | 設定 | Xe での状態 |
| --- | --- | --- |
| `src/program/generate.cpp` | ネイティブの pack で、キャッシュにないエキスパートを PCIe で運ぶ割合 0.55（正準形は 0.2）。リンクの速さ（256 MiB の転送 4 回の最良）/ 20 GB/s で縮める | 新しい開発機で測ると、0.15 で 2〜4% 速く、それより大きい割合はどれも遅かった。0.55 はこの測定に支持されない（[記録](../bench/results/2026-10-02-new-machine/README.md#the-expert-arena-and-the-pcie-share)） |
| `tools/calibrate.py` | 基準は Ryzen 5 7600 と RTX 5070。PCIe の割合の候補 0、0.2、0.35、0.55、0.75、推測の下限 0.3、0.5、0.7、ワーカー数の候補。交互の確認で 3% を超えて速いときだけ採用 | 候補と基準は upstream のまま |
| `setup.py` | `--spec 4`、`--spec-min-p 0.5`、`--expert-cache auto`、`--prefill auto` | `--prefill auto` は 32768 トークンの塊まで選ぶように変えた（[記録](../bench/results/2026-10-03-prompt-upstream/README.md)）。既定のリングで収まらない塊は、リングを 96 スロットにして選ぶ（upstream #583。B70 を 8 GB に絞った IQ3_S で、8K と 32K のプロンプトが 1.5〜2 倍）。ほかは upstream のまま |
| `src/program/generate.cpp` | VRAM の予約 700 MiB（小さい GPU では 300 まで）、プロンプトの経路のスロットの借り方と塊の選び方、4 ラウンドごとのキャッシュの入れ替え（最大 96 個） | upstream のまま |
| `src/program/generate.cpp` と CPU のプール | 物理コア数に基づくワーカー数とホストのワーカー、キャッシュのヒットの通知、グラフとドアベルの段取り | ワーカー数は新しい開発機で測り、7〜19 で速さが変わらなかったので既定のまま（[記録](../bench/results/2026-10-02-new-machine/README.md#cpu-worker-threads)） |
| `src/kernels/cpu/iq_avx512.cpp` | ソフトウェアのプリフェッチの距離と、AVX-512 向けの調整 | 開発機は AVX-512 を持たないので測っていない |
