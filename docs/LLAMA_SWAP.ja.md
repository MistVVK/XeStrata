<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# llama-swap の後ろで XeStrata を動かす

[llama-swap](https://github.com/mostlygeek/llama-swap) は、いくつものモデルのサーバーを 1 つのアドレスの後ろに置き、要求が名指ししたものを起動して、ほかを止めて GPU を空けます。
XeStrata は、そのモデルの 1 つとして動きます。
この文書は upstream（Strata 0.1.39、Tesla V100）で確かめた手順を、XeStrata の名前に合わせたものです。
XeStrata では試していません（`unverified`）。

## 設定

先にいつもどおり setup をします（`./setup.sh --no-start ...`）。
setup が書く起動スクリプト（`run-<model>.sh`）はポートを固定するので、llama-swap のポートで `serve/server.py` を直接呼びます。

```yaml
healthCheckTimeout: 300            # 読み込みに 1〜2 分かかる

models:
  xestrata:
    cmd: >-
      sh -c 'cd /path/to/XeStrata && exec .venv/bin/python -m serve.server
      --engine strata --config xestrata-<model>.json --port ${PORT}'
    env: ["STRATA_ALLOWED_HOSTS=my-server"]     # クライアントが使うホスト名（下を参照）
    checkEndpoint: /health
    ttl: 1800
```

- `exec` で、llama-swap が止めるプロセスを Python の server にします。llama-swap の停止（SIGTERM）で、server はエンジンも止めます。
- `checkEndpoint: /health`: XeStrata の `/health` は、モデルを読み込み終えてから答えます。
- ほかのモデルと GPU を分け合うなら、llama-swap のグループに入れるか、入れ替えで降ろさせます。XeStrata は GPU のメモリーを自分だけで使います。

## つまずくところ: 「Host ... is not allowed」

llama-swap を通した最初の要求が、すぐに次のように断られることがあります。

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection)
```

XeStrata は知っているホスト名にだけ答えます。ウェブのページが DNS の名前を 127.0.0.1 に向けて届くのを防ぐためです。
llama-swap はクライアントの Host ヘッダーをそのまま渡すので、名前でプロキシに来たクライアント（`http://my-server:8040`）は断られます。
IP アドレスと `localhost` への要求はいつも通るので、手元の試しでは動き、ほかの機械からだけ失敗することがあります。
次のどれかで直ります。

- エントリーの `env` に `STRATA_ALLOWED_HOSTS=my-server`（カンマ区切りの名前。ポートは見ない）
- `xestrata-<model>.json` に `"allowed_hosts": ["my-server"]`
- `"api_key"` を設定する（この確かめはなくなり、どのクライアントも鍵を送る）

## llama-swap を通した XeStrata のページ

llama-swap は `/upstream/<model>/...` をそのまま渡すので、状態と health のページにも届き、呼ぶとチャットの要求なしにモデルを読み込みます。

```bash
curl http://my-server:8040/upstream/xestrata/health
curl http://my-server:8040/upstream/xestrata/v1/status
```

## Claude Code など Anthropic の API のクライアント: 考える深さ

Claude Code（と Claude Agent SDK）は、指定しなければ毎回 `"output_config": {"effort": "high"}` を送るので、XeStrata はいちばん深く考えます。
クライアントの環境の `CLAUDE_CODE_EFFORT_LEVEL=medium`（または `low`）で変わります。
