<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="../../app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">IA poderosa no seu próprio computador.</h3>
<p align="center">Converse, programe e trabalhe com imagens, localmente.</p>

<p align="center">
  <a href="#modelos">Modelos</a> · <a href="#instalação"><b>Instalação</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#ferramentas">Ferramentas</a> ·
  <a href="#ajuda">Ajuda</a> · <a href="#apoie-o-strata">Apoiar</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Baixar o Strata"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licença MIT"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Apoie o Strata no GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Apoie o Strata no Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="../../README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <b>Português</b> · <a href="README.pl.md">Polski</a></sub></p>

**Gratuito, de código aberto e seu para executar.** Use o Strata no desktop ou com suas ferramentas de programação favoritas. Não precisa de conta.

## Modelos

Recomendamos **primeiro a ISTA-DASLab**, depois a **Unsloth** e outros provedores, quando um modelo compatível cabe na memória.

<details>
<summary><img src="../media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provedor | Opções |
| --- | --- |
| **ISTA-DASLab · recomendado** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](../MODELS.md#coder) para programar. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Usa menos memória, mas é mais lento. |
| **Outras opções** | [Swift 1.5](../MODELS.md#swift-15), [NVFP4](../NVFP4.md), [OrcaRouter](../ORCA.md) |

[Compare tamanhos, necessidades de memória e compatibilidade](../MODELS.md).

</details>

<details>
<summary><img src="../media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Esses são os pacotes documentados atualmente para este modelo.

[Guia do modelo e compatibilidade](../QWEN36.md).

</details>

<a id="the-other-sizes"></a>

Não sabe o que cabe? [Escolha um modelo para o seu computador](../MODELS.md#pick-by-memory).

## Instalação

<a id="option-a-the-strata-app"></a>

### Aplicativo de desktop

**[Baixar o Strata para Windows, Linux ou Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Instale o aplicativo, escolha um modelo e comece a conversar. Há um instalador por sistema. Na primeira vez que abre, o aplicativo baixa o motor da sua placa de vídeo. [Ajuda com o instalador](../INSTALL.md#the-app).

<sub>Arquivos: `Strata-0.2.16-windows-x64-setup.exe` · `Strata-0.2.16-linux-x86_64.AppImage` · `Strata-0.2.16-macos-arm64.dmg`</sub>

<p align="center">
  <img src="../media/app-chat.png" width="900" alt="Aplicativo de desktop do Strata com um chat e uma prévia do código ao vivo lado a lado">
  <br><sub>Chat, arquivos e prévia ao vivo em uma só janela.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Aplicativo de Terminal

Prefere o terminal? Instale o Strata e execute **`strata launch`**.

<details open>
<summary><b>Comandos de instalação</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

No Windows, abra um novo terminal depois. Em seguida:

```sh
strata launch
```

Siga as instruções para baixar e iniciar um modelo. [Guia do Terminal](../TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="../media/terminal-monitor.png" width="900" alt="Aplicativo de Terminal do Strata mostrando a atividade do modelo, a velocidade, a memória e as solicitações recentes">
  <br><sub>Acompanhe seu modelo e seu hardware enquanto ele roda.</sub>
</p>

<a id="build-from-source"></a>

[Compilar a partir do código-fonte](../SOURCE_INSTALL.md) · [Docker](../DOCKER.md) · [Guia de instalação completo](../INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Hardware compatível | Guia |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [Instalação para NVIDIA](../INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Instalação para AMD](../AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [Instalação para RX 6000](../AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Instalação para Intel](../INTEL_ARC.md) |
| **Apple Silicon · M1 ou mais novo** | [Instalação para Mac](../MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Instalação para Strix Halo](../STRIX_HALO.md) |

<a id="using-it"></a>

## Ferramentas

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, e outras, pelas APIs compatíveis com OpenAI e Anthropic.

Execute **`strata connect`** para ver as instruções da sua ferramenta. [Guia de conexão](../TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Ajuda

[Todos os guias](..) · [Solução de problemas](../TROUBLESHOOTING.md) · [Relatar um bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](../COMMUNITY_BENCHMARKS.md) · [Como funciona](../HOW_IT_WORKS.md) · [Contribuir](../../.github/CONTRIBUTING.md)

## Apoie o Strata

**Ajude a manter a IA local aberta para todos.** Um patrocínio no GitHub ou um café ajuda a financiar o desenvolvimento e os testes em mais hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: apoie o Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: apoie o Strata" height="48"></a>

### Apoiadores fiéis

Obrigado a todas as pessoas que apoiam o Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

[Licença MIT](../../LICENSE) · [Créditos e licenças dos modelos](../HOW_IT_WORKS.md#credits) · [Privacidade](../SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Para ferramentas de IA e leitores automáticos: <a href="../../AGENTS.md">referência técnica</a> · <a href="../AI_SETUP.md">instruções de instalação</a>.</sub>

**Patrocínios, parcerias de hardware ou colaborações:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
