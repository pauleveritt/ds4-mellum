# Mellum 2.1 on DwarfStar (ds4-mellum)

This branch of a fork of antirez's [DwarfStar](https://github.com/antirez/ds4) (ds4) runs JetBrains' Mellum 2.1
(12B parameters, 2.5B active per token, a thinking coding model) on a Mac and serves it to a coding agent such as
[Pi](https://pi.dev). Mellum mixes 64 routed experts (8 per token) with sliding-window and full attention layers,
3 to 1; this branch adds Metal kernels for that shape, and can stream experts from the SSD so the model runs on an
8 GB Mac.

| Mac | Setup | Prefill, 2K prompt | Generation |
|---|---|---|---|
| M5 Max, 128 GB | Q8_0, resident | 4,026 t/s | 123 t/s |
| M5 Max, 128 GB | MXFP4 experts, resident | 4,170 t/s | 139 t/s |
| M1 Pro, 32 GB | Q8_0, resident | 370 t/s | 38 t/s |
| iMac M1, 8 GB | MXFP4 experts, streamed from the SSD, 2 GiB cache | not settled | 5.9 t/s |

*Note: speed inside the engine, not server → Pi; 2K context; no thermal load; one simple request. Thus, a
ceiling.*

In a real Pi session the M5's generation falls as the context grows, to about 109 t/s (Q8_0) by 8K: see
[Real use in Pi](#real-use-in-pi).

Details, run counts and ranges are under [Pick your Mac](#pick-your-mac).

**Before you start**

- **Mac only.** Metal on Apple Silicon. CUDA, ROCm, the CPU path and distributed inference are not built or
  tested for Mellum.
- **Two quants.** JetBrains' Q8_0 file and its `MXFP4_MOE` file (routed experts in MXFP4, the rest Q8_0). Every
  other quant is refused, including the Q4_K_M that JetBrains' model card recommends.
- **Three Macs measured.** An M5 Max (128 GB), an M1 Pro (32 GB) and an M1 iMac (8 GB). Part of the M5's prefill
  speed comes from Metal 4 tensor tiles that only M5-class chips get.

## Build and get the model

```bash
git clone https://github.com/pauleveritt/ds4-mellum
cd ds4-mellum
make
```

Run everything below from this directory: `ds4` and `ds4-server` load their Metal kernels from `metal/`.

Download the quant your Mac's section names (both from JetBrains'
[repository](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF)):

```bash
hf download JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF \
  Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf --local-dir ~/models
hf download JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF \
  Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf --local-dir ~/models
```

| File | Bytes | sha256 |
|---|---|---|
| `Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` | 12,925,417,760 | `69e0ac02a16586c2c9e991f1e3b8a1eb403882e7ab9e69ea01eb23dbbbc6841f` |
| `Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf` | 7,029,537,056 | `cf09584a30423f6f80b0c9fe2f42e152520dabf08fecd1cad641ce21a7588166` |

Q8_0 is closer to the original weights; MXFP4 is smaller, and on the M5 decodes faster. JetBrains' own figures against BF16:
KL divergence 0.008 for Q8_0 and 0.115 for MXFP4.

Quick check (the last line on stderr gives the prefill and generation rates):

```bash
./ds4 -m ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf -c 32768 \
  -p "Write a Python function that checks whether a string is a palindrome."
```

Mellum thinks first: its reasoning prints as plain text, then the answer.

## Serve it and connect Pi

```bash
./ds4-server -m ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf -c 32768 --port 8099
```

Use the server command from your Mac's section. The server speaks the OpenAI chat-completions API, with tools and
streaming, and returns Mellum's reasoning separately as `reasoning_content`; `curl
http://127.0.0.1:8099/v1/models` lists `mellum-2.1`.

Mellum was trained on Pi's four tools (`read`, `write`, `edit`, `bash`). Add a `ds4` provider to
`~/.pi/agent/models.json` (merge it into `"providers"` if the file exists):

```json
{
  "providers": {
    "ds4": {
      "baseUrl": "http://127.0.0.1:8099/v1",
      "api": "openai-completions",
      "apiKey": "ds4",
      "models": [
        {
          "id": "mellum-2.1",
          "contextWindow": 32768,
          "samplingParams": { "temperature": 0.6, "top_p": 0.95, "top_k": 20 }
        }
      ]
    }
  }
}
```

Then, with `ds4-server` running: `pi --model ds4/mellum-2.1`. The key is a placeholder.

**Settings for every Mac.**
- Sampling: temperature 0.6, top-p 0.95, top-k 20 (JetBrains' model card), set in Pi's `samplingParams`. A
  client that sends none gets temperature 1.0 and no top-k, because Mellum's `generation_config.json` sets none.
- Thinking: leave it on. With `--nothink`, Mellum still reasons in plain text inside its coding answers.
- Context: keep Pi's `contextWindow` equal to the server's `-c`. Mellum is trained to 131,072 tokens; 32,768 is
  the size measured here.

## Or skip the server: ds4-agent

`ds4-agent` is DwarfStar's own coding agent. It runs the model in the same process, with no HTTP hop, and on
this branch it drives Mellum with Pi's four tools:

```bash
./ds4-agent -m ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf -c 32768 --temp 0.6 --top-p 0.95
```

It is the other path to speedups, because the agent and the engine change together. Today the next user turn
reuses the live KV cache instead of prefilling the conversation again, and one tool result is capped at a quarter
of the context. The agent also repairs Mellum's bare quotes in `write` and `edit` arguments. It has not been
measured against Pi. It has no top-k option.

## Pick your Mac

Three settings matter: the **quant** (Q8_0 or MXFP4), **SSD streaming** (off, or on with an expert-cache size)
and the **context** (`-c`). Every speed below comes from the runners in [Measure on your own Mac](#measure-on-your-own-mac)
unless it says otherwise: 256 greedy tokens after real prose, n = 6 processes per cell, median and range.

### M5 Max, more than 32 GB

| Setting | Use |
|---|---|
| Quant | Q8_0, or MXFP4 for faster decode |
| SSD streaming | off |
| Context | `-c 32768` |

```bash
./ds4-server -m ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf -c 32768 --port 8099
```

Memory at `-c 32768` (Metal peak, 2,686-token request): Q8_0 12.78 GiB, MXFP4 7.29 GiB (n = 3 each). Apple
M5 Max, 128 GB, AC power, commit `67b4670`
([captures](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-29-mellum-readme-speed/m5)):

| Model file | Prompt tokens | Prefill t/s | Generation t/s |
|---|---|---|---|
| Q8_0 | 2,056 | 4,026 (4,000-4,033) | 123.3 (113.9-129.8) |
| Q8_0 | 8,195 | 3,474 (3,468-3,478) | 124.5 (124.0-125.0) |
| Q8_0 | 32,778 | 1,826 (1,610-1,929) | 91.3 (89.6-102.2) |
| MXFP4 experts | 2,056 | 4,170 (4,158-4,173) | 138.5 (117.4-154.9) |
| MXFP4 experts | 8,195 | 3,577 (3,572-3,580) | 147.5 (145.9-147.9) |
| MXFP4 experts | 32,778 | 1,920 (1,877-2,015) | 98.6 (92.0-113.6) |

#### Real use in Pi

Plain Pi 1.1.0 driving `ds4-server -c 32768` through one AgentClinic task ("Implement Phase 1 — Home Page."),
3 runs per quant, same Mac, High Power. Medians per request, grouped by the context the request started with
([note](docs/superpowers/research/2026-10-31-mellum-pi-real-use.md)):

| Model file | Context at start | Prefill t/s (n) | Generation t/s (n) |
|---|---|---|---|
| Q8_0 | 0-4K | 3,105 (7) | 124.4 (25) |
| Q8_0 | 4-8K | 2,558 (1) | 114.1 (23) |
| Q8_0 | 8-16K | 2,395 (3) | 109.3 (7) |
| MXFP4 experts | 0-4K | 4,008 (4) | 147.1 (22) |
| MXFP4 experts | 4-8K | 2,436 (2) | 128.7 (6) |

Prefill medians count only requests adding 256 tokens or more. Most turns add fewer (66 of 77 after the first),
often 18-44 tokens of tool output, at 280-830 t/s; each still takes only 0.05-0.12 s. Q8_0 finished the task in 2 of 3 runs (about a minute each),
MXFP4 in 0 of 3. Two runs stalled when Mellum started the app's server in the foreground, which plain Pi waits on
forever; one MXFP4 run ended on an unterminated tool call, and one on a file read that overflowed the 32K context.

### M1 Pro, 32 GB

| Setting | Use |
|---|---|
| Quant | Q8_0 (MXFP4 is no faster here) |
| SSD streaming | off |
| Context | `-c 32768` |

```bash
./ds4-server -m ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf -c 32768 --port 8099
```

Memory planned at a 33,792 context: Q8_0 12.82 GiB, MXFP4 7.33 GiB (the engine's startup line). Apple M1 Pro,
32 GB, AC power, commit `a699e14`, other apps open
([captures](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-29-mellum-readme-speed/m1pro)):

| Model file | Prompt tokens | Prefill t/s | Generation t/s |
|---|---|---|---|
| Q8_0 | 2,056 | 370 (368-371) | 37.7 (37.7-37.9) |
| Q8_0 | 8,195 | 328 (328-329) | 35.5 (35.5-35.5) |
| Q8_0 | 32,778 | 226 (224-230) | 28.9 (28.9-29.0) |
| MXFP4 experts | 2,056 | 354 (354-354) | 34.9 (34.7-35.6) |
| MXFP4 experts | 8,195 | 314 (313-315) | 33.0 (32.9-33.4) |
| MXFP4 experts | 32,778 | 219 (218-220) | 27.1 (27.1-27.1) |

On this chip MXFP4 is slightly slower than Q8_0 in both prefill and generation, the reverse of the M5; pick it
only to save memory.

### iMac M1, 8 GB

| Setting | Use |
|---|---|
| Quant | MXFP4 (half the bytes per expert read; Q8_0 was not run here) |
| SSD streaming | `--ssd-streaming --ssd-streaming-cache-experts 2GB` |
| Context | `-c 32768` |

```bash
./ds4-server -m ~/models/Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf -c 32768 --port 8099 \
  --ssd-streaming --ssd-streaming-cache-experts 2GB
```

Quit other apps first. Measured with apps closed, on a 2,686-token coding-agent request through the measure tool
(not these runners), n = 3: decode 5.85 t/s, first token 19.4 s, Metal peak 3.86 GiB, no swap growth; the
greedy output matches the M5's resident logits (29/29 and 64/64 top tokens)
([result](docs/superpowers/research/2026-10-28-mellum-s6-c-imac.md)). A 2.5 GiB cache decodes about 25% faster
(n = 1) but goes 69 MB over the memory limit, so 2 GiB is the setting.

Keep `-c 32768`. At a short context, streaming locks the shared weights in memory, and on this iMac prefill then
varied from about 3 to 120 t/s between runs; if you need a short context, set
`DS4_METAL_DISABLE_STREAMING_STATIC_LOCK=1` ([note](docs/superpowers/research/2026-10-29-mellum-imac-static-lock.md)).
Prefill rows are not published until that is settled.

### Measure on your own Mac

The runners measure one build on real prose (the start of `speed-bench/promessi_sposi.txt` plus a request), in
two blocks of four processes per prompt length, with cooldowns; each block's first process is dropped (n = 6):

```bash
ARMS=AA tests/mellum_cli_prefill_abba.sh ~/ds4-runs/prefill ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf 2048
ARMS=AA tests/mellum_cli_decode_abba.sh ~/ds4-runs/decode ~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf tests/mellum_prompts 2048
```

Add `8192 32768` after `2048` for longer prompts. For a streamed row, set
`DS4_ARGS="--ssd-streaming --ssd-streaming-cache-experts 2GB"`. Each output directory gets `summary.txt`,
`runs.txt`, every process's log, and `events.txt` (power source and thermal state per block). Run on AC power
with nothing else on the GPU.

## Speed work

Each area lists what worked (measured), what did not, and what is next. Figures are from the M5 Max unless noted.

Read the prefill peak (4,055 t/s) as a ceiling: it was measured inside the engine, not over the server to Pi, on
a 2K-token prompt, with no thermal load and one simple request.

### Prefill

Prefill is compute-bound, so the work went into batching rows and tiling.

- **Worked**
  - Experts as tiled GEMMs and Q8_0 projections as batched GEMMs, for chunks of 32 rows or more (every layer):
    about 350 to 2,000 t/s at 2,200 tokens.
  - Routed Q8_0 experts on the M5's Metal 4 tensor tiles (expert layers): +40% at 2,200 tokens (n = 10 per arm).
  - Tiled attention, 8 tokens per threadgroup sharing staged keys (sliding and full layers): 2,968 to 4,055 t/s at
    2K, 2,132 to 3,528 at 8K, 939 to 1,908 at 32K (n = 6).
  - Streamed: grouped, layer-major prefill cut a 30K prompt's cold first token from 60.6 to 37.7 s (n = 3).
- **Didn't work**
  - Tiled attention with f32 operands: about twice the attention time of the landed half-input kernel at 8K
    (n = 1).
- **Next**
  - Long contexts: at 32K full attention still dominates; more tiling of full attention.
  - Pre-M5 chips: measure what the simdgroup path leaves on the table without tensor tiles.
  - Streamed: grouped prefill loads 3.5% more experts than predicted; check eviction inside a group.

### Decode

Resident decode is GPU-bound at about 53-60% of measured memory bandwidth, so each kernel idea has a small
ceiling.

- **Worked**
  - Sliding-window decode attention loading two keys per step: 1.035x at 2K (n = 6).
  - Four fused small kernels (norm with RoPE, K/V copies, MoE reduce with residual, add with norm): about 1.03x
    in the screen (n = 6).
  - MXFP4 experts: 1.13-1.20x Q8_0's decode, at a quality cost.
- **Didn't work**
  - Computing K and V in the Q projection's dispatch: 1.008x. Removing dispatches alone buys little.
- **Next**
  - Sliding attention with more partitions per head (ceiling 1.14x), full attention at long context (1.12x at
    32K), GPU idle between steps (1.08x).
  - A cheaper two-row verify, which prompt lookup and an MTP head both need. Outside reports find speculation
    rarely helps batch-1 decode on Apple GPUs, so measure first.

### SSD streaming

With experts streamed, decode is limited by how often a token stops to read experts, not by SSD bandwidth.

- **Worked**
  - Gated stops (dispatches after a stop are suppressed, not re-run): 32K decode 26.2 to 39.0 t/s, Q8_0 (n = 3).
  - Cheaper stops (reads run while the resumed pass is encoded): MXFP4 at 2.5 GiB, 22 to 40 t/s (n = 3).
  - MXFP4 experts halve every read, which is what makes 8 GB possible.
- **Didn't work**
  - Chunked reads and windowed passes (screens).
  - Synchronous next-layer prefetch: avoided 3.7 stops per token but ran 1.3 ms/token slower (n = 2).
  - A 2.5 GiB cache on 8 GB: 69 MB over the memory limit.
- **Next**
  - Free about 69 MB so 2.5 GiB fits on 8 GB (+25% decode, n = 1).
  - Fewer stops per token through hotness and eviction policy; stops rise from 8.7 to 12.7 per token after a
    tool result is appended.
  - Cheaper stops: on the iMac each costs about 1 ms fixed plus 2 ms per re-run layer.
  - Asynchronous prefetch: pre-gating predicts 7.0 of the next layer's 8 experts.
  - Decide when to lock shared weights on small machines.

### General next steps

- Measure real use: Python coding sessions in Pi, with tool-call success and time per turn.
- Offer the patches that make sense outside Mellum upstream.
- Measure a 16 GB Mac.
- A CUDA version for Nvidia GPUs.
- Keep shrinking the fork: about 19,800 added lines of C, Objective-C and Metal across 44 files today, tests included
  (`git diff --shortstat c97964b mellum -- '*.c' '*.h' '*.m' '*.metal' '*.inc' Makefile`). Mellum's code in `ds4.c`
  and `tests/ds4_test.c` now sits in `ds4_mellum_*.inc` and `tests/ds4_test_mellum*.inc`, leaving about 700 lines of
  hooks in `ds4.c`; `ds4_agent.c` and `ds4_server.c` (the Hermes syntax, Pi's tools) are next.
- Find people to contribute code or run measurements on their Macs.
- Make this part of Satyrn AI.

## More

- [`MELLUM.md`](MELLUM.md): the port's record. Every change, every measurement with its command and run count,
  known divergences from the model's training, and what is not covered.
- [`ROADMAP.md`](ROADMAP.md): the SSD streaming work, phase by phase.
- [`tests/`](tests/): the runners, the oracle comparison against the earlier port, and the model-gated tests
  (`DS4_TEST_MELLUM_MODEL`).

---

# DwarfStar (upstream README)

Everything below is antirez's DwarfStar README as of the commit this branch is based on, unchanged. It describes
upstream's models and backends; for Mellum, the guide above and [`MELLUM.md`](MELLUM.md) apply.

<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 64 GB (for Qwen Flash next) and 96 GB or more (for DeepSeek and GLM). Smaller machines can use SSD streaming. SSD streaming is also needed in order to run very large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
