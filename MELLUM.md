# Mellum 2.1 on ds4

Branch `mellum` ports JetBrains' Mellum 2.1 (12B-A2.5B, thinking) onto
antirez/ds4, starting at `5379c1c`, in small commits. This file records what
works, what was measured (each number with its command and n), what diverges
from the model's training, and what is not covered.

## Status

Nothing of Mellum is ported yet. This commit records the baseline.

## Backends covered

Metal only, on Apple Silicon. CUDA, ROCm, the CPU reference path and
tensor-parallel/distributed inference are not built or tested for Mellum here,
and Mellum is not expected to run on them until someone ports and checks it.

## Baseline at 5379c1c (2026-10-26)

Machine: Apple M5 Max, 128 GiB (`sysctl -n hw.memsize machdep.cpu.brand_string`),
macOS SDK 27.0 (`xcrun --show-sdk-version`).

`make -j8` builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval` and `ds4-agent`
with no warnings (10.9 s wall). The model-free checks, each run once (n = 1),
all exit 0:

| command | result |
|---|---|
| `./ds4-eval --validate-cases` | 92 core and 50 hard cases validated |
| `./ds4-eval --self-test-extractors` | passed |
| `./ds4_agent_test` | ok |
| `./ds4_test --server` | ok |
| `./ds4_test --metal-kernels` | ok |
| `./tests/test_layer_pack` | 97/97 |
| `./tests/test_engine_mgpu_placement` | 114/114 |
| `./tests/test_gpu_args` | passed |
| `./tests/test_gpu_args_cli.sh` | 90 passed, 0 failed |
| `./tests/test_prompt_prefix` | passed |
| `./tests/test_sampling` | OK |
| `./tests/test_deepseek4_vision_image` | exit 0 |
| `./tests/test_image_decode` | exit 0 |

The rest of `make test` (`./ds4_test` with no arguments) needs a DeepSeek V4
Flash model and was not run.
