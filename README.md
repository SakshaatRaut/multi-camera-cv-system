# Multi-Camera GPU-Accelerated Computer Vision System

Real-time parallel video processing pipeline with object detection, automated benchmarking across seven experiments, and a multi-device leaderboard dashboard for comparing CUDA discrete GPUs against Apple Silicon and other accelerators.

## What this project does

The pipeline ingests video from multiple cameras simultaneously, runs YOLOv8 object detection on each stream, and reports per-stream and aggregate throughput. Implementation uses Python multiprocessing to dedicate one OS process per camera (avoiding the GIL on capture and decoding) and a shared GPU process for batched inference.

The benchmark suite measures seven experiments. The first four are the originally proposed comparisons; the last three were added to broaden the analysis:

| Experiment | What it measures |
|---|---|
| **E1** CPU vs GPU | Same workload run on CPU and GPU to isolate the device speedup. |
| **E2** Batch size sweep | Throughput vs inference batch size (1, 2, 4, 8, 16). |
| **E3** Multi-stream scaling | Throughput vs camera count (1, 2, 4, 8) with auto-fitted Amdahl's Law analysis. |
| **E4** Bottleneck identification | Per-stage latency decomposition with p50/p95/p99 percentiles. |
| **E5** Model-size sweep | YOLOv8 variants (n/s/m/l/x) — accuracy vs throughput Pareto. |
| **E6** Inference engine comparison | PyTorch FP32, PyTorch FP16, ONNX Runtime side by side. |
| **E7** GPU preprocessing | CPU preprocessing vs GPU-side resize/normalize, with optional FP16 + CUDA streams. |

A coordinator + agent system collects results from heterogeneous hardware (CUDA, Apple MPS, CPU) into one shared dashboard. See `web/README.md` for the dashboard architecture.

## Quick start

### Prerequisites

- Python 3.10+
- One of: NVIDIA GPU + CUDA 12.1+, Apple Silicon Mac, or any x86-64 CPU
- Eight `.mp4` video files in `videos/cam1.mp4` … `videos/cam8.mp4` for the canonical benchmark

### Install

```bash
git clone https://github.com/<your-username>/multi-camera-cv-system.git
cd multi-camera-cv-system

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

For best ONNX performance on CUDA boxes, replace the universal `onnxruntime` with the GPU build:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### Smoke test (30 seconds)

```bash
python main.py --video-sources videos/cam1.mp4 videos/cam2.mp4 \
               --device auto --duration 30
```

You should see per-camera FPS lines streaming for 30 seconds, then a summary report. Outputs land in `results/`.

### Full benchmark (≈30 minutes)

```bash
python main.py --video-sources videos/cam1.mp4 videos/cam2.mp4 videos/cam3.mp4 \
                               videos/cam4.mp4 videos/cam5.mp4 videos/cam6.mp4 \
                               videos/cam7.mp4 videos/cam8.mp4 \
               --device auto --duration 30 --run-experiments all
```

This runs the baseline plus all seven experiments. Baseline outputs land in `results/baseline/`; experiment outputs in `experiments/results/`.

## Repository layout

```
multi-camera-cv-system/
├── main.py                            Entry point. --run-experiments triggers the full suite.
├── config/
│   ├── config.py                      Default SYSTEM_CONFIG and COCO class names.
│   └── __init__.py
├── camera/
│   └── camera_stream.py               Per-camera mp.Process producer; shared frame queue.
├── detection/
│   ├── detector.py                    PyTorch YOLODetector with FP16 + GPU preprocess + CUDA streams.
│   └── onnx_detector.py               ONNX Runtime drop-in detector.
├── pipeline/
│   └── pipeline_manager.py            Batched producer-consumer pipeline.
├── profiling/
│   ├── profiler.py                    System-level CPU/RAM/GPU monitoring.
│   └── detailed_profiler.py           Per-stage percentile profiler (E4).
├── analysis/
│   └── amdahls_law.py                 Parallel-fraction estimator + speedup-curve plot.
├── experiments/
│   └── run_all_experiments.py         E1–E7 automation; spawns main.py subprocesses.
├── visualization/
│   └── visualizer.py                  Matplotlib reports for the baseline run.
├── web/
│   ├── coordinator.py                 FastAPI + SQLite leaderboard server.
│   ├── agent.py                       Per-device benchmark runner + uploader.
│   ├── dashboard.html                 Single-page leaderboard UI (Chart.js).
│   ├── requirements.txt               FastAPI dependencies (separate from main).
│   └── README.md                      Dashboard architecture and deployment.
├── videos/                            Input video files (gitignored).
├── results/                           Per-run scratch outputs (gitignored).
├── experiments/results/               Full-suite outputs (gitignored).
├── requirements.txt
└── README.md                          This file.
```

## Command-line reference

`main.py` accepts the following flags. The benchmark agent (`web/agent.py`) wraps these with sensible defaults — most users do not invoke `main.py` directly.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cameras N` | int | — | Use webcam at index 0, replicated N times. |
| `--video-sources A B C …` | paths | — | Video files to use as camera streams. |
| `--webcam` | flag | — | Single-camera webcam mode. |
| `--duration` | int | None | Run length in seconds. None = run until Ctrl+C. |
| `--device` | choice | `cuda` | `auto` / `cuda` / `mps` / `cpu`. |
| `--model` | choice | `yolov8n.pt` | `yolov8{n,s,m,l,x}.pt`. |
| `--batch-size` | int | 4 | Inference batch size. |
| `--backend` | choice | `pytorch` | `pytorch` or `onnx`. |
| `--fp16` | flag | off | Enable FP16 mixed precision (CUDA only; no-op on MPS). |
| `--gpu-preprocess` | flag | off | Run resize/normalize on the GPU (CUDA or MPS). |
| `--onnx-coreml` | flag | off | Opt into CoreMLExecutionProvider (CPU EP is the default on Mac because CoreML is incompatible with YOLOv8 dynamic-batch graphs). |
| `--no-detailed-profiling` | flag | off | Disable per-stage profiler. |
| `--run-experiments` | choice | `none` | After the baseline, also run experiments. Values: `none` / `core` (E1–E4) / `all` (E1–E7) / `e1` … `e7`. |

Exactly one of `--cameras`, `--video-sources`, or `--webcam` must be specified.

## Running the benchmark

There are two ways to produce results:

### Option A: Local run via `main.py` (single device)

Useful for development or testing on one machine. Outputs go to local `results/` and `experiments/results/`.

```bash
# Baseline only (≈30 s)
python main.py --video-sources videos/cam1.mp4 videos/cam2.mp4 ... videos/cam8.mp4 \
               --device auto --duration 30

# Baseline + a single experiment
python main.py --video-sources videos/cam{1..8}.mp4 \
               --device auto --duration 30 --run-experiments e3

# Baseline + all seven experiments
python main.py --video-sources videos/cam{1..8}.mp4 \
               --device auto --duration 30 --run-experiments all
```

### Option B: Submission via the agent (multi-device)

Recommended for the actual benchmark. The agent runs `main.py --run-experiments all` with auto-detected hardware settings, then uploads the bundle to a coordinator running on one of the team's machines. Results from every participant accumulate on a shared dashboard.

```bash
# Start the coordinator (do this once on one machine)
uvicorn web.coordinator:app --host 0.0.0.0 --port 8000

# On every participating laptop:
python web/agent.py --server http://<coordinator-host>:8000 \
                    --device-label "My Laptop" --mode full
```

The agent auto-enables FP16 + GPU preprocessing on CUDA hardware to surface realistic optimized performance on the headline chart. On Apple Silicon those flags are no-ops in the current detector code (FP16 is CUDA-only), so the headline reflects honest MPS-vanilla performance.

See [`web/README.md`](web/README.md) for full coordinator deployment instructions, networking options (LAN, Tailscale, ngrok), and submission workflow.

## Output files

A single run of `main.py` produces (under `results/`):

```
results/
├── run_summary.json              Machine-readable headline numbers + config.
├── summary_report.txt            Human-readable performance summary.
├── bottleneck_analysis.json      Per-stage E4 data (machine-readable).
├── bottleneck_analysis.txt       Per-stage E4 data (human-readable).
├── stage_latency_histograms.png  Per-stage latency distributions (E4 visual).
├── fps_comparison.png            Throughput-over-time plot.
├── cpu_usage.png
├── memory_usage.png
├── system_overview.png
└── system_stats.csv              Raw psutil samples.
```

When `--run-experiments` is set, these files are first preserved into `results/baseline/` before the experiment subprocesses overwrite `results/` with their per-experiment outputs. Experiment outputs land in `experiments/results/` with the layout:

```
experiments/results/
├── experiment_metadata.json     Aggregate of all completed experiments.
├── e1_gpu_summary.json          E1 GPU run.
├── e1_cpu_summary.json
├── e2_batch_{1,2,4,8,16}_summary.json
├── e2_batch_size_curve.png
├── e3_streams_{1,2,4,8}_summary.json
├── e3_amdahls_law_report.txt    Auto-fitted P-fraction analysis.
├── e3_amdahls_law_plot.png
├── e4_summary.json
├── e5_yolov8{n,s,m,l,x}_summary.json
├── e5_model_sweep.png
├── e6_pytorch_{fp32,fp16}_summary.json
├── e6_onnxruntime_summary.json
├── e6_engines.png
├── e7_{cpu,gpu,gpu_fp16}_preproc_summary.json
└── e7_gpu_preproc.png
```

## Architecture

```
                       Each camera process reads
                       its video at native FPS;
                       puts frames into shared queue
                       ──────────────────────────────

  Camera 0 (mp.Process) ─┐
  Camera 1 (mp.Process) ─┤
  Camera 2 (mp.Process) ─┼──▶ mp.Queue ──▶ Pipeline manager (main process)
  Camera 3 (mp.Process) ─┤                       │
   …                     │                       ▼
  Camera 7 (mp.Process) ─┘                Batch formation
                                                 │
                                                 ▼
                                          GPU detector (CUDA / MPS / CPU)
                                                 │
                                                 ▼
                                       Per-stage profiler:
                                       frame_loading,
                                       preprocessing,
                                       inference,
                                       postprocessing,
                                       output

  Background:  system profiler (psutil) samples CPU/RAM at 1 Hz.
```

Five stages are instrumented per frame; the DetailedProfiler reports their p50/p95/p99 latencies and a percentage breakdown of where total pipeline time is spent.

## Hardware notes

**CUDA discrete GPUs** — The agent auto-enables `--fp16` and `--gpu-preprocess`. ONNX Runtime uses `CUDAExecutionProvider`. Best performance on this stack.

**Apple Silicon (MPS)** — The detector uses Metal Performance Shaders for inference. FP16 autocast is intentionally disabled on MPS in the detector code because PyTorch's MPS backend has incomplete FP16 op coverage and silently falls back to FP32 for many ops; forcing it on would mislabel FP32 numbers as FP16. ONNX Runtime uses `CPUExecutionProvider` because `CoreMLExecutionProvider` has a known incompatibility with YOLOv8 dynamic-batch graphs.

**CPU-only** — All paths fall back gracefully. The pipeline still works but throughput drops by 5–10× depending on the CPU.

## Troubleshooting

**`CUDA out of memory`** — Reduce `--batch-size` or use a smaller model variant (`--model yolov8n.pt`). On 4 GB GPUs the eight-camera default may exceed VRAM at batch sizes above 4.

**Pipeline shows 0 frames processed** — Either no `videos/cam*.mp4` files are present, or the configured device couldn't open them. Check that exactly eight valid `.mp4` files exist in `videos/`.

**`Field power.draw is not supported` from `nvidia-smi`** — Some older laptop GPUs disable power telemetry in vBIOS; the FPS-per-watt experiment isn't possible on those cards.

**ONNX scenario silently skipped in E6** — Usually means `onnxruntime` isn't installed. `pip install onnxruntime` (or `onnxruntime-gpu` on CUDA hosts) and rerun.

**Subprocess hangs at shutdown** — Has been observed when forced child termination corrupts the multiprocessing queue's pipe state on macOS. The pipeline manager mitigates this with a daemon-thread drain and bounded queue close. If you still see it, please open an issue.

## License

MIT. See LICENSE.
