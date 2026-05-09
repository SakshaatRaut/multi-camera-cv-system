# Multi-Camera GPU-Accelerated Computer Vision System

> Real-time parallel video processing pipeline with YOLOv8 object detection, benchmarked across NVIDIA CUDA, Apple Silicon MPS, and CPU hardware via a shared leaderboard dashboard.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Benchmark](#running-the-benchmark)
- [Experiment Suite (E1–E7)](#experiment-suite-e1e7)
- [Multi-Device Leaderboard](#multi-device-leaderboard)
- [Command Reference](#command-reference)
- [Output Files](#output-files)
- [Repository Layout](#repository-layout)
- [Architecture](#architecture)
- [Hardware Notes](#hardware-notes)
- [Troubleshooting](#troubleshooting)

---

## What This Project Does

The pipeline ingests video from **up to 8 simultaneous camera streams**, runs **YOLOv8 object detection** on every frame, and reports per-stream and aggregate throughput.

**How it works under the hood:**

- One dedicated OS process per camera (bypasses Python's GIL for true parallel capture)
- Frames are batched and sent to a single GPU inference process (CUDA / Apple MPS / CPU)
- Five pipeline stages are instrumented per frame for microsecond-resolution profiling
- A background daemon samples CPU, RAM, and GPU metrics at 1 Hz

**Benchmark suite — 7 experiments:**

| # | Experiment | What It Measures |
|---|-----------|-----------------|
| E1 | CPU vs. GPU | Device speedup factor on the same workload |
| E2 | Batch size sweep | Throughput vs. batch size (1, 2, 4, 8, 16) |
| E3 | Multi-stream scaling | FPS vs. camera count (1, 2, 4, 8) + Amdahl's Law fit |
| E4 | Bottleneck identification | Per-stage p50 / p95 / p99 latency breakdown |
| E5 | Model size sweep | YOLOv8 n / s / m — accuracy vs. throughput |
| E6 | Inference engine comparison | PyTorch FP32 vs. FP16 vs. ONNX Runtime |
| E7 | GPU preprocessing | CPU-side vs. GPU-side resize/normalize |

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/multi-camera-cv-system.git
cd multi-camera-cv-system

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a 30-second smoke test (2 streams)
python main.py --video-sources videos/cam1.mp4 videos/cam2.mp4 \
               --device auto --duration 30
```

You should see per-camera FPS lines streaming for 30 seconds, then a summary report. Results land in `results/`.

---

## Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Required |
| NVIDIA GPU + CUDA 12.1+ | For CUDA acceleration |
| Apple Silicon Mac | For MPS acceleration |
| Any x86-64 CPU | CPU-only fallback (5–10× slower) |
| 8 × `.mp4` video files | Place in `videos/cam1.mp4` … `cam8.mp4` |

### Standard Install

```bash
pip install -r requirements.txt
```

### Optional: CUDA-Optimized PyTorch

For best performance on NVIDIA GPUs, replace the default PyTorch wheel with the CUDA build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Optional: GPU-Accelerated ONNX Runtime

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu          # NVIDIA CUDA
# pip install onnxruntime-silicon    # Apple Silicon
```

---

## Running the Benchmark

### Option A — Local run (single device)

```bash
# Baseline only (~30 seconds)
python main.py \
  --video-sources videos/cam1.mp4 videos/cam2.mp4 videos/cam3.mp4 \
                  videos/cam4.mp4 videos/cam5.mp4 videos/cam6.mp4 \
                  videos/cam7.mp4 videos/cam8.mp4 \
  --device auto --duration 30

# Baseline + single experiment
python main.py --video-sources videos/cam{1..8}.mp4 \
               --device auto --duration 30 --run-experiments e3

# Baseline + full experiment suite (~20–40 minutes)
python main.py --video-sources videos/cam{1..8}.mp4 \
               --device auto --duration 30 --run-experiments all
```

### Option B — Agent submission (multi-device leaderboard)

Run this on **every participating device**. The agent auto-detects your hardware, runs the full benchmark, and uploads results to the coordinator.

```bash
# Step 1: Start the coordinator on one shared machine (do this once)
uvicorn web.coordinator:app --host 0.0.0.0 --port 8000

# Step 2: On every participating device, run the agent
python web/agent.py \
  --server http://<coordinator-host>:8000 \
  --device-label "My Laptop" \
  --mode full
```

Open `http://<coordinator-host>:8000` to view the live leaderboard.

> See [`web/README.md`](web/README.md) for full coordinator deployment options including LAN, Tailscale, and ngrok.

---

## Experiment Suite (E1–E7)

Each experiment is run as a subprocess of `main.py` with a single parameter changed. Results are saved in `experiments/results/`.

| Flag | Options | Default |
|------|---------|---------|
| `--run-experiments` | `none` / `core` (E1–E4) / `all` (E1–E7) / `e1`…`e7` | `none` |

**Examples:**

```bash
# Run only E3 (multi-stream scaling)
python main.py --video-sources videos/cam{1..8}.mp4 --device auto --run-experiments e3

# Run E1–E4 (core experiments only)
python main.py --video-sources videos/cam{1..8}.mp4 --device auto --run-experiments core
```

When `--run-experiments` is used, the baseline artifacts are automatically preserved to `results/baseline/` before experiment runs overwrite `results/`.

---

## Multi-Device Leaderboard

The coordinator + agent system collects results from any number of devices and displays them on a single dashboard with cross-device charts.

```
┌─────────────────────────────────────────────┐
│  Coordinator (one machine, always on)        │
│  uvicorn web.coordinator:app --port 8000     │
│  → Stores results in web/data/runs.db        │
│  → Serves dashboard at /                     │
└─────────────┬───────────────────────────────┘
              │ HTTP POST /api/submissions
   ┌──────────┼──────────┬────────────┐
   │          │          │            │
 Device A   Device B   Device C   Device D
 (CUDA)     (MPS)      (CPU)      (CUDA)
```

**Agent modes:**

| Mode | What it runs | Time |
|------|-------------|------|
| `--mode quick` | Baseline only | ~30 s |
| `--mode core` | Baseline + E1–E4 | ~10 min |
| `--mode full` | Baseline + E1–E7 | ~20–40 min |

---

## Command Reference

`main.py` accepts the following flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--video-sources A B …` | paths | — | Video files to use as streams |
| `--cameras N` | int | — | Replicate webcam index 0, N times |
| `--webcam` | flag | — | Single webcam mode |
| `--duration` | int | ∞ | Run length in seconds |
| `--device` | choice | `cuda` | `auto` / `cuda` / `mps` / `cpu` |
| `--model` | choice | `yolov8n.pt` | `yolov8{n,s,m,l,x}.pt` |
| `--batch-size` | int | `4` | Inference batch size |
| `--backend` | choice | `pytorch` | `pytorch` or `onnx` |
| `--fp16` | flag | off | FP16 mixed precision (CUDA only) |
| `--gpu-preprocess` | flag | off | GPU-side resize/normalize |
| `--run-experiments` | choice | `none` | Experiment suite to run after baseline |
| `--no-detailed-profiling` | flag | off | Disable per-stage profiler |
| `--onnx-coreml` | flag | off | CoreML EP for ONNX on Apple Silicon |

> Exactly one of `--video-sources`, `--cameras`, or `--webcam` must be provided.

---

## Output Files

**After a baseline run** (`results/`):

```
results/
├── run_summary.json            Machine-readable metrics (FPS, latency, config)
├── summary_report.txt          Human-readable performance summary
├── bottleneck_analysis.json    Per-stage E4 latency data
├── bottleneck_analysis.txt     Per-stage E4 data (human-readable)
├── stage_latency_histograms.png  Per-stage latency distributions
├── fps_comparison.png          Throughput-over-time plot
├── cpu_usage.png
├── memory_usage.png
├── gpu_usage.png
├── system_overview.png
└── system_stats.csv            Raw psutil samples (1 Hz)
```

**After `--run-experiments`**, baseline is preserved to `results/baseline/` and experiment outputs land in `experiments/results/`:

```
experiments/results/
├── experiment_metadata.json
├── e1_gpu_summary.json / e1_cpu_summary.json
├── e2_batch_{1,2,4,8,16}_summary.json
├── e2_batch_size_curve.png
├── e3_streams_{1,2,4,8}_summary.json
├── e3_amdahls_law_report.txt / e3_amdahls_law_plot.png
├── e5_yolov8{n,s,m,l,x}_summary.json / e5_model_sweep.png
├── e6_pytorch_{fp32,fp16}_summary.json / e6_onnxruntime_summary.json
└── e7_{cpu,gpu,gpu_fp16}_preproc_summary.json / e7_gpu_preproc.png
```

---

## Repository Layout

```
multi-camera-cv-system/
├── main.py                         Entry point
├── requirements.txt
│
├── config/
│   └── config.py                   SYSTEM_CONFIG + COCO class names
│
├── camera/
│   └── camera_stream.py            Per-camera mp.Process producer
│
├── detection/
│   ├── detector.py                 PyTorch YOLOv8 (FP16 + GPU preprocess)
│   └── onnx_detector.py            ONNX Runtime drop-in detector
│
├── pipeline/
│   └── pipeline_manager.py         Batched producer-consumer orchestrator
│
├── profiling/
│   ├── profiler.py                 System-level CPU / RAM / GPU monitoring
│   └── detailed_profiler.py        Per-stage latency profiler (E4)
│
├── analysis/
│   └── amdahls_law.py              Parallel-fraction estimator (E3)
│
├── experiments/
│   └── run_all_experiments.py      E1–E7 automation
│
├── visualization/
│   └── visualizer.py               Matplotlib plots + reports
│
├── utils/
│   └── logger.py                   SystemLogger
│
├── web/
│   ├── coordinator.py              FastAPI leaderboard server + SQLite
│   ├── agent.py                    Per-device benchmark runner + uploader
│   ├── dashboard.html              Single-page Chart.js dashboard
│   └── README.md                   Coordinator deployment guide
│
├── videos/                         Input video files (gitignored)
├── results/                        Baseline run outputs (gitignored)
└── experiments/results/            Full-suite outputs (gitignored)
```

---

## Architecture

```
  Each camera process reads its video at native FPS
  and places decoded frames into the shared queue.

  Camera 0 (mp.Process) ──┐
  Camera 1 (mp.Process) ──┤
  Camera 2 (mp.Process) ──┼──▶  mp.Queue  ──▶  Pipeline Manager (main process)
  Camera 3 (mp.Process) ──┤                            │
   ...                    │                     Batch formation
  Camera 7 (mp.Process) ──┘                            │
                                              GPU Detector
                                        (CUDA / MPS / CPU)
                                                        │
                                           Per-stage profiler:
                                           1. frame_loading
                                           2. preprocessing
                                           3. inference
                                           4. postprocessing
                                           5. output

  Background: system profiler (psutil) samples CPU/RAM/GPU at 1 Hz
```

---

## Hardware Notes

**NVIDIA CUDA** — The agent auto-enables `--fp16` and `--gpu-preprocess` for optimized baselines. ONNX Runtime uses `CUDAExecutionProvider`. Best overall performance.

**Apple Silicon (MPS)** — Metal Performance Shaders are used for inference. FP16 autocast is intentionally disabled because PyTorch's MPS backend has incomplete FP16 op coverage. ONNX Runtime uses `CPUExecutionProvider` (CoreML EP is incompatible with YOLOv8 dynamic-batch graphs).

**CPU-only** — All paths fall back gracefully. Expect 5–10× lower throughput compared to a mid-range GPU.

---

## Troubleshooting

**`CUDA out of memory`**
Reduce `--batch-size` or switch to a smaller model (`--model yolov8n.pt`). On 4 GB VRAM, eight streams may exceed capacity at batch sizes above 4.

**Pipeline reports 0 frames processed**
Either no `videos/cam*.mp4` files are present, or OpenCV cannot open them. Quick test:
```bash
python -c "import cv2; cap = cv2.VideoCapture('videos/cam1.mp4'); print(cap.read()[0])"
# Should print: True
```

**`Field power.draw is not supported` from nvidia-smi**
Some laptop GPUs disable power telemetry in vBIOS. FPS-per-watt measurements are unavailable on those cards.

**ONNX scenario silently skipped in E6**
`onnxruntime` is not installed. Run:
```bash
pip install onnxruntime        # CPU
pip install onnxruntime-gpu    # CUDA
```

**Subprocess hangs at shutdown (macOS)**
Occasionally observed when forced child termination corrupts the multiprocessing queue's pipe state. The pipeline manager mitigates this with a daemon-thread drain and bounded queue close. If it persists, open an issue.

**Agent upload fails (connection refused)**
Confirm the coordinator is running: `uvicorn web.coordinator:app --host 0.0.0.0 --port 8000`. Check firewall rules if connecting across a network.

---

## License

MIT. See `LICENSE`.
