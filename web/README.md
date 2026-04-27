# Multi-Camera CV Benchmark — Async Leaderboard

This folder contains the dashboard layer for the project. It lets
multiple laptops (mixed CUDA + Apple Silicon) run the benchmark
independently and accumulate results on a shared leaderboard, so the
team can compare hardware side by side without anyone having to be
online at the same time.

## Architecture

```
┌───────────────────┐      zip upload (POST)       ┌──────────────────┐
│  agent.py  on     │  ───────────────────────▶    │  coordinator.py  │
│  each laptop      │                              │  FastAPI server  │
└────────┬──────────┘                              │  + SQLite DB     │
         │                                         │  + dashboard.html│
         │ runs `python main.py --run-experiments`  └─────────┬────────┘
         │                                                    │
         │                                                    │ serves
         ▼                                                    ▼
   results/baseline/   ─── bundled ───▶    http://HOST:8000/  leaderboard
   experiments/results/                    (anyone can watch)
```

## Running the coordinator (one machine, any OS)

```bash
cd multi-camera-cv-system
pip install -r web/requirements.txt
uvicorn web.coordinator:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/>.

If your laptops are on different networks, the easiest way to expose
this publicly is [Tailscale](https://tailscale.com): install on the
coordinator and on every participating laptop, then the coordinator is
reachable at its stable `100.x.y.z` address. No port forwarding, no
NAT pain, free for personal use.

## Submitting a run (from any participating laptop)

On the laptop you want to benchmark:

```bash
cd multi-camera-cv-system
python web/agent.py \
    --server http://COORDINATOR_HOST:8000 \
    --device-label "Sahil's M4 Air" \
    --mode full
```

This runs the entire pipeline under `main.py --run-experiments all`,
then bundles `results/baseline/` + `experiments/results/` into a zip
and uploads it. Total wall-clock: ~20–40 minutes for the full suite.

### Modes

| `--mode`  | What it runs                                         | Time         |
|-----------|------------------------------------------------------|--------------|
| `quick`   | Baseline only (30s run of `main.py`)                 | ~30 seconds  |
| `core`    | Baseline + E1–E4                                     | ~10 minutes  |
| `full`    | Baseline + E1–E7 (default, full experiment suite)    | ~20–40 min   |

### Other agent flags

```
--device auto|cuda|mps|cpu   (default auto)
--duration 30                baseline duration in seconds
--model yolov8n.pt           YOLO variant for the baseline run
--notes "..."                free-form notes shown next to this submission
--skip-run                   don't run main.py, just re-upload existing results/
--dry-run                    build the zip but don't upload
```

### One-line prerequisites on the client

The agent uses only the stdlib for upload and `psutil`/`torch` for
hardware autodetect. Since the laptop already has the project running,
everything you need should be installed. If not:

```bash
pip install psutil
```

## What happens inside `main.py`

The new `--run-experiments` flag (values:
`none`/`core`/`all`/`e1`…`e7`) turns `main.py` into a single entry
point that produces:

1. The usual baseline run (configurable via `--video-sources`,
   `--device`, `--duration`, …)
2. Preserves every artifact from that run into `results/baseline/`
3. Then calls `ExperimentRunner` to execute the requested experiment
   set, which spawns further `main.py` subprocesses (without the
   `--run-experiments` flag, so no recursion)
4. Experiment results land in `experiments/results/` with per-run
   `e*_summary.json` and the consolidated `experiment_metadata.json`

So one command reproduces the whole dataset the dashboard expects.

## What the dashboard shows

- **Leaderboard table** — one row per submission (most recent first)
- **Headline FPS chart** — bar per device, sortable by throughput
- **Latency chart** — bar per device
- **Experiment-suite tabs (E1, E2, E3, E5, E6, E7)** — overlay lines
  or grouped bars across all devices that submitted a full suite
- **Device detail pane** — click a row to see hardware, config,
  baseline numbers, and every PNG plot that device produced (stage
  histograms, Amdahl curve, batch-size curve, model sweep, etc.)

The dashboard auto-refreshes every 30 seconds, so someone running an
experiment on the coordinator host will see their result pop in when
their agent finishes.

## Data layout on the coordinator

```
web/data/
  runs.db                           # SQLite
  submissions/
    <id>/
      bundle.zip                    # the raw upload
      manifest.json                 # identity + hardware
      baseline/
        run_summary.json
        summary_report.txt
        bottleneck_analysis.json
        stage_latency_histograms.png
        ...
      experiments/
        experiment_metadata.json
        e1_gpu_summary.json ... e7_*_summary.json
        e2_batch_size_curve.png
        e3_amdahls_law_plot.png
        e5_model_sweep.png
        e6_engines.png
        e7_gpu_preproc.png
      plots/                         # denormalised copy served to dashboard
```

## Admin

- **Delete a bad submission**: `curl -X DELETE
  http://HOST:8000/api/submissions/<id>`
- **Raw data dump**: `GET /api/submissions` (list) or
  `GET /api/submissions/<id>` (detail JSON).
- **Health check**: `GET /api/health`.

## Security caveats

There is no authentication. Anyone who can reach the coordinator can
submit and delete. For a coursework demo on Tailscale this is fine —
only your teammates can reach the tailnet IP. Don't expose
`:8000` to the public internet without adding at least basic auth.
