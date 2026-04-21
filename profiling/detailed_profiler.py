# Enhanced profiler with stage-by-stage timing breakdown for bottleneck identification
"""
Detailed profiler for stage-by-stage performance analysis
Implements Experiment E4: Bottleneck Identification

Tracks per-stage latency distributions (mean + p50/p95/p99) so that latency
spikes, not just average time, can be attributed to a specific stage.
"""

import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.logger import get_logger


class DetailedProfiler:
    """
    Tracks time spent in each pipeline stage to identify bottlenecks.

    Stages:
    1. Frame Loading (I/O bound)        - capture -> queue wait
    2. Preprocessing (CPU/GPU bound)
    3. Inference (GPU/CPU bound)
    4. Postprocessing (CPU bound)
    5. Output/Display (I/O bound)

    This class is thread-safe so multiple consumers in the pipeline can record
    concurrently. Per-stage samples are bounded (ring-buffer) to avoid
    unbounded memory growth during long runs.
    """

    STAGES = [
        'frame_loading',
        'preprocessing',
        'inference',
        'postprocessing',
        'output',
    ]

    def __init__(self, max_samples_per_stage=50_000):
        self.logger = get_logger('DetailedProfiler')

        self._lock = threading.Lock()
        self._max_samples = max_samples_per_stage

        # Per-stage ring buffers of durations (seconds)
        self.stage_times = defaultdict(list)
        self.total_frames = 0

        # Wall-clock reference so we can compute throughput during a run
        self._start_wall = time.time()

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_stage(self, stage_name, duration):
        """Record time spent in a stage. Thread-safe."""
        if duration is None or duration < 0:
            return
        with self._lock:
            buf = self.stage_times[stage_name]
            buf.append(float(duration))
            if len(buf) > self._max_samples:
                # drop oldest half — keeps the memory bound cheap
                del buf[: len(buf) // 2]

    def stage_timer(self, stage_name):
        """Context-manager helper: ``with profiler.stage_timer('inference'): ...``"""
        return _StageTimer(self, stage_name)

    def mark_frame(self):
        """Increment frame counter (call once per fully processed frame)."""
        with self._lock:
            self.total_frames += 1

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def _snapshot(self):
        """Return a copy of the current stage samples under the lock."""
        with self._lock:
            return {k: list(v) for k, v in self.stage_times.items()}, self.total_frames

    def get_bottleneck_analysis(self):
        """
        Identify which stage dominates wall-time.

        Returns a dict with, per stage:
            total_time, percent, avg_time, p50, p95, p99, max, count
        and a top-level `bottleneck` entry keyed by share of total time.
        """
        samples, total_frames = self._snapshot()
        total_time = sum(sum(v) for v in samples.values())
        if total_time == 0:
            return {}

        analysis = {
            'total_time': total_time,
            'total_frames': total_frames,
            'stages': {},
        }

        max_percent = 0.0
        bottleneck = None

        for stage in self.STAGES:
            times = samples.get(stage) or []
            if not times:
                continue
            arr = np.asarray(times, dtype=np.float64)
            stage_total = float(arr.sum())
            percent = (stage_total / total_time) * 100.0

            analysis['stages'][stage] = {
                'total_time': stage_total,
                'percent': percent,
                'avg_time': float(arr.mean()),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'max': float(arr.max()),
                'count': int(arr.size),
            }

            if percent > max_percent:
                max_percent = percent
                bottleneck = stage

        analysis['bottleneck'] = {'stage': bottleneck, 'percent': max_percent}
        return analysis

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def print_bottleneck_report(self):
        """Print per-stage percentile breakdown and name the dominant stage."""
        analysis = self.get_bottleneck_analysis()
        if not analysis:
            print("No stage data collected")
            return

        print("\n" + "=" * 88)
        print("BOTTLENECK ANALYSIS (Experiment E4)")
        print("=" * 88)
        print(f"Total pipeline time: {analysis['total_time']:.3f}s "
              f"across {analysis['total_frames']} frames\n")

        header = (f"{'Stage':<16} {'Share':>7} {'Mean(ms)':>10} "
                  f"{'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} "
                  f"{'Max(ms)':>9} {'N':>8}")
        print(header)
        print("-" * 88)

        for stage in self.STAGES:
            info = analysis['stages'].get(stage)
            if not info:
                continue
            print(f"{stage:<16} {info['percent']:>6.1f}% "
                  f"{info['avg_time']*1000:>10.2f} "
                  f"{info['p50']*1000:>9.2f} "
                  f"{info['p95']*1000:>9.2f} "
                  f"{info['p99']*1000:>9.2f} "
                  f"{info['max']*1000:>9.2f} "
                  f"{info['count']:>8d}")
        print("-" * 88)

        b = analysis['bottleneck']
        print(f"\nBOTTLENECK: {b['stage']} ({b['percent']:.1f}% of total time)")
        print(self._recommendations(b['stage']))
        print("=" * 88 + "\n")

    def save_report(self, output_dir='results'):
        """Persist bottleneck analysis as JSON, text and histogram PNG."""
        analysis = self.get_bottleneck_analysis()
        if not analysis:
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. JSON dump (machine-readable)
        json_path = output_dir / 'bottleneck_analysis.json'
        with json_path.open('w') as f:
            json.dump(analysis, f, indent=2)

        # 2. Text summary (human-readable)
        txt_path = output_dir / 'bottleneck_analysis.txt'
        with txt_path.open('w') as f:
            f.write("BOTTLENECK ANALYSIS (Experiment E4)\n")
            f.write("=" * 88 + "\n")
            f.write(f"Total pipeline time: {analysis['total_time']:.3f}s\n")
            f.write(f"Total frames: {analysis['total_frames']}\n\n")
            f.write(f"{'Stage':<16} {'Share':>7} {'Mean(ms)':>10} "
                    f"{'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} "
                    f"{'Max(ms)':>9} {'N':>8}\n")
            f.write("-" * 88 + "\n")
            for stage in self.STAGES:
                info = analysis['stages'].get(stage)
                if not info:
                    continue
                f.write(f"{stage:<16} {info['percent']:>6.1f}% "
                        f"{info['avg_time']*1000:>10.2f} "
                        f"{info['p50']*1000:>9.2f} "
                        f"{info['p95']*1000:>9.2f} "
                        f"{info['p99']*1000:>9.2f} "
                        f"{info['max']*1000:>9.2f} "
                        f"{info['count']:>8d}\n")
            f.write("-" * 88 + "\n")
            b = analysis['bottleneck']
            f.write(f"\nBottleneck: {b['stage']} ({b['percent']:.1f}%)\n")
            f.write(self._recommendations(b['stage']) + "\n")

        # 3. Histogram PNG (only if matplotlib is available)
        png_path = output_dir / 'stage_latency_histograms.png'
        try:
            self._plot_histograms(png_path)
        except Exception as exc:  # pragma: no cover - optional
            self.logger.warning(f"Histogram plot failed: {exc}")
            png_path = None

        paths = {
            'json': str(json_path),
            'txt': str(txt_path),
            'histogram_png': str(png_path) if png_path else None,
        }
        self.logger.info(f"DetailedProfiler report saved: {paths}")
        return paths

    def _plot_histograms(self, output_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        samples, _ = self._snapshot()
        stages_with_data = [s for s in self.STAGES if samples.get(s)]
        if not stages_with_data:
            return

        fig, axes = plt.subplots(
            len(stages_with_data), 1,
            figsize=(9, 2.6 * len(stages_with_data)),
            sharex=False,
        )
        if len(stages_with_data) == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages_with_data):
            data_ms = np.asarray(samples[stage]) * 1000.0
            ax.hist(data_ms, bins=40, alpha=0.8, color='#3572A5', edgecolor='white')
            p50 = float(np.percentile(data_ms, 50))
            p95 = float(np.percentile(data_ms, 95))
            p99 = float(np.percentile(data_ms, 99))
            ax.axvline(p50, color='green', linestyle='--', linewidth=1, label=f'p50={p50:.1f}ms')
            ax.axvline(p95, color='orange', linestyle='--', linewidth=1, label=f'p95={p95:.1f}ms')
            ax.axvline(p99, color='red', linestyle='--', linewidth=1, label=f'p99={p99:.1f}ms')
            ax.set_title(f"{stage} latency (n={data_ms.size})")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Count")
            ax.legend(loc='best', fontsize=9)

        fig.suptitle("Per-stage latency distributions", fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Recommendations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _recommendations(stage):
        if stage == 'frame_loading':
            return (
                "Recommendations:\n"
                "  - Faster storage (NVMe), larger queue, hardware video decode.\n"
                "  - Increase buffer_size / max_queue_size; reduce resolution."
            )
        if stage == 'inference':
            return (
                "Recommendations:\n"
                "  - Raise batch size (until p95 latency degrades).\n"
                "  - Enable FP16 autocast / TensorRT / ONNX Runtime.\n"
                "  - Try a smaller YOLO variant (n/s) on the same hardware."
            )
        if stage == 'preprocessing':
            return (
                "Recommendations:\n"
                "  - Move resize/normalize to GPU (torchvision.transforms.v2 / Kornia).\n"
                "  - Use pinned memory + non_blocking=True H2D copies.\n"
                "  - Batch preprocessing across frames."
            )
        if stage == 'postprocessing':
            return (
                "Recommendations:\n"
                "  - Skip annotation drawing for benchmark runs.\n"
                "  - Vectorize NMS / bbox parsing; filter classes earlier."
            )
        if stage == 'output':
            return (
                "Recommendations:\n"
                "  - Async disk writes, lower output resolution, drop display path."
            )
        return ""


class _StageTimer:
    """Context manager returned by ``DetailedProfiler.stage_timer``."""

    __slots__ = ('_profiler', '_stage', '_t0')

    def __init__(self, profiler, stage):
        self._profiler = profiler
        self._stage = stage
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._profiler.record_stage(self._stage, time.perf_counter() - self._t0)
        return False
