"""
Automated Experiment Runner.

Implements the four experiments from the proposal (E1-E4) plus three
scaling experiments added for the extended project:

    E5 - Model-size sweep (yolov8n/s/m/l/x): accuracy-vs-throughput Pareto.
    E6 - Inference-engine sweep (PyTorch FP32 vs PyTorch FP16 vs ONNX Runtime).
    E7 - GPU preprocessing + CUDA streams overlap.

E3 now automatically feeds its measurements into AmdahlsLawAnalyzer and
saves both the text report and the speedup-curve PNG without any manual
step.

All runs spawn `main.py` as a subprocess so each experiment gets a clean
Python/CUDA state. Per-run metrics are read from `results/run_summary.json`
which `main.py` writes at shutdown.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

# Make project root importable so we can reuse AmdahlsLawAnalyzer and
# the visualizer without spawning a separate interpreter.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.amdahls_law import AmdahlsLawAnalyzer  # noqa: E402


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _safe_move(src: Path, dst: Path):
    """Move src -> dst if src exists, else warn."""
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    else:
        print(f"  (warning: {src} not found, skipping move)")


def _safe_copy(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))


class ExperimentRunner:
    """Automates running all experiments."""

    def __init__(self, output_dir='experiments/results'):
        self.project_root = PROJECT_ROOT
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.project_root / 'results'
        self.results = {}
        self.python_executable = sys.executable
        self.cuda_available = torch.cuda.is_available()

        self.video_directories = [
            self.project_root / 'videos',
            self.project_root / 'data' / 'videos',
            self.project_root / 'sample_videos',
        ]
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}

    # ------------------------------------------------------------------ #
    # Video / device helpers
    # ------------------------------------------------------------------ #
    def _discover_video_files(self):
        videos = []
        for d in self.video_directories:
            if d.exists():
                for p in sorted(d.iterdir()):
                    if p.is_file() and p.suffix.lower() in self.video_extensions:
                        videos.append(p.resolve())
        return videos

    def _get_video_sources(self, num_sources):
        videos = self._discover_video_files()
        if not videos:
            searched = ", ".join(str(d) for d in self.video_directories)
            raise FileNotFoundError(
                f"No input videos found in: {searched}. "
                "Add at least one .mp4/.avi/.mov file."
            )
        if len(videos) >= num_sources:
            return [str(p) for p in videos[:num_sources]]
        return [str(p) for p in itertools.islice(
            itertools.cycle(videos), num_sources)]

    def _resolve_device(self, requested):
        if requested == 'cuda' and not self.cuda_available:
            # Prefer Apple MPS over CPU when running on Mac Silicon.
            try:
                if (getattr(torch.backends, 'mps', None)
                        and torch.backends.mps.is_available()):
                    print("  note: CUDA unavailable, using MPS (Apple Silicon)")
                    return 'mps'
            except Exception:
                pass
            print("  note: CUDA unavailable, running on CPU")
            return 'cpu'
        return requested

    # ------------------------------------------------------------------ #
    # main.py invocation
    # ------------------------------------------------------------------ #
    def _build_cmd(self, *, num_sources, device, batch_size, duration,
                   model='yolov8n.pt', backend='pytorch', fp16=False,
                   gpu_preprocess=False):
        resolved = self._resolve_device(device)
        sources = self._get_video_sources(num_sources)
        cmd = [
            self.python_executable, 'main.py',
            '--video-sources', *sources,
            '--device', resolved,
            '--batch-size', str(batch_size),
            '--duration', str(duration),
            '--model', model,
            '--backend', backend,
        ]
        if fp16:
            cmd.append('--fp16')
        if gpu_preprocess:
            cmd.append('--gpu-preprocess')
        return cmd, resolved, sources

    def _run_and_capture(self, cmd, label_prefix):
        """Run main.py, then relocate its results/ files under experiments/results/."""
        subprocess.run(cmd, cwd=self.project_root)

        # Read the machine-readable summary (for aggregation)
        summary_path = self.results_dir / 'run_summary.json'
        run_summary = {}
        if summary_path.exists():
            try:
                run_summary = json.loads(summary_path.read_text())
            except Exception as exc:
                print(f"  (warning: could not parse run_summary.json: {exc})")

        # Relocate artifacts for archival (summary, stats, bottleneck, json)
        for src_name, dst_suffix in [
            ('summary_report.txt', '_report.txt'),
            ('system_stats.csv', '_stats.csv'),
            ('run_summary.json', '_summary.json'),
            ('bottleneck_analysis.txt', '_bottleneck.txt'),
            ('bottleneck_analysis.json', '_bottleneck.json'),
            ('stage_latency_histograms.png', '_stage_latencies.png'),
        ]:
            _safe_move(self.results_dir / src_name,
                       self.output_dir / f'{label_prefix}{dst_suffix}')

        return run_summary

    # ================================================================== #
    # E1 - CPU vs GPU
    # ================================================================== #
    def run_experiment_e1_cpu_vs_gpu(self, duration=30):
        print("\n" + "=" * 70)
        print("EXPERIMENT E1: CPU vs GPU Inference")
        print("=" * 70)

        runs = []

        gpu_cmd, gpu_device, _ = self._build_cmd(
            num_sources=4, device='cuda', batch_size=4, duration=duration)
        print(f"\n[1/2] Running {gpu_device.upper()} ...")
        gpu_summary = self._run_and_capture(gpu_cmd, 'e1_gpu')
        runs.append(('gpu', gpu_device, gpu_summary))

        time.sleep(2)

        cpu_cmd, cpu_device, _ = self._build_cmd(
            num_sources=4, device='cpu', batch_size=2, duration=duration)
        print(f"\n[2/2] Running {cpu_device.upper()} ...")
        cpu_summary = self._run_and_capture(cpu_cmd, 'e1_cpu')
        runs.append(('cpu', cpu_device, cpu_summary))

        self.results['E1'] = {'runs': [
            {'label': lab, 'device': dev,
             'overall_fps': s.get('pipeline', {}).get('overall_fps'),
             'avg_latency_ms': s.get('pipeline', {}).get('avg_latency_ms')}
            for lab, dev, s in runs
        ]}
        print("\nE1 complete.")

    # ================================================================== #
    # E2 - Batch size sweep
    # ================================================================== #
    def run_experiment_e2_batch_size(self, duration=20):
        print("\n" + "=" * 70)
        print("EXPERIMENT E2: Batch Size vs Throughput")
        print("=" * 70)

        batch_sizes = [1, 2, 4, 8, 16]
        collected = []

        for i, bs in enumerate(batch_sizes):
            print(f"\n[{i+1}/{len(batch_sizes)}] batch_size={bs}")
            cmd, resolved, _ = self._build_cmd(
                num_sources=4, device='cuda', batch_size=bs, duration=duration)
            summary = self._run_and_capture(cmd, f'e2_batch_{bs}')
            collected.append({
                'batch_size': bs,
                'device': resolved,
                'fps': summary.get('pipeline', {}).get('overall_fps'),
                'latency_ms': summary.get('pipeline', {}).get('avg_latency_ms'),
            })
            time.sleep(2)

        self.results['E2'] = {'runs': collected}
        self._plot_batch_size_curve(collected)
        print("\nE2 complete.")

    # ================================================================== #
    # E3 - Multi-stream scaling (auto-runs Amdahl analysis)
    # ================================================================== #
    def run_experiment_e3_scaling(self, duration=30):
        print("\n" + "=" * 70)
        print("EXPERIMENT E3: Multi-Stream Scaling (Amdahl's Law)")
        print("=" * 70)

        stream_counts = [1, 2, 4, 8]
        collected = []

        for i, n in enumerate(stream_counts):
            print(f"\n[{i+1}/{len(stream_counts)}] streams={n}")
            cmd, resolved, _ = self._build_cmd(
                num_sources=n, device='cuda', batch_size=4, duration=duration)
            summary = self._run_and_capture(cmd, f'e3_streams_{n}')
            pipeline = summary.get('pipeline', {})
            collected.append({
                'streams': n,
                'device': resolved,
                'fps': pipeline.get('overall_fps'),
                'latency_ms': pipeline.get('avg_latency_ms'),
            })
            time.sleep(2)

        self.results['E3'] = {'runs': collected}

        # ----- Automated Amdahl analysis -----
        analyzer = AmdahlsLawAnalyzer()
        for row in collected:
            if row['fps'] is not None and row['fps'] > 0:
                analyzer.add_measurement(row['streams'], row['fps'],
                                         latency=(row['latency_ms'] or 0) / 1000.0)

        if len(analyzer.measurements) >= 2:
            report_file = self.output_dir / 'e3_amdahls_law_report.txt'
            plot_file = self.output_dir / 'e3_amdahls_law_plot.png'
            analyzer.generate_report(output_file=str(report_file))
            analyzer.plot_speedup_curve(output_file=str(plot_file))
            print(f"\nAmdahl's Law outputs:")
            print(f"  report: {report_file}")
            print(f"  plot:   {plot_file}")
        else:
            print("\nSkipping Amdahl analysis (need >=2 valid runs)")

        print("E3 complete.")

    # ================================================================== #
    # E4 - Bottleneck (stage-level percentiles, now auto-saved)
    # ================================================================== #
    def run_experiment_e4_bottleneck(self, duration=30):
        print("\n" + "=" * 70)
        print("EXPERIMENT E4: Bottleneck Identification")
        print("=" * 70)

        cmd, resolved, _ = self._build_cmd(
            num_sources=4, device='cuda', batch_size=4, duration=duration)
        summary = self._run_and_capture(cmd, 'e4')
        self.results['E4'] = {
            'device': resolved,
            'pipeline': summary.get('pipeline', {}),
        }
        print("E4 complete.  Bottleneck breakdown is in "
              "experiments/results/e4_bottleneck.* and stage_latencies.png")

    # ================================================================== #
    # E5 - Model-size sweep
    # ================================================================== #
    def run_experiment_e5_model_sweep(self, duration=20):
        print("\n" + "=" * 70)
        print("EXPERIMENT E5: Model-Size Sweep (Accuracy vs Speed)")
        print("=" * 70)

        variants = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
                    'yolov8l.pt', 'yolov8x.pt']
        collected = []

        for i, model in enumerate(variants):
            print(f"\n[{i+1}/{len(variants)}] model={model}")
            try:
                cmd, resolved, _ = self._build_cmd(
                    num_sources=4, device='cuda', batch_size=4,
                    duration=duration, model=model)
                summary = self._run_and_capture(cmd, f'e5_{model.replace(".pt", "")}')
                pipeline = summary.get('pipeline', {})
                detector = summary.get('detector', {})
                collected.append({
                    'model': model,
                    'device': resolved,
                    'fps': pipeline.get('overall_fps'),
                    'latency_ms': pipeline.get('avg_latency_ms'),
                    'total_detections': detector.get('total_detections'),
                })
            except Exception as exc:
                print(f"  (skipping {model}: {exc})")
            time.sleep(2)

        self.results['E5'] = {'runs': collected}
        self._plot_model_sweep(collected)
        print("\nE5 complete.")

    # ================================================================== #
    # E6 - Inference engines (PyTorch FP32 vs FP16 vs ONNX)
    # ================================================================== #
    def run_experiment_e6_engines(self, duration=20):
        print("\n" + "=" * 70)
        print("EXPERIMENT E6: Inference Engine Comparison")
        print("=" * 70)

        scenarios = [
            ('pytorch_fp32', {'backend': 'pytorch', 'fp16': False}),
            ('pytorch_fp16', {'backend': 'pytorch', 'fp16': True}),
            ('onnxruntime',  {'backend': 'onnx',    'fp16': False}),
        ]
        collected = []

        for i, (label, kwargs) in enumerate(scenarios):
            print(f"\n[{i+1}/{len(scenarios)}] {label}")
            try:
                cmd, resolved, _ = self._build_cmd(
                    num_sources=4, device='cuda', batch_size=4,
                    duration=duration, **kwargs)
                summary = self._run_and_capture(cmd, f'e6_{label}')
                pipeline = summary.get('pipeline', {})
                collected.append({
                    'scenario': label,
                    'device': resolved,
                    'fps': pipeline.get('overall_fps'),
                    'latency_ms': pipeline.get('avg_latency_ms'),
                })
            except Exception as exc:
                print(f"  (skipping {label}: {exc})")
            time.sleep(2)

        self.results['E6'] = {'runs': collected}
        self._plot_engine_comparison(collected)
        print("\nE6 complete.")

    # ================================================================== #
    # E7 - GPU preprocessing + CUDA streams
    # ================================================================== #
    def run_experiment_e7_gpu_preprocess(self, duration=20):
        print("\n" + "=" * 70)
        print("EXPERIMENT E7: GPU Preprocessing + CUDA Streams Overlap")
        print("=" * 70)

        scenarios = [
            ('cpu_preproc',      {'gpu_preprocess': False, 'fp16': False}),
            ('gpu_preproc',      {'gpu_preprocess': True,  'fp16': False}),
            ('gpu_preproc_fp16', {'gpu_preprocess': True,  'fp16': True}),
        ]
        collected = []

        for i, (label, kwargs) in enumerate(scenarios):
            print(f"\n[{i+1}/{len(scenarios)}] {label}")
            cmd, resolved, _ = self._build_cmd(
                num_sources=4, device='cuda', batch_size=4,
                duration=duration, **kwargs)
            summary = self._run_and_capture(cmd, f'e7_{label}')
            pipeline = summary.get('pipeline', {})
            collected.append({
                'scenario': label,
                'device': resolved,
                'fps': pipeline.get('overall_fps'),
                'latency_ms': pipeline.get('avg_latency_ms'),
            })
            time.sleep(2)

        self.results['E7'] = {'runs': collected}
        self._plot_engine_comparison(collected, fname='e7_gpu_preproc.png',
                                     title='E7: GPU preprocessing + CUDA streams')
        print("\nE7 complete.")

    # ================================================================== #
    # Plot helpers
    # ================================================================== #
    def _plot_batch_size_curve(self, rows):
        self._safe_plot(
            rows, x_key='batch_size', y_key='fps',
            title='E2: Throughput vs Batch Size',
            xlabel='Batch size', ylabel='FPS',
            fname='e2_batch_size_curve.png',
        )

    def _plot_model_sweep(self, rows):
        self._safe_plot(
            rows, x_key='model', y_key='fps',
            title='E5: Model size vs Throughput',
            xlabel='YOLOv8 variant', ylabel='FPS',
            fname='e5_model_sweep.png', bar=True,
        )

    def _plot_engine_comparison(self, rows, fname='e6_engines.png',
                                title='E6: Inference engine comparison'):
        self._safe_plot(
            rows, x_key='scenario', y_key='fps',
            title=title, xlabel='Scenario', ylabel='FPS',
            fname=fname, bar=True,
        )

    def _safe_plot(self, rows, *, x_key, y_key, title, xlabel, ylabel,
                   fname, bar=False):
        rows = [r for r in rows if r.get(y_key) is not None]
        if not rows:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            xs = [str(r[x_key]) for r in rows]
            ys = [r[y_key] for r in rows]

            fig, ax = plt.subplots(figsize=(8, 5))
            if bar:
                ax.bar(xs, ys, color='#3572A5')
                for x, y in zip(xs, ys):
                    ax.text(x, y, f"{y:.1f}", ha='center', va='bottom', fontsize=9)
            else:
                ax.plot(xs, ys, marker='o', linewidth=2, color='#3572A5')
                for x, y in zip(xs, ys):
                    ax.text(x, y, f"{y:.1f}", fontsize=9)

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            out = self.output_dir / fname
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"  plot saved: {out}")
        except Exception as exc:  # pragma: no cover
            print(f"  (plot {fname} skipped: {exc})")

    # ================================================================== #
    # Aggregate runner
    # ================================================================== #
    def run_all_experiments(self, duration=None, include_extended=True):
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENTS")
        print("=" * 70)

        start = time.time()
        d = duration or 30

        self.run_experiment_e1_cpu_vs_gpu(duration=d)
        self.run_experiment_e2_batch_size(duration=max(15, d - 10))
        self.run_experiment_e3_scaling(duration=d)
        self.run_experiment_e4_bottleneck(duration=d)
        if include_extended:
            self.run_experiment_e5_model_sweep(duration=max(15, d - 10))
            self.run_experiment_e6_engines(duration=max(15, d - 10))
            self.run_experiment_e7_gpu_preprocess(duration=max(15, d - 10))

        elapsed = time.time() - start
        print("\n" + "=" * 70)
        print(f"ALL EXPERIMENTS DONE in {elapsed/60:.1f} min")
        print(f"Results in: {self.output_dir}")
        print("=" * 70)

        meta = {
            'total_time_s': elapsed,
            'cuda_available': self.cuda_available,
            'experiments': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        (self.output_dir / 'experiment_metadata.json').write_text(
            json.dumps(meta, indent=2, default=str))


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Run experiments E1-E7')
    parser.add_argument(
        '--experiment',
        choices=['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'all', 'core'],
        default='all',
        help='Which experiment to run (core = E1-E4 only)',
    )
    parser.add_argument('--duration', type=int, default=30,
                        help='Per-run duration (seconds)')
    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.experiment == 'all':
        runner.run_all_experiments(duration=args.duration,
                                   include_extended=True)
    elif args.experiment == 'core':
        runner.run_all_experiments(duration=args.duration,
                                   include_extended=False)
    elif args.experiment == 'e1':
        runner.run_experiment_e1_cpu_vs_gpu(args.duration)
    elif args.experiment == 'e2':
        runner.run_experiment_e2_batch_size(args.duration)
    elif args.experiment == 'e3':
        runner.run_experiment_e3_scaling(args.duration)
    elif args.experiment == 'e4':
        runner.run_experiment_e4_bottleneck(args.duration)
    elif args.experiment == 'e5':
        runner.run_experiment_e5_model_sweep(args.duration)
    elif args.experiment == 'e6':
        runner.run_experiment_e6_engines(args.duration)
    elif args.experiment == 'e7':
        runner.run_experiment_e7_gpu_preprocess(args.duration)


if __name__ == '__main__':
    main()
