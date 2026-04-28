# Main Entry Point
"""
Multi-Camera GPU-Accelerated Computer Vision System
Main entry point

Usage:
    python main.py --cameras 4 --duration 60
    python main.py --webcam --duration 30
    python main.py --video-sources v1.mp4 v2.mp4 v3.mp4
"""

import sys
import argparse
import signal
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import SYSTEM_CONFIG
from camera.camera_stream import MultiCameraManager
from detection.detector import YOLODetector
from pipeline.pipeline_manager import PipelineManager
from profiling.profiler import Profiler
from visualization.visualizer import Visualizer
from utils.logger import SystemLogger, get_logger


class MultiCameraSystem:
    """Main system orchestrator"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize logger
        self.system_logger = SystemLogger()
        self.logger = get_logger()
        
        self.logger.info("="*60)
        self.logger.info("Multi-Camera GPU-Accelerated CV System")
        self.logger.info("="*60)
        
        self.camera_manager = None
        self.detector = None
        self.profiler = None
        self.pipeline_manager = None
        
        self.running = False
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.logger.info("Shutdown signal received")
        self.stop()
    
    def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing components...")
        
        try:
            # Profiler
            self.logger.info("Initializing profiler...")
            self.profiler = Profiler(self.config)

            # Camera manager
            self.logger.info("Initializing camera manager...")
            self.camera_manager = MultiCameraManager(self.config)

            # On Windows, starting worker processes before CUDA initialization
            # avoids child startup issues after the parent has touched the GPU.
            if self.config.get('device') == 'cuda':
                self.logger.info("Pre-starting camera manager before CUDA initialization...")
                self.camera_manager.start()

            # Detector (backend selectable: pytorch | onnx)
            self.logger.info("Initializing detector...")
            backend = self.config.get('backend', 'pytorch')
            if backend == 'onnx':
                from detection.onnx_detector import ONNXDetector  # lazy import
                self.detector = ONNXDetector(self.config)
            else:
                self.detector = YOLODetector(self.config)
            
            # Pipeline
            self.logger.info("Initializing pipeline...")
            self.pipeline_manager = PipelineManager(
                self.camera_manager,
                self.detector,
                self.profiler,
                self.config
            )
            
            self.logger.info("All components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            if self.camera_manager:
                self.camera_manager.stop()
            return False
    
    def start(self, duration=None):
        """Start the system"""
        if not self.initialize():
            self.logger.error("Cannot start - initialization failed")
            return
        
        self.logger.info("Starting system...")
        self.running = True
        
        # Start profiler
        self.profiler.start()
        
        # Start pipeline (starts cameras)
        self.pipeline_manager.start()
        
        self.logger.info("System started")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            if duration:
                self.logger.info(f"Running for {duration}s...")
                time.sleep(duration)
                self.stop()
            else:
                while self.running:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
            self.stop()
    
    def stop(self):
        """Stop the system"""
        if not self.running:
            return
        
        self.logger.info("Stopping system...")
        self.running = False
        
        # Stop pipeline (stops cameras)
        if self.pipeline_manager:
            self.pipeline_manager.stop()
        
        # Stop profiler
        if self.profiler:
            self.profiler.stop()
        
        # Generate visualizations
        self._generate_visualizations()
        
        self.logger.info("System stopped")
    
    def _generate_visualizations(self):
        """Generate performance visualizations"""
        self.logger.info("Generating visualizations...")

        try:
            viz = Visualizer(
                self.config['stats_file'],
                self.config['output_dir']
            )

            viz.generate_all_plots()

            summary = self.profiler.get_summary()
            viz.generate_summary_report(summary)

            self.profiler.print_summary()
            self.detector.print_stats()

            # Machine-readable run summary for experiment runners
            self._write_run_summary(summary)

            self.logger.info("Visualizations complete")

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")

    def _write_run_summary(self, profiler_summary):
        """Write a compact JSON summary of the run for automated analysis."""
        import json
        from pathlib import Path as _P

        output_dir = _P(self.config.get('output_dir', 'results/'))
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_summary = (self.pipeline_manager.get_summary()
                            if self.pipeline_manager else {})
        detector_stats = (self.detector.get_statistics()
                          if self.detector else {})

        payload = {
            'config': {
                'device': self.config.get('device'),
                'model_name': self.config.get('model_name'),
                'batch_size': self.config.get('batch_size'),
                'num_cameras': self.config.get('num_cameras'),
                'use_fp16': self.config.get('use_fp16', False),
                'use_gpu_preprocess': self.config.get('use_gpu_preprocess', False),
                'backend': self.config.get('backend', 'pytorch'),
            },
            'pipeline': pipeline_summary,
            'detector': detector_stats,
            'profiler': profiler_summary,
        }

        out_path = output_dir / 'run_summary.json'
        # encoding='utf-8' for cross-platform safety. Without it Windows
        # defaults to cp1252 and any non-ASCII string in the payload crashes.
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
        self.logger.info(f"Saved run summary: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Camera GPU-Accelerated Computer Vision System'
    )
    
    # Camera sources
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--cameras',
        type=int,
        help='Number of cameras (uses webcam index 0)'
    )
    source_group.add_argument(
        '--video-sources',
        nargs='+',
        help='Video file paths'
    )
    source_group.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam (single camera)'
    )
    
    # Options
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in seconds (default: continuous)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model (n=fastest, x=most accurate)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps', 'auto'],
        help='Processing device (mps = Apple Silicon GPU, auto = pick best)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='pytorch',
        choices=['pytorch', 'onnx'],
        help='Inference backend (pytorch = Ultralytics YOLOv8, onnx = ONNX Runtime)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Enable FP16 mixed precision (CUDA only, no-op on CPU/MPS)'
    )
    parser.add_argument(
        '--gpu-preprocess',
        action='store_true',
        help='Run resize/normalize on the GPU (requires CUDA or MPS)'
    )
    parser.add_argument(
        '--no-detailed-profiling',
        action='store_true',
        help='Disable per-stage profiler (default: enabled)'
    )
    parser.add_argument(
        '--onnx-coreml',
        action='store_true',
        help='Opt into CoreMLExecutionProvider for ONNX Runtime on Apple '
             'Silicon. Off by default because CoreML NN backend is '
             'incompatible with YOLOv8 dynamic-batch graphs. CPU is the '
             'safe fallback.'
    )
    parser.add_argument(
        '--run-experiments',
        type=str,
        default='none',
        choices=['none', 'core', 'all',
                 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'],
        help=('After the baseline run, also run the experiment suite. '
              'core=E1-E4, all=E1-E7, or pick a single experiment.')
    )

    args = parser.parse_args()

    # Create configuration
    config = SYSTEM_CONFIG.copy()
    config['model_name'] = args.model
    config['device'] = args.device
    config['batch_size'] = args.batch_size
    config['backend'] = args.backend
    config['use_fp16'] = bool(args.fp16)
    config['use_gpu_preprocess'] = bool(args.gpu_preprocess)
    config['enable_detailed_profiling'] = not args.no_detailed_profiling
    config['onnx_use_coreml'] = bool(args.onnx_coreml)
    
    if args.webcam:
        config['num_cameras'] = 1
        config['video_sources'] = [0]
    elif args.cameras:
        config['num_cameras'] = args.cameras
        config['video_sources'] = [0] * args.cameras
    else:
        config['video_sources'] = args.video_sources
        config['num_cameras'] = len(args.video_sources)
    
    # Print configuration
    print("\n" + "="*70)
    print("MULTI-CAMERA COMPUTER VISION SYSTEM")
    print("="*70)
    print(f"Cameras: {config['num_cameras']}")
    print(f"Model: {config['model_name']}")
    print(f"Device: {config['device'].upper()}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Backend: {config['backend']}")
    print(f"FP16: {config['use_fp16']} | GPU preprocess: {config['use_gpu_preprocess']}")
    print(f"Duration: {args.duration if args.duration else 'Continuous'}s")
    print("="*70)
    print("\nKey Features:")
    print("  * Multiprocessing (one process per camera)")
    print("  * GPU-accelerated detection (CUDA / MPS / CPU)")
    print("  * Per-stage latency histograms (E4)")
    print("  * Machine-readable run_summary.json for automation")
    print("="*70 + "\n")
    
    # Run system
    system = MultiCameraSystem(config)
    system.start(duration=args.duration)

    # ---------------------------------------------------------------- #
    # Optional: chain the experiment suite after the baseline run.
    #
    # ExperimentRunner spawns main.py subprocesses for each experiment
    # configuration, but those subprocess invocations do NOT include
    # --run-experiments, so there is no recursion.
    # ---------------------------------------------------------------- #
    if args.run_experiments != 'none':
        import shutil
        from pathlib import Path as _P

        results_dir = _P(config.get('output_dir', 'results/'))
        baseline_dir = results_dir / 'baseline'
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the baseline artifacts BEFORE ExperimentRunner starts
        # overwriting results/ with per-experiment runs.
        preserved = 0
        for src in results_dir.iterdir():
            if src.is_file() and src.name != '.DS_Store':
                try:
                    shutil.copy2(src, baseline_dir / src.name)
                    preserved += 1
                except Exception as exc:
                    print(f"  warning: could not preserve {src.name}: {exc}")

        print("\n" + "=" * 70)
        print(f"BASELINE RUN COMPLETE. Starting experiment suite: "
              f"{args.run_experiments}")
        print(f"Preserved {preserved} baseline artifacts in: {baseline_dir}")
        print("=" * 70 + "\n")

        try:
            from experiments.run_all_experiments import ExperimentRunner
        except Exception as exc:
            print(f"[ERROR] Could not import ExperimentRunner: {exc}")
            return

        runner = ExperimentRunner()
        exp_duration = args.duration or 30
        choice = args.run_experiments

        dispatch = {
            'e1': lambda: runner.run_experiment_e1_cpu_vs_gpu(exp_duration),
            'e2': lambda: runner.run_experiment_e2_batch_size(exp_duration),
            'e3': lambda: runner.run_experiment_e3_scaling(exp_duration),
            'e4': lambda: runner.run_experiment_e4_bottleneck(exp_duration),
            'e5': lambda: runner.run_experiment_e5_model_sweep(exp_duration),
            'e6': lambda: runner.run_experiment_e6_engines(exp_duration),
            'e7': lambda: runner.run_experiment_e7_gpu_preprocess(exp_duration),
            'core': lambda: runner.run_all_experiments(
                duration=exp_duration, include_extended=False),
            'all': lambda: runner.run_all_experiments(
                duration=exp_duration, include_extended=True),
        }

        try:
            dispatch[choice]()
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Experiment suite failed: {exc}")
            traceback.print_exc()

        print("\n" + "=" * 70)
        print("ALL DONE.")
        print(f"  Baseline:    {baseline_dir}")
        print(f"  Experiments: {runner.output_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
