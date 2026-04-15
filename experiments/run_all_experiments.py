"""
Automated Experiment Runner
Runs all 4 experiments from the proposal and generates comprehensive reports
"""

import subprocess
import time
import json
import sys
import itertools
from pathlib import Path
import torch
import shutil   # ✅ ADDED


class ExperimentRunner:
    """
    Automates running all experiments from the proposal.
    
    Experiments:
    E1: CPU vs GPU Inference
    E2: Batch Size vs Throughput  
    E3: Multi-Stream Scaling
    E4: Bottleneck Identification
    """
    
    def __init__(self, output_dir='experiments/results'):
        self.project_root = Path(__file__).resolve().parent.parent
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.python_executable = sys.executable
        self.cuda_available = torch.cuda.is_available()
        self.video_directories = [
            self.project_root / 'videos',
            self.project_root / 'data' / 'videos',
            self.project_root / 'sample_videos',
        ]
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}

    # ✅ HELPER FUNCTION ADDED
    def _move_file(self, src, dst):
        src_path = self.project_root / src
        if src_path.exists():
            shutil.move(str(src_path), str(dst))
        else:
            print(f"⚠ Warning: {src} not found, skipping move.")

    
    def _discover_video_files(self):
        video_files = []
        for directory in self.video_directories:
            if not directory.exists():
                continue

            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in self.video_extensions:
                    video_files.append(path.resolve())

        return video_files

    def _get_video_sources(self, num_sources):
        video_files = self._discover_video_files()
        if not video_files:
            searched_dirs = ", ".join(str(path) for path in self.video_directories)
            raise FileNotFoundError(
                "No input videos found. Add at least one video file to one of these folders: "
                f"{searched_dirs}"
            )

        if len(video_files) >= num_sources:
            selected_files = video_files[:num_sources]
        else:
            selected_files = list(itertools.islice(itertools.cycle(video_files), num_sources))

        return [str(path) for path in selected_files]

    def _resolve_device(self, requested_device):
        if requested_device == 'cuda' and not self.cuda_available:
            print("\nWarning: CUDA requested but not available. Running on CPU instead.")
            return 'cpu'
        return requested_device

    def _build_main_command(self, num_sources, requested_device, batch_size, duration):
        resolved_device = self._resolve_device(requested_device)
        video_sources = self._get_video_sources(num_sources)
        cmd = [
            self.python_executable, 'main.py',
            '--video-sources', *video_sources,
            '--device', resolved_device,
            '--batch-size', str(batch_size),
            '--duration', str(duration)
        ]
        return cmd, resolved_device, video_sources

    def run_experiment_e1_cpu_vs_gpu(self, duration=30):
        """
        E1: CPU vs GPU Inference
        Metric: Latency (ms/frame), Throughput (FPS)
        Concept: GPU SIMD parallelism, data-level parallelism
        """
        print("\n" + "="*70)
        print("EXPERIMENT E1: CPU vs GPU Inference")
        print("="*70)
        
        results = {}
        
        # GPU run
        gpu_cmd, gpu_device, gpu_sources = self._build_main_command(4, 'cuda', 4, duration)
        print(f"\n[1/2] Running with {gpu_device.upper()} on {len(gpu_sources)} video stream(s)...")
        subprocess.run(gpu_cmd, cwd=self.project_root)
        
        # Save GPU results
        self._move_file('results/summary_report.txt', 
                        self.output_dir / 'e1_gpu_report.txt')
        self._move_file('results/system_stats.csv',
                        self.output_dir / 'e1_gpu_stats.csv')
        
        time.sleep(2)
        
        # CPU run
        print("\n[2/2] Running with CPU...")
        cpu_cmd, cpu_device, cpu_sources = self._build_main_command(4, 'cpu', 2, duration)
        print(f"Using {cpu_device.upper()} on {len(cpu_sources)} video stream(s)...")
        subprocess.run(cpu_cmd, cwd=self.project_root)
        
        # Save CPU results
        self._move_file('results/summary_report.txt',
                        self.output_dir / 'e1_cpu_report.txt')
        self._move_file('results/system_stats.csv',
                        self.output_dir / 'e1_cpu_stats.csv')
        
        self.results['E1'] = {
            'requested_gpu_device': 'cuda',
            'actual_gpu_device': gpu_device,
            'video_sources_used': gpu_sources,
            'gpu_report': str(self.output_dir / 'e1_gpu_report.txt'),
            'cpu_report': str(self.output_dir / 'e1_cpu_report.txt')
        }
        
        print("\n✓ E1 Complete: Results saved to experiments/results/e1_*")
    
    def run_experiment_e2_batch_size(self, duration=20):
        """
        E2: Batch Size vs Throughput
        Metric: FPS at batch sizes 1, 4, 8, 16, 32
        Concept: Throughput optimization, memory bandwidth
        """
        print("\n" + "="*70)
        print("EXPERIMENT E2: Batch Size vs Throughput")
        print("="*70)
        
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        for i, batch_size in enumerate(batch_sizes):
            print(f"\n[{i+1}/{len(batch_sizes)}] Testing batch size: {batch_size}")
            
            cmd, resolved_device, video_sources = self._build_main_command(4, 'cuda', batch_size, duration)
            print(f"    Device: {resolved_device.upper()} | Video sources: {len(video_sources)}")
            subprocess.run(cmd, cwd=self.project_root)
            
            # Save results
            self._move_file('results/summary_report.txt',
                            self.output_dir / f'e2_batch_{batch_size}_report.txt')
            self._move_file('results/system_stats.csv',
                            self.output_dir / f'e2_batch_{batch_size}_stats.csv')
            
            time.sleep(2)
        
        self.results['E2'] = {
            'batch_sizes': batch_sizes,
            'requested_device': 'cuda',
            'actual_device': self._resolve_device('cuda'),
            'reports': [str(self.output_dir / f'e2_batch_{b}_report.txt') 
                       for b in batch_sizes]
        }
        
        print("\n✓ E2 Complete: Results saved to experiments/results/e2_*")
    
    def run_experiment_e3_scaling(self, duration=30):
        """
        E3: Multi-Stream Scaling
        Metric: FPS & latency at 1, 2, 4 streams
        Concept: Amdahl's Law, parallel speedup limits
        """
        print("\n" + "="*70)
        print("EXPERIMENT E3: Multi-Stream Scaling (Amdahl's Law)")
        print("="*70)
        
        stream_counts = [1, 2, 4, 8]
        results = {}
        
        for i, num_streams in enumerate(stream_counts):
            print(f"\n[{i+1}/{len(stream_counts)}] Testing {num_streams} stream(s)")
            
            cmd, resolved_device, video_sources = self._build_main_command(num_streams, 'cuda', 4, duration)
            print(f"    Device: {resolved_device.upper()} | Video sources: {len(video_sources)}")
            subprocess.run(cmd, cwd=self.project_root)
            
            # Save results
            self._move_file('results/summary_report.txt',
                            self.output_dir / f'e3_streams_{num_streams}_report.txt')
            self._move_file('results/system_stats.csv',
                            self.output_dir / f'e3_streams_{num_streams}_stats.csv')
            
            time.sleep(2)
        
        self.results['E3'] = {
            'stream_counts': stream_counts,
            'requested_device': 'cuda',
            'actual_device': self._resolve_device('cuda'),
            'reports': [str(self.output_dir / f'e3_streams_{s}_report.txt') 
                       for s in stream_counts]
        }
        
        print("\n✓ E3 Complete: Results saved to experiments/results/e3_*")
        print("  Run Amdahl's Law analysis on these results")
    
    def run_experiment_e4_bottleneck(self, duration=30):
        """
        E4: Bottleneck Identification
        Metric: % time in loading vs inference vs postprocess
        Concept: System bottlenecks, pipeline balance
        """
        print("\n" + "="*70)
        print("EXPERIMENT E4: Bottleneck Identification")
        print("="*70)
        print("\nNote: This requires detailed_profiler.py integration")
        print("Run the main system and check detailed timing breakdown")
        
        cmd, resolved_device, video_sources = self._build_main_command(4, 'cuda', 4, duration)
        print(f"Using {resolved_device.upper()} on {len(video_sources)} video stream(s)")
        subprocess.run(cmd, cwd=self.project_root)
        
        self._move_file('results/summary_report.txt',
                        self.output_dir / 'e4_bottleneck_report.txt')
        
        self.results['E4'] = {
            'requested_device': 'cuda',
            'actual_device': resolved_device,
            'video_sources_used': video_sources,
            'report': str(self.output_dir / 'e4_bottleneck_report.txt')
        }
        
        print("\n✓ E4 Complete")
    
    def run_all_experiments(self):
        """Run all 4 experiments sequentially"""
        print("\n" + "="*70)
        print("RUNNING ALL EXPERIMENTS (E1-E4)")
        print("="*70)
        print("\nThis will take approximately 15-20 minutes...")
        print("Experiments use local video files and auto-fallback to CPU if CUDA is unavailable")
        print("Add video files before starting if you have not done that yet")
        
        input("\nPress Enter to start...")
        
        start_time = time.time()
        
        # Run all experiments
        self.run_experiment_e1_cpu_vs_gpu(duration=30)
        self.run_experiment_e2_batch_size(duration=20)
        self.run_experiment_e3_scaling(duration=30)
        self.run_experiment_e4_bottleneck(duration=30)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*70)
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Analyze results in experiments/results/")
        print("2. Run Amdahl's Law analysis on E3 data")
        print("3. Generate comparison plots")
        print("4. Write final report")
        print("="*70)
        
        # Save experiment metadata
        metadata = {
            'total_time': elapsed,
            'cuda_available': self.cuda_available,
            'video_sources_found': [str(path) for path in self._discover_video_files()],
            'experiments': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.output_dir / 'experiment_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report from all experiments"""
        print("\nGenerating comparison report...")
        
        # This would parse all the saved reports and create a unified comparison
        # Implementation depends on the exact format of summary_report.txt
        pass


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run proposal experiments')
    parser.add_argument('--experiment', type=str, choices=['e1', 'e2', 'e3', 'e4', 'all'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration for each run (seconds)')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.experiment == 'all':
        runner.run_all_experiments()
    elif args.experiment == 'e1':
        runner.run_experiment_e1_cpu_vs_gpu(args.duration)
    elif args.experiment == 'e2':
        runner.run_experiment_e2_batch_size(args.duration)
    elif args.experiment == 'e3':
        runner.run_experiment_e3_scaling(args.duration)
    elif args.experiment == 'e4':
        runner.run_experiment_e4_bottleneck(args.duration)


if __name__ == '__main__':
    main()
