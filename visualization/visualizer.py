# Performance Visualizer
"""
Performance visualizer for creating plots and reports
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger


class Visualizer:
    """Creates performance visualizations from profiling data"""
    
    def __init__(self, stats_file, output_dir):
        self.logger = get_logger('Visualizer')
        self.stats_file = stats_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        
        self.df = None
        self._load_data()
    
    def _load_data(self):
        try:
            if Path(self.stats_file).exists():
                self.df = pd.read_csv(self.stats_file)
                self.logger.info(f"Loaded {len(self.df)} data points")
            else:
                self.logger.warning(f"Stats file not found: {self.stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to load stats: {e}")
    
    def generate_all_plots(self):
        if self.df is None or len(self.df) == 0:
            self.logger.warning("No data to plot")
            return
        
        self.logger.info("Generating visualizations...")
        
        try:
            self.plot_fps_comparison()
            self.plot_cpu_usage()
            self.plot_memory_usage()
            
            if 'gpu_memory_allocated' in self.df.columns:
                self.plot_gpu_usage()
            
            self.plot_system_overview()
            
            self.logger.info("All plots generated")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def plot_fps_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fps_cols = [col for col in self.df.columns if col.endswith('_fps')]
        
        if not fps_cols:
            return
        
        for col in fps_cols:
            camera_id = col.split('_')[1]
            ax.plot(self.df['elapsed'], self.df[col], 
                   label=f'Camera {camera_id}', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('FPS', fontsize=12)
        ax.set_title('Frame Rate Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        output_file = self.output_dir / 'fps_comparison.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def plot_cpu_usage(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.df['elapsed'], self.df['cpu_percent'], 
               color='#2E86AB', linewidth=2)
        ax.fill_between(self.df['elapsed'], self.df['cpu_percent'], 
                        alpha=0.3, color='#2E86AB')
        
        avg_cpu = self.df['cpu_percent'].mean()
        ax.axhline(y=avg_cpu, color='red', linestyle='--', 
                  label=f'Average: {avg_cpu:.1f}%', linewidth=2)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('CPU Usage (%)', fontsize=12)
        ax.set_title('CPU Utilization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        output_file = self.output_dir / 'cpu_usage.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def plot_memory_usage(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.df['elapsed'], self.df['ram_used_gb'], 
               color='#F18F01', linewidth=2, label='Used RAM')
        ax.fill_between(self.df['elapsed'], self.df['ram_used_gb'], 
                        alpha=0.3, color='#F18F01')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('RAM (GB)', fontsize=12)
        ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_file = self.output_dir / 'memory_usage.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def plot_gpu_usage(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.df['elapsed'], self.df['gpu_memory_allocated'], 
               color='#76B041', linewidth=2, label='Allocated')
        ax.fill_between(self.df['elapsed'], self.df['gpu_memory_allocated'], 
                        alpha=0.3, color='#76B041')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('GPU Memory (GB)', fontsize=12)
        ax.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_file = self.output_dir / 'gpu_usage.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def plot_system_overview(self):
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # FPS
        ax1 = fig.add_subplot(gs[0, :])
        fps_cols = [col for col in self.df.columns if col.endswith('_fps')]
        for col in fps_cols:
            camera_id = col.split('_')[1]
            ax1.plot(self.df['elapsed'], self.df[col], label=f'Camera {camera_id}')
        ax1.set_ylabel('FPS')
        ax1.set_title('Frame Rate', fontweight='bold')
        ax1.legend(ncol=4)
        ax1.grid(True, alpha=0.3)
        
        # CPU
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.df['elapsed'], self.df['cpu_percent'], color='#2E86AB', linewidth=2)
        ax2.fill_between(self.df['elapsed'], self.df['cpu_percent'], alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('CPU (%)')
        ax2.set_title('CPU Usage', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # RAM
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.df['elapsed'], self.df['ram_used_gb'], color='#F18F01', linewidth=2)
        ax3.fill_between(self.df['elapsed'], self.df['ram_used_gb'], alpha=0.3, color='#F18F01')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('RAM (GB)')
        ax3.set_title('Memory Usage', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle('System Performance Overview', fontsize=16, fontweight='bold')
        
        output_file = self.output_dir / 'system_overview.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved: {output_file}")
    
    def generate_summary_report(self, profiler_summary):
        report_file = self.output_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PERFORMANCE REPORT\n")
            f.write("="*60 + "\n\n")
            
            if profiler_summary:
                f.write(f"Duration: {profiler_summary.get('duration', 0):.2f}s\n\n")
                
                f.write("CPU:\n")
                f.write(f"  Average: {profiler_summary.get('avg_cpu_percent', 0):.1f}%\n")
                f.write(f"  Peak: {profiler_summary.get('max_cpu_percent', 0):.1f}%\n\n")
                
                f.write("RAM:\n")
                f.write(f"  Average: {profiler_summary.get('avg_ram_gb', 0):.2f}GB\n")
                f.write(f"  Peak: {profiler_summary.get('max_ram_gb', 0):.2f}GB\n\n")
                
                if 'avg_gpu_memory_gb' in profiler_summary:
                    f.write("GPU:\n")
                    f.write(f"  Average: {profiler_summary['avg_gpu_memory_gb']:.2f}GB\n")
                    f.write(f"  Peak: {profiler_summary['max_gpu_memory_gb']:.2f}GB\n\n")
                
                camera_keys = [k for k in profiler_summary.keys() if k.endswith('_avg_fps')]
                if camera_keys:
                    f.write("PER-CAMERA:\n")
                    for key in camera_keys:
                        cam_id = key.split('_')[1]
                        fps = profiler_summary.get(f'camera_{cam_id}_avg_fps', 0)
                        latency = profiler_summary.get(f'camera_{cam_id}_avg_latency', 0)
                        f.write(f"  Camera {cam_id}: {fps:.2f} FPS, {latency*1000:.1f}ms\n")
            
            f.write("\n" + "="*60 + "\n")
        
        self.logger.info(f"Saved: {report_file}")