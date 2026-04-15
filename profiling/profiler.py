# System Profiler
"""
System profiler for comprehensive performance monitoring
Tracks CPU, GPU, RAM, and pipeline statistics
"""

import time
import psutil
import threading
import pandas as pd
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from utils.logger import get_logger


class Profiler:
    """Monitors system performance in real-time"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger('Profiler')
        
        self.profile_interval = config.get('profile_interval', 1.0)
        self.save_stats = config.get('save_stats', True)
        self.stats_file = config.get('stats_file', 'results/system_stats.csv')
        
        Path(self.stats_file).parent.mkdir(exist_ok=True)
        
        self.stats_history = []
        self.camera_stats = {}
        
        # System info
        self.num_cores = psutil.cpu_count(logical=False)
        self.total_ram = psutil.virtual_memory().total / (1024**3)
        
        # GPU info
        self.has_gpu = HAS_TORCH and torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.running = False
        self.profile_thread = None
        self.start_time = None
        
        self.logger.info(f"Profiler initialized - CPU: {self.num_cores} cores, RAM: {self.total_ram:.2f}GB")
        if self.has_gpu:
            self.logger.info(f"GPU: {self.gpu_name}, {self.gpu_total_memory:.2f}GB")
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.profile_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.profile_thread.start()
        self.logger.info("Profiler started")
    
    def stop(self):
        self.running = False
        if self.profile_thread:
            self.profile_thread.join(timeout=5.0)
        
        if self.save_stats:
            self._save_statistics()
        
        self.logger.info("Profiler stopped")
    
    def _profile_loop(self):
        while self.running:
            stats = self._collect_statistics()
            self.stats_history.append(stats)
            time.sleep(self.profile_interval)
    
    def _collect_statistics(self):
        timestamp = time.time()
        elapsed = timestamp - self.start_time if self.start_time else 0
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_percent = ram.percent
        
        stats = {
            'timestamp': timestamp,
            'elapsed': elapsed,
            'cpu_percent': cpu_percent,
            'cpu_cores': str(cpu_per_core),
            'ram_used_gb': ram_used_gb,
            'ram_percent': ram_percent,
        }
        
        if self.has_gpu:
            try:
                stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
                stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
            except:
                stats['gpu_memory_allocated'] = 0
                stats['gpu_memory_reserved'] = 0
        
        for camera_id, cam_stats in self.camera_stats.items():
            stats[f'camera_{camera_id}_fps'] = cam_stats.get('fps', 0)
            stats[f'camera_{camera_id}_latency'] = cam_stats.get('latency', 0)
        
        return stats
    
    def update_camera_stats(self, camera_id, fps, latency):
        self.camera_stats[camera_id] = {'fps': fps, 'latency': latency}
    
    def _save_statistics(self):
        if not self.stats_history:
            return
        
        try:
            df = pd.DataFrame(self.stats_history)
            df.to_csv(self.stats_file, index=False)
            self.logger.info(f"Statistics saved to {self.stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
    
    def get_summary(self):
        if not self.stats_history:
            return {}
        
        df = pd.DataFrame(self.stats_history)
        
        summary = {
            'duration': df['elapsed'].max(),
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'max_cpu_percent': df['cpu_percent'].max(),
            'avg_ram_gb': df['ram_used_gb'].mean(),
            'max_ram_gb': df['ram_used_gb'].max(),
        }
        
        if self.has_gpu and 'gpu_memory_allocated' in df.columns:
            summary['avg_gpu_memory_gb'] = df['gpu_memory_allocated'].mean()
            summary['max_gpu_memory_gb'] = df['gpu_memory_allocated'].max()
        
        for camera_id in self.camera_stats.keys():
            fps_col = f'camera_{camera_id}_fps'
            latency_col = f'camera_{camera_id}_latency'
            
            if fps_col in df.columns:
                summary[f'camera_{camera_id}_avg_fps'] = df[fps_col].mean()
                summary[f'camera_{camera_id}_avg_latency'] = df[latency_col].mean()
        
        return summary
    
    def print_summary(self):
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if summary:
            print(f"\nDuration: {summary.get('duration', 0):.2f}s")
            
            print("\nCPU:")
            print(f"  Average: {summary.get('avg_cpu_percent', 0):.1f}%")
            print(f"  Peak: {summary.get('max_cpu_percent', 0):.1f}%")
            
            print("\nRAM:")
            print(f"  Average: {summary.get('avg_ram_gb', 0):.2f}GB")
            print(f"  Peak: {summary.get('max_ram_gb', 0):.2f}GB")
            
            if self.has_gpu:
                print("\nGPU:")
                print(f"  Average: {summary.get('avg_gpu_memory_gb', 0):.2f}GB")
                print(f"  Peak: {summary.get('max_gpu_memory_gb', 0):.2f}GB")
            
            print("\nPer-Camera:")
            for camera_id in self.camera_stats.keys():
                fps = summary.get(f'camera_{camera_id}_avg_fps', 0)
                latency = summary.get(f'camera_{camera_id}_avg_latency', 0)
                print(f"  Camera {camera_id}: {fps:.2f} FPS, {latency*1000:.1f}ms")
        
        print("="*60 + "\n")