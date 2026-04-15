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

            # Detector
            self.logger.info("Initializing detector...")
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
            
            self.logger.info("Visualizations complete")
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")


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
        choices=['cuda', 'cpu'],
        help='Processing device'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = SYSTEM_CONFIG.copy()
    config['model_name'] = args.model
    config['device'] = args.device
    config['batch_size'] = args.batch_size
    
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
    print(f"Duration: {args.duration if args.duration else 'Continuous'}s")
    print("="*70)
    print("\nKey Features:")
    print("  ✓ Multiprocessing (one process per camera)")
    print("  ✓ GPU-accelerated detection")
    print("  ✓ Real-time performance profiling")
    print("  ✓ Automatic visualization generation")
    print("="*70 + "\n")
    
    # Run system
    system = MultiCameraSystem(config)
    system.start(duration=args.duration)


if __name__ == '__main__':
    main()
