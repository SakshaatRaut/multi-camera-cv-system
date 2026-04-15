# Pipeline Manager
"""
Pipeline manager for orchestrating the multi-camera CV system
Implements producer-consumer pattern with frame buffering
"""

import time
import queue
import threading
from collections import defaultdict
from utils.logger import get_logger


class PipelineManager:
    """Pipeline manager coordinates camera inputs with GPU detection"""
    
    def __init__(self, camera_manager, detector, profiler, config):
        self.logger = get_logger('Pipeline')
        self.camera_manager = camera_manager
        self.detector = detector
        self.profiler = profiler
        self.config = config
        
        self.input_queue = camera_manager.get_frame_queue()
        
        self.batch_size = config.get('batch_size', 4)
        self.batch_timeout = config.get('batch_timeout', 0.05)
        
        self.running = False
        self.process_thread = None
        
        self.stats = defaultdict(lambda: {
            'frames_processed': 0,
            'total_detections': 0,
            'total_latency': 0.0
        })
        self.total_frames = 0
        self.start_time = None
    
    def start(self):
        self.logger.info("Starting pipeline manager")
        self.running = True
        self.start_time = time.time()
        
        self.camera_manager.start()
        
        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name='PipelineProcessor'
        )
        self.process_thread.start()
        
        self.logger.info("Pipeline started")
    
    def stop(self):
        self.logger.info("Stopping pipeline")
        self.running = False
        
        self.camera_manager.stop()
        
        if self.process_thread:
            self.process_thread.join(timeout=5.0)
        
        self._print_statistics()
        self.logger.info("Pipeline stopped")
    
    def _process_loop(self):
        batch = []
        batch_metadata = []
        last_batch_time = time.time()
        last_report = time.time()
        
        while self.running:
            try:
                frame_data = self.input_queue.get(timeout=0.1)
                
                batch.append(frame_data['frame'])
                batch_metadata.append({
                    'camera_id': frame_data['camera_id'],
                    'timestamp': frame_data['timestamp'],
                    'frame_number': frame_data['frame_number']
                })
                
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and current_time - last_batch_time > self.batch_timeout)
                )
                
                if should_process:
                    self._process_batch(batch, batch_metadata)
                    batch = []
                    batch_metadata = []
                    last_batch_time = current_time
                
                # Report every 5 seconds
                if current_time - last_report >= 5.0:
                    self._report_progress()
                    last_report = current_time
                    
            except queue.Empty:
                if batch:
                    self._process_batch(batch, batch_metadata)
                    batch = []
                    batch_metadata = []
                    last_batch_time = time.time()
                continue
            except Exception as e:
                self.logger.error(f"Error in process loop: {e}")
    
    def _process_batch(self, frames, metadata):
        if not frames:
            return
        
        for i, (frame, meta) in enumerate(zip(frames, metadata)):
            camera_id = meta['camera_id']
            capture_time = meta['timestamp']
            
            result = self.detector.detect_single(frame)
            
            latency = time.time() - capture_time
            
            cam_stats = self.stats[camera_id]
            cam_stats['frames_processed'] += 1
            cam_stats['total_detections'] += result['num_detections']
            cam_stats['total_latency'] += latency
            
            self.total_frames += 1
            
            # Update profiler
            if self.profiler:
                elapsed = time.time() - self.start_time
                fps = cam_stats['frames_processed'] / elapsed if elapsed > 0 else 0
                avg_latency = cam_stats['total_latency'] / cam_stats['frames_processed']
                self.profiler.update_camera_stats(camera_id, fps, avg_latency)
    
    def _report_progress(self):
        if not self.start_time:
            return
        
        elapsed = time.time() - self.start_time
        overall_fps = self.total_frames / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Pipeline: {self.total_frames} frames, {overall_fps:.2f} FPS, "
            f"Queue: {self.input_queue.qsize()}"
        )
        
        for camera_id in sorted(self.stats.keys()):
            cam_stats = self.stats[camera_id]
            frames = cam_stats['frames_processed']
            fps = frames / elapsed if elapsed > 0 else 0
            avg_latency = cam_stats['total_latency'] / frames if frames > 0 else 0
            
            self.logger.info(
                f"  Camera {camera_id}: {frames} frames, {fps:.2f} FPS, "
                f"{avg_latency*1000:.1f}ms latency, "
                f"{cam_stats['total_detections']} detections"
            )
    
    def _print_statistics(self):
        if not self.start_time:
            return
        
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("PIPELINE STATISTICS")
        print("="*60)
        print(f"Runtime: {elapsed:.2f}s")
        print(f"Total Frames: {self.total_frames}")
        print(f"Overall FPS: {self.total_frames/elapsed:.2f}")
        
        print("\nPer-Camera:")
        for camera_id in sorted(self.stats.keys()):
            cam_stats = self.stats[camera_id]
            frames = cam_stats['frames_processed']
            fps = frames / elapsed if elapsed > 0 else 0
            avg_latency = cam_stats['total_latency'] / frames if frames > 0 else 0
            
            print(f"  Camera {camera_id}:")
            print(f"    Frames: {frames}")
            print(f"    FPS: {fps:.2f}")
            print(f"    Latency: {avg_latency*1000:.1f}ms")
            print(f"    Detections: {cam_stats['total_detections']}")
        
        print("="*60 + "\n")