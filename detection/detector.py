# GPU-Accelerated Object Detector
"""
GPU-accelerated object detection using YOLOv8
Demonstrates GPU utilization, batch processing, and memory management
"""

import torch
import numpy as np
import time
import cv2
from ultralytics import YOLO
from utils.logger import get_logger


class YOLODetector:
    """YOLOv8 detector with GPU acceleration"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger('YOLODetector')
        
        self.model_name = config.get('model_name', 'yolov8n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.batch_size = config.get('batch_size', 4)
        
        self.device = self._setup_device(config.get('device', 'cuda'))
        
        self.model = None
        self._load_model()
        
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.inference_count = 0
    
    def _setup_device(self, device_preference):
        if device_preference == 'cuda' and torch.cuda.is_available():
            device = 'cuda'
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU Memory: {gpu_mem:.2f} GB")
        else:
            device = 'cpu'
            self.logger.info("Using CPU for inference")
            if device_preference == 'cuda':
                self.logger.warning("CUDA not available, using CPU")
        
        return device
    
    def _load_model(self):
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            if self.device == 'cuda':
                self.model.to('cuda')
            
            self.logger.info(f"Model loaded on {self.device}")
            self._warmup()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _warmup(self):
        self.logger.info("Warming up model...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        start = time.time()
        _ = self.model(dummy_img, verbose=False, device=self.device)
        self.logger.info(f"Warmup completed in {time.time() - start:.3f}s")
    
    def detect_single(self, frame):
        start_time = time.time()
        
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )[0]
        
        inference_time = time.time() - start_time
        detections = self._extract_detections(results)
        
        self.total_inference_time += inference_time
        self.inference_count += 1
        self.total_detections += len(detections)
        
        return {
            'detections': detections,
            'inference_time': inference_time,
            'num_detections': len(detections)
        }
    
    def detect_batch(self, frames):
        if not frames:
            return []
        
        start_time = time.time()
        
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
            stream=False
        )
        
        inference_time = time.time() - start_time
        
        batch_results = []
        total_dets = 0
        
        for result in results:
            detections = self._extract_detections(result)
            batch_results.append({
                'detections': detections,
                'num_detections': len(detections)
            })
            total_dets += len(detections)
        
        self.total_inference_time += inference_time
        self.inference_count += len(frames)
        self.total_detections += total_dets
        
        avg_time = inference_time / len(frames)
        for result in batch_results:
            result['inference_time'] = avg_time
        
        return batch_results
    
    def _extract_detections(self, result):
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            label = det['class_name']
            
            color = self._get_color(det['class_id'])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(output, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output
    
    def _get_color(self, class_id):
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def get_statistics(self):
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'total_detections': self.total_detections,
            'inference_count': self.inference_count,
            'avg_inference_time': avg_time,
            'avg_fps': avg_fps,
            'device': self.device,
            'model': self.model_name
        }
    
    def get_gpu_memory_usage(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    def print_stats(self):
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("DETECTION STATISTICS")
        print("="*60)
        print(f"Total Frames: {stats['inference_count']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Avg Inference Time: {stats['avg_inference_time']*1000:.2f} ms")
        print(f"Avg FPS: {stats['avg_fps']:.2f}")
        print(f"Device: {stats['device']}")
        print("="*60 + "\n")