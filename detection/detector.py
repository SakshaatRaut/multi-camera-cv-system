# GPU-Accelerated Object Detector
"""
GPU-accelerated object detection using YOLOv8.

Supports:
    * Device auto-selection: CUDA (NVIDIA), MPS (Apple Silicon), CPU.
    * FP16 mixed-precision inference on CUDA (`--fp16`).
    * Optional GPU-side preprocessing (`--gpu-preprocess`):
        resize + normalize + NCHW on the target device, with pinned-memory
        + non_blocking=True host->device copies.
    * Optional overlap via a dedicated CUDA stream for the pre+H2D stage.

All of these are switched on/off via config and degrade gracefully when the
required hardware is not available.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.logger import get_logger


class YOLODetector:
    """YOLOv8 detector with CUDA / MPS / CPU support."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger('YOLODetector')

        self.model_name = config.get('model_name', 'yolov8n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.batch_size = config.get('batch_size', 4)

        self.device = self._setup_device(config.get('device', 'cuda'))
        self.use_fp16 = bool(config.get('use_fp16', False)) and self.device == 'cuda'
        self.use_gpu_preprocess = (
            bool(config.get('use_gpu_preprocess', False))
            and self.device in ('cuda', 'mps')
        )
        self.imgsz = int(config.get('imgsz', 640))

        # CUDA stream for async H2D overlap (only useful with FP16+GPU preproc)
        self._h2d_stream = (
            torch.cuda.Stream() if self.device == 'cuda' else None
        )

        self.model = None
        self._load_model()

        self.total_detections = 0
        self.total_inference_time = 0.0
        self.inference_count = 0

    # ------------------------------------------------------------------ #
    # Device / model setup
    # ------------------------------------------------------------------ #
    def _setup_device(self, preference):
        pref = (preference or 'auto').lower()

        if pref == 'auto':
            if torch.cuda.is_available():
                pref = 'cuda'
            elif getattr(torch.backends, 'mps', None) and \
                    torch.backends.mps.is_available():
                pref = 'mps'
            else:
                pref = 'cpu'

        if pref == 'cuda':
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.logger.info(f"Using CUDA GPU: {name} ({mem_gb:.2f} GB)")
                return 'cuda'
            self.logger.warning("CUDA requested but unavailable; falling back to CPU")
            return 'cpu'

        if pref == 'mps':
            if getattr(torch.backends, 'mps', None) and \
                    torch.backends.mps.is_available():
                self.logger.info("Using Apple MPS (Metal Performance Shaders)")
                return 'mps'
            self.logger.warning("MPS requested but unavailable; falling back to CPU")
            return 'cpu'

        self.logger.info("Using CPU for inference")
        return 'cpu'

    def _load_model(self):
        self.logger.info(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)
        try:
            self.model.to(self.device)
        except Exception as exc:
            # MPS can be finicky with some layers on older torch; fall back.
            if self.device == 'mps':
                self.logger.warning(f"MPS move failed ({exc}); using CPU")
                self.device = 'cpu'
                self.model.to('cpu')
                self.use_fp16 = False
                self.use_gpu_preprocess = False
            else:
                raise

        if self.use_fp16:
            try:
                self.model.model.half()
                self.logger.info("FP16 mixed precision enabled (model.half())")
            except Exception as exc:  # pragma: no cover
                self.logger.warning(f"FP16 not supported here: {exc}")
                self.use_fp16 = False

        self.logger.info(
            f"Model loaded on {self.device} "
            f"(fp16={self.use_fp16}, gpu_preproc={self.use_gpu_preprocess})"
        )
        self._warmup()

    def _warmup(self):
        self.logger.info("Warming up model...")
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3),
                                  dtype=np.uint8)
        t0 = time.time()
        _ =  self._run_model(dummy)
        self.logger.info(f"Warmup complete in {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess_batch(self, frames):
        """
        Optional GPU-side preprocessing hook used by PipelineManager.

        If enabled, returns a torch tensor of shape (N, 3, imgsz, imgsz)
        already on the target device. Otherwise returns the input list
        untouched (Ultralytics will preprocess on CPU).

        Using pinned-memory host staging + non_blocking=True H2D copies
        overlaps the transfer with inference on the default CUDA stream.
        """
        if not self.use_gpu_preprocess or not frames:
            return frames

        # Stage 1: resize + BGR->RGB + uint8->float + NHWC on CPU (vectorized)
        size = self.imgsz
        processed = np.empty((len(frames), size, size, 3), dtype=np.uint8)
        for i, f in enumerate(frames):
            processed[i] = cv2.resize(
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size),
                interpolation=cv2.INTER_LINEAR,
            )

        host = torch.from_numpy(processed)  # (N, H, W, C) uint8
        if self.device == 'cuda':
            host = host.pin_memory()

        # Stage 2: async copy + layout transform on the target device
        if self.device == 'cuda' and self._h2d_stream is not None:
            with torch.cuda.stream(self._h2d_stream):
                dev = host.to('cuda', non_blocking=True)
                dev = dev.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
                dev = dev.to(torch.float16 if self.use_fp16
                             else torch.float32).div_(255.0)
            # Make the main stream wait for our copy+cast to finish.
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
        else:
            dev = host.to(self.device)
            dev = dev.permute(0, 3, 1, 2).contiguous()
            dev = dev.to(torch.float32).div_(255.0)

        return dev

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _run_model(self, inputs):
        """Run the underlying YOLO model with FP16 autocast on CUDA."""
        if self.use_fp16 and self.device == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return self.model(
                    inputs,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device=self.device,
                )
        return self.model(
            inputs,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

    def detect_single(self, frame):
        t0 = time.time()
        results = self._run_model(frame)[0]
        elapsed = time.time() - t0

        detections = self._extract_detections(results)
        self.total_inference_time += elapsed
        self.inference_count += 1
        self.total_detections += len(detections)

        return {
            'detections': detections,
            'inference_time': elapsed,
            'num_detections': len(detections),
        }

    def detect_batch(self, frames):
        if frames is None:
            return []
        # frames can be: list of np.ndarray, or a torch tensor (gpu-preprocessed)
        n = len(frames) if not torch.is_tensor(frames) else frames.shape[0]
        if n == 0:
            return []

        t0 = time.time()
        results = self._run_model(frames)
        elapsed = time.time() - t0

        batch_results = []
        total_dets = 0
        for result in results:
            dets = self._extract_detections(result)
            batch_results.append({
                'detections': dets,
                'num_detections': len(dets),
            })
            total_dets += len(dets)

        self.total_inference_time += elapsed
        self.inference_count += n
        self.total_detections += total_dets

        per_frame = elapsed / n
        for r in batch_results:
            r['inference_time'] = per_frame

        return batch_results

    # ------------------------------------------------------------------ #
    # Results parsing + rendering
    # ------------------------------------------------------------------ #
    def _extract_detections(self, result):
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])],
                })
        return detections

    def draw_detections(self, frame, detections):
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = self._get_color(det['class_id'])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{det['class_name']}: {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(output, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return output

    @staticmethod
    def _get_color(class_id):
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def get_statistics(self):
        avg_time = (self.total_inference_time / self.inference_count
                    if self.inference_count > 0 else 0.0)
        return {
            'total_detections': self.total_detections,
            'inference_count': self.inference_count,
            'avg_inference_time': avg_time,
            'avg_fps': (1.0 / avg_time) if avg_time > 0 else 0.0,
            'device': self.device,
            'model': self.model_name,
            'fp16': self.use_fp16,
            'gpu_preprocess': self.use_gpu_preprocess,
        }

    def get_gpu_memory_usage(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

    def print_stats(self):
        s = self.get_statistics()
        print("\n" + "=" * 60)
        print("DETECTION STATISTICS")
        print("=" * 60)
        print(f"Total Frames: {s['inference_count']}")
        print(f"Total Detections: {s['total_detections']}")
        print(f"Avg Inference Time: {s['avg_inference_time']*1000:.2f} ms")
        print(f"Avg FPS: {s['avg_fps']:.2f}")
        print(f"Device: {s['device']} | FP16: {s['fp16']} | "
              f"GPU Preproc: {s['gpu_preprocess']}")
        print("=" * 60 + "\n")
