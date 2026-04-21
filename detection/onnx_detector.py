# ONNX Runtime detector
"""
Drop-in replacement for YOLODetector that runs inference through
ONNX Runtime instead of PyTorch.

Why this exists:
    * ONNX Runtime is often noticeably faster than Ultralytics' default
      PyTorch path for YOLO-class models, especially with the CUDA or
      CoreML execution providers.
    * It lets us contrast inference engines (E6) while keeping the rest
      of the pipeline (multiprocessing capture, batching, profiling)
      identical.

Usage:
    python main.py --backend onnx --device cuda --batch-size 4 ...

The first run auto-exports the .pt model to .onnx using Ultralytics
(``YOLO.export(format='onnx')``) and caches it beside the .pt file.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger


class ONNXDetector:
    """ONNX Runtime YOLOv8 detector."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger('ONNXDetector')

        self.model_name = config.get('model_name', 'yolov8n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.batch_size = config.get('batch_size', 4)
        self.imgsz = int(config.get('imgsz', 640))

        # Resolved later from providers list
        self.device = 'cpu'

        self.total_detections = 0
        self.total_inference_time = 0.0
        self.inference_count = 0

        self._session = None
        self._input_name = None
        self._class_names = None
        self._dynamic_batch = False
        self._init_session(config.get('device', 'cuda'))

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _init_session(self, device_preference):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. Install via "
                "'pip install onnxruntime-gpu' (CUDA) or "
                "'pip install onnxruntime' (CPU) or "
                "'pip install onnxruntime-silicon' (macOS M-series)."
            ) from exc

        onnx_path = self._ensure_onnx_model()

        # Build provider list based on device preference + availability
        available = ort.get_available_providers()
        providers = self._select_providers(device_preference, available)
        self.logger.info(f"ONNX Runtime providers: {providers} "
                         f"(available: {available})")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name

        shape = self._session.get_inputs()[0].shape
        # Dynamic axis shows up as string or -1
        self._dynamic_batch = isinstance(shape[0], str) or shape[0] in (-1, None)

        # Resolve actual execution device (for reporting)
        prov = self._session.get_providers()[0]
        if 'CUDA' in prov:
            self.device = 'cuda'
        elif 'CoreML' in prov:
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Pull COCO class names from Ultralytics model metadata (fallback)
        self._class_names = self._load_class_names()

        self.logger.info(f"ONNX session ready (device={self.device}, "
                         f"dynamic_batch={self._dynamic_batch})")
        self._warmup()

    def _select_providers(self, device_preference, available):
        pref = (device_preference or 'auto').lower()
        ordered = []
        if pref in ('cuda', 'auto') and 'CUDAExecutionProvider' in available:
            ordered.append('CUDAExecutionProvider')
        if pref in ('mps', 'auto') and 'CoreMLExecutionProvider' in available:
            ordered.append('CoreMLExecutionProvider')
        # Always include CPU as a fallback
        if 'CPUExecutionProvider' not in ordered:
            ordered.append('CPUExecutionProvider')
        return ordered

    def _ensure_onnx_model(self):
        pt_path = Path(self.model_name)
        onnx_path = pt_path.with_suffix('.onnx')
        if onnx_path.exists():
            return onnx_path

        # Export with Ultralytics (supports dynamic batch + simplify)
        from ultralytics import YOLO
        self.logger.info(f"Exporting {pt_path} -> {onnx_path} (one-time)...")
        model = YOLO(str(pt_path))
        exported = model.export(
            format='onnx',
            imgsz=self.imgsz,
            dynamic=True,
            simplify=True,
            opset=12,
        )
        # Ultralytics may write alongside the .pt file; normalize path
        exported_path = Path(exported) if exported else onnx_path
        if exported_path != onnx_path and exported_path.exists():
            os.replace(exported_path, onnx_path)
        return onnx_path

    def _load_class_names(self):
        try:
            from config.config import COCO_CLASSES
            return list(COCO_CLASSES)
        except Exception:  # pragma: no cover
            return [str(i) for i in range(80)]

    def _warmup(self):
        dummy = np.random.rand(1, 3, self.imgsz, self.imgsz).astype(np.float32)
        t0 = time.time()
        _ = self._session.run(None, {self._input_name: dummy})
        self.logger.info(f"ONNX warmup: {time.time() - t0:.3f}s")

    # ------------------------------------------------------------------ #
    # Preprocessing (CPU; ONNX Runtime handles device placement)
    # ------------------------------------------------------------------ #
    def preprocess_batch(self, frames):
        """Return a float32 NCHW tensor, values in [0, 1]."""
        if not frames:
            return np.zeros((0, 3, self.imgsz, self.imgsz), dtype=np.float32)

        out = np.empty((len(frames), 3, self.imgsz, self.imgsz),
                       dtype=np.float32)
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_LINEAR)
            out[i] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return out

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _run(self, inp):
        return self._session.run(None, {self._input_name: inp})

    def detect_single(self, frame):
        inp = self.preprocess_batch([frame])
        t0 = time.time()
        outputs = self._run(inp)
        elapsed = time.time() - t0

        dets = self._decode(outputs, orig_frames=[frame])[0]
        self.total_inference_time += elapsed
        self.inference_count += 1
        self.total_detections += len(dets)

        return {
            'detections': dets,
            'inference_time': elapsed,
            'num_detections': len(dets),
        }

    def detect_batch(self, frames):
        if frames is None or len(frames) == 0:
            return []

        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            inp = frames.astype(np.float32, copy=False)
            orig_frames = None  # Already preprocessed upstream
        else:
            inp = self.preprocess_batch(list(frames))
            orig_frames = list(frames)

        # Static-batch ONNX exports only handle 1 frame at a time.
        all_dets = []
        if self._dynamic_batch:
            t0 = time.time()
            outputs = self._run(inp)
            elapsed = time.time() - t0
            all_dets = self._decode(outputs, orig_frames=orig_frames)
        else:
            elapsed = 0.0
            for i in range(inp.shape[0]):
                t0 = time.time()
                outputs = self._run(inp[i:i+1])
                elapsed += time.time() - t0
                frame_for_decode = ([orig_frames[i]]
                                    if orig_frames is not None else None)
                all_dets.extend(self._decode(outputs,
                                             orig_frames=frame_for_decode))

        n = inp.shape[0]
        self.total_inference_time += elapsed
        self.inference_count += n
        self.total_detections += sum(len(d) for d in all_dets)

        per_frame = elapsed / max(n, 1)
        results = []
        for dets in all_dets:
            results.append({
                'detections': dets,
                'num_detections': len(dets),
                'inference_time': per_frame,
            })
        return results

    # ------------------------------------------------------------------ #
    # Post-processing (NMS + bbox scaling)
    # ------------------------------------------------------------------ #
    def _decode(self, outputs, orig_frames=None):
        """
        YOLOv8 ONNX output shape: (B, 4 + num_classes, N) or (B, N, 4+nc).
        We normalize to (B, N, 4+nc), then apply confidence + NMS.
        """
        preds = outputs[0]
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            # (B, C, N) -> (B, N, C)
            preds = np.transpose(preds, (0, 2, 1))

        batch_dets = []
        for b in range(preds.shape[0]):
            frame_preds = preds[b]
            boxes_xywh = frame_preds[:, :4]
            class_scores = frame_preds[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[np.arange(len(class_ids)), class_ids]

            keep = confidences >= self.confidence_threshold
            if not np.any(keep):
                batch_dets.append([])
                continue

            boxes_xywh = boxes_xywh[keep]
            class_ids = class_ids[keep]
            confidences = confidences[keep]
            boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

            # OpenCV NMS expects (x, y, w, h)
            nms_boxes = boxes_xywh.tolist()
            nms_scores = confidences.tolist()
            idx = cv2.dnn.NMSBoxes(
                nms_boxes, nms_scores,
                self.confidence_threshold, self.iou_threshold,
            )
            if isinstance(idx, (list, tuple)):
                idx = np.asarray(idx).flatten()
            else:
                idx = np.asarray(idx).flatten() if len(idx) else np.array([], dtype=int)

            if len(idx) == 0:
                batch_dets.append([])
                continue

            # Rescale boxes from imgsz-space back to the original frame if
            # we know the source frame (dynamic-batch path with orig_frames).
            boxes_out = boxes_xyxy[idx]
            cls_out = class_ids[idx]
            conf_out = confidences[idx]

            if orig_frames is not None and b < len(orig_frames):
                h, w = orig_frames[b].shape[:2]
                scale_x = w / self.imgsz
                scale_y = h / self.imgsz
                boxes_out[:, [0, 2]] *= scale_x
                boxes_out[:, [1, 3]] *= scale_y

            dets = [{
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(c),
                'class_id': int(k),
                'class_name': self._class_names[int(k)]
                              if int(k) < len(self._class_names) else str(int(k)),
            } for (x1, y1, x2, y2), c, k in zip(boxes_out, conf_out, cls_out)]
            batch_dets.append(dets)

        return batch_dets

    @staticmethod
    def _xywh_to_xyxy(boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        out = np.empty_like(boxes)
        out[:, 0] = x - w / 2
        out[:, 1] = y - h / 2
        out[:, 2] = x + w / 2
        out[:, 3] = y + h / 2
        return out

    # ------------------------------------------------------------------ #
    # API parity with YOLODetector
    # ------------------------------------------------------------------ #
    def draw_detections(self, frame, detections):
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            np.random.seed(det['class_id'])
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(output, text, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output

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
            'backend': 'onnxruntime',
        }

    def get_gpu_memory_usage(self):
        # onnxruntime doesn't expose per-session GPU mem cheaply; return zeros.
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

    def print_stats(self):
        s = self.get_statistics()
        print("\n" + "=" * 60)
        print("ONNX DETECTION STATISTICS")
        print("=" * 60)
        print(f"Total Frames: {s['inference_count']}")
        print(f"Total Detections: {s['total_detections']}")
        print(f"Avg Inference Time: {s['avg_inference_time']*1000:.2f} ms")
        print(f"Avg FPS: {s['avg_fps']:.2f}")
        print(f"Device: {s['device']} (backend=onnxruntime)")
        print("=" * 60 + "\n")
