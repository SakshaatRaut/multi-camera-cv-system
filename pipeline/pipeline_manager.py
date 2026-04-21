# Pipeline Manager
"""
Pipeline manager for orchestrating the multi-camera CV system.

Implements a producer-consumer pipeline:
    cameras (mp.Process) -> shared queue -> batcher -> GPU inference -> stats

Stage timings are recorded via DetailedProfiler so Experiment E4
(bottleneck identification) reports real per-stage percentiles, not
just an aggregate FPS.
"""

import time
import queue
import threading
from collections import defaultdict

from profiling.detailed_profiler import DetailedProfiler
from utils.logger import get_logger


class PipelineManager:
    """Pipeline manager coordinates camera inputs with GPU detection."""

    def __init__(self, camera_manager, detector, profiler, config,
                 detailed_profiler=None):
        self.logger = get_logger('Pipeline')
        self.camera_manager = camera_manager
        self.detector = detector
        self.profiler = profiler
        self.config = config

        self.input_queue = camera_manager.get_frame_queue()

        self.batch_size = config.get('batch_size', 4)
        self.batch_timeout = config.get('batch_timeout', 0.05)
        self.use_batch_inference = config.get('use_batch_inference', True)

        # Detailed per-stage profiler (E4 + latency histograms).
        # Auto-enabled by default; can be disabled via config.
        if detailed_profiler is not None:
            self.detailed_profiler = detailed_profiler
        elif config.get('enable_detailed_profiling', True):
            self.detailed_profiler = DetailedProfiler()
        else:
            self.detailed_profiler = None

        self.running = False
        self.process_thread = None

        self.stats = defaultdict(lambda: {
            'frames_processed': 0,
            'total_detections': 0,
            'total_latency': 0.0,
            'latency_samples': [],   # bounded - see _record_latency
        })
        self.total_frames = 0
        self.start_time = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        self.logger.info("Starting pipeline manager")
        self.running = True
        self.start_time = time.time()

        self.camera_manager.start()

        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name='PipelineProcessor',
        )
        self.process_thread.start()

        self.logger.info(
            f"Pipeline started (batch_size={self.batch_size}, "
            f"batch_timeout={self.batch_timeout:.3f}s, "
            f"batched_inference={self.use_batch_inference})"
        )

    def stop(self):
        self.logger.info("Stopping pipeline")
        self.running = False

        # Stop cameras first so no new frames are enqueued.
        self.camera_manager.stop()

        # Drain any frames still sitting in the queue so the processing
        # thread's `queue.get(timeout=0.1)` returns promptly.
        try:
            while True:
                self.input_queue.get_nowait()
        except Exception:
            pass

        if self.process_thread:
            self.process_thread.join(timeout=2.0)

        self._print_statistics()

        # Persist stage-level report for E4 / post-hoc analysis
        if self.detailed_profiler is not None:
            try:
                self.detailed_profiler.print_bottleneck_report()
                output_dir = self.config.get('output_dir', 'results/')
                self.detailed_profiler.save_report(output_dir=output_dir)
            except Exception as exc:
                self.logger.warning(f"DetailedProfiler report failed: {exc}")

        self.logger.info("Pipeline stopped")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _process_loop(self):
        batch = []
        batch_metadata = []
        last_batch_time = time.time()
        last_report = time.time()

        dp = self.detailed_profiler  # local alias for hot path

        while self.running:
            try:
                # --- Stage 1: frame_loading (queue wait + capture lag) -----
                t_get = time.perf_counter()
                frame_data = self.input_queue.get(timeout=0.1)
                if dp is not None:
                    # Measures effective time we spent waiting on the queue,
                    # which captures capture/IO back-pressure.
                    dp.record_stage('frame_loading',
                                    time.perf_counter() - t_get)

                batch.append(frame_data['frame'])
                batch_metadata.append({
                    'camera_id': frame_data['camera_id'],
                    'timestamp': frame_data['timestamp'],
                    'frame_number': frame_data['frame_number'],
                })

                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size
                    or (batch and current_time - last_batch_time > self.batch_timeout)
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
            except (BrokenPipeError, EOFError, OSError) as e:
                # These typically come from the camera queue being torn down
                # during shutdown. Exit the loop cleanly instead of spamming.
                if self.running:
                    self.logger.warning(
                        f"Queue closed during operation ({type(e).__name__}: {e}); "
                        "exiting pipeline loop"
                    )
                break
            except Exception as e:
                # Only log if we're actually running; during stop() the queue
                # races with cleanup and generates benign transient errors.
                if self.running:
                    self.logger.error(
                        f"Error in process loop: "
                        f"{type(e).__name__}: {e}",
                        exc_info=True,
                    )
                else:
                    break

    # ------------------------------------------------------------------ #
    # Batch processing
    # ------------------------------------------------------------------ #
    def _process_batch(self, frames, metadata):
        if not frames:
            return

        dp = self.detailed_profiler

        # --- Stage 2: preprocessing ---------------------------------------
        # Ultralytics handles resize/normalize internally, but we still
        # measure any preprocessing we do here. If a GPUPreprocessor is
        # configured on the detector, it will be used via detect_batch.
        if dp is not None:
            t_pre = time.perf_counter()
            prepared = self._preprocess_batch(frames)
            dp.record_stage('preprocessing', time.perf_counter() - t_pre)
        else:
            prepared = self._preprocess_batch(frames)

        # --- Stage 3: inference -------------------------------------------
        t_inf = time.perf_counter()
        if self.use_batch_inference and len(prepared) > 1:
            results = self.detector.detect_batch(prepared)
        else:
            results = [self.detector.detect_single(f) for f in prepared]
        inf_elapsed = time.perf_counter() - t_inf

        # Attribute inference time per-frame for fair per-stage histogram
        per_frame_inf = inf_elapsed / max(len(prepared), 1)
        if dp is not None:
            for _ in prepared:
                dp.record_stage('inference', per_frame_inf)

        # --- Stage 4: postprocessing --------------------------------------
        # Parse detections + update stats (no rendering - keeps benchmark honest)
        now = time.time()
        for result, meta in zip(results, metadata):
            if dp is not None:
                t_post = time.perf_counter()
                self._update_stats(result, meta, now)
                dp.record_stage('postprocessing',
                                time.perf_counter() - t_post)
            else:
                self._update_stats(result, meta, now)

        # --- Stage 5: output ----------------------------------------------
        # The benchmark pipeline doesn't display frames, so 'output' here
        # represents any aggregate write-out work (profiler push + counters).
        if dp is not None:
            t_out = time.perf_counter()
            self._push_to_profiler()
            dp.record_stage('output', time.perf_counter() - t_out)
            # Mark frames complete (one per metadata entry)
            for _ in metadata:
                dp.mark_frame()
        else:
            self._push_to_profiler()

    def _preprocess_batch(self, frames):
        """Hook point for optional GPU preprocessing.

        If the detector exposes a ``preprocess_batch`` method (e.g. the
        GPU-accelerated detector does), we delegate to it so the CPU does
        less work. Otherwise we pass frames through untouched - Ultralytics
        will resize them internally.
        """
        preprocess = getattr(self.detector, 'preprocess_batch', None)
        if callable(preprocess):
            try:
                return preprocess(frames)
            except Exception as exc:  # pragma: no cover - best-effort
                self.logger.warning(f"detector.preprocess_batch failed, "
                                    f"falling back to CPU path: {exc}")
        return frames

    def _update_stats(self, result, meta, now):
        camera_id = meta['camera_id']
        capture_time = meta['timestamp']
        latency = now - capture_time

        cam_stats = self.stats[camera_id]
        cam_stats['frames_processed'] += 1
        cam_stats['total_detections'] += result.get('num_detections', 0)
        cam_stats['total_latency'] += latency

        # Bounded latency-sample buffer for per-camera percentiles
        samples = cam_stats['latency_samples']
        samples.append(latency)
        if len(samples) > 5000:
            del samples[: len(samples) // 2]

        self.total_frames += 1

    def _push_to_profiler(self):
        if not self.profiler or not self.start_time:
            return
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return
        for camera_id, cam_stats in self.stats.items():
            frames = cam_stats['frames_processed']
            if frames == 0:
                continue
            fps = frames / elapsed
            avg_latency = cam_stats['total_latency'] / frames
            self.profiler.update_camera_stats(camera_id, fps, avg_latency)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _report_progress(self):
        if not self.start_time:
            return

        elapsed = time.time() - self.start_time
        overall_fps = self.total_frames / elapsed if elapsed > 0 else 0

        # qsize() raises NotImplementedError on macOS (Darwin has no
        # sem_getvalue), so fall back to "?" there.
        try:
            q_size = str(self.input_queue.qsize())
        except (NotImplementedError, AttributeError):
            q_size = "?"

        self.logger.info(
            f"Pipeline: {self.total_frames} frames, {overall_fps:.2f} FPS, "
            f"Queue: {q_size}"
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

        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None

        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("PIPELINE STATISTICS")
        print("=" * 60)
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
            print(f"    Avg Latency: {avg_latency*1000:.1f}ms")

            samples = cam_stats.get('latency_samples') or []
            if np is not None and samples:
                p50 = float(np.percentile(samples, 50)) * 1000.0
                p95 = float(np.percentile(samples, 95)) * 1000.0
                p99 = float(np.percentile(samples, 99)) * 1000.0
                print(f"    Latency p50/p95/p99: "
                      f"{p50:.1f} / {p95:.1f} / {p99:.1f} ms")

            print(f"    Detections: {cam_stats['total_detections']}")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def get_overall_fps(self):
        if not self.start_time:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0

    def get_summary(self):
        """Compact summary useful for experiment runners."""
        elapsed = (time.time() - self.start_time) if self.start_time else 0.0
        per_cam = {}
        total_latency = 0.0
        total_frames = 0
        for cid, s in self.stats.items():
            frames = s['frames_processed']
            per_cam[cid] = {
                'frames': frames,
                'fps': (frames / elapsed) if elapsed > 0 else 0.0,
                'avg_latency': (s['total_latency'] / frames) if frames else 0.0,
                'detections': s['total_detections'],
            }
            total_latency += s['total_latency']
            total_frames += frames

        return {
            'runtime_s': elapsed,
            'total_frames': total_frames,
            'overall_fps': (total_frames / elapsed) if elapsed > 0 else 0.0,
            'avg_latency_ms': ((total_latency / total_frames) * 1000.0)
                              if total_frames else 0.0,
            'per_camera': per_cam,
        }
