# Camera Stream with Multiprocessing
"""
Camera stream handler with multiprocessing
Demonstrates multi-core CPU utilization for parallel video ingestion
"""

import cv2
import time
import multiprocessing as mp
from queue import Full
from pathlib import Path
from utils.logger import get_logger


class CameraStream:
    """Individual camera stream running in separate process"""

    def __init__(self, camera_id, source, frame_queue, config, stop_event=None):
        self.camera_id = camera_id
        self.source = source
        self.frame_queue = frame_queue
        self.config = config
        self.logger = get_logger(f'Camera_{camera_id}')

        # CRITICAL: use an mp.Event so the child process actually sees when
        # the parent asks us to stop. A plain bool attribute would not work
        # because multiprocessing copies the object into the child.
        self.stop_event = stop_event if stop_event is not None else mp.Event()
        self.frames_processed = 0
        self.frames_dropped = 0

    def run(self):
        """Main loop for camera processing (runs in separate process)"""
        self.logger.info(f"Starting camera {self.camera_id} with source: {self.source}")
        
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {self.source}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / fps
        
        self.logger.info(f"Camera {self.camera_id} opened (FPS: {fps})")
        
        frame_count = 0
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()

                if not ret:
                    if isinstance(self.source, str) and Path(self.source).exists():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                frame_data = {
                    'camera_id': self.camera_id,
                    'frame': frame,
                    'timestamp': time.time(),
                    'frame_number': frame_count,
                }

                try:
                    if self.config.get('drop_frames_on_full', True):
                        self.frame_queue.put_nowait(frame_data)
                    else:
                        self.frame_queue.put(frame_data, timeout=1.0)

                    self.frames_processed += 1

                except Full:
                    self.frames_dropped += 1
                except (BrokenPipeError, EOFError, OSError):
                    # Queue closed by parent during shutdown - exit cleanly.
                    break

                frame_count += 1

                # Log statistics every 10 seconds
                current_time = time.time()
                if current_time - last_log_time >= 10.0:
                    elapsed = current_time - start_time
                    actual_fps = frame_count / elapsed
                    drop_rate = (self.frames_dropped / frame_count * 100) if frame_count > 0 else 0

                    self.logger.info(
                        f"Camera {self.camera_id} - "
                        f"Frames: {frame_count}, FPS: {actual_fps:.2f}, "
                        f"Dropped: {self.frames_dropped} ({drop_rate:.1f}%)"
                    )
                    last_log_time = current_time

                # Use stop_event.wait() instead of time.sleep so we can
                # react to shutdown within ~one frame time.
                if self.stop_event.wait(timeout=frame_time * 0.8):
                    break

        except KeyboardInterrupt:
            self.logger.info(f"Camera {self.camera_id} interrupted")
        finally:
            cap.release()
            self.logger.info(
                f"Camera {self.camera_id} stopped - "
                f"Total: {frame_count}, Dropped: {self.frames_dropped}"
            )

    def stop(self):
        self.stop_event.set()


class MultiCameraManager:
    """Manages multiple camera streams using multiprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger('CameraManager')

        self.frame_queue = mp.Queue(maxsize=config.get('max_queue_size', 50))

        self.cameras = []
        self.processes = []
        self.stop_events = []  # per-camera mp.Event, shared with the child
        self.started = False

        num_cameras = config.get('num_cameras', 1)
        video_sources = config.get('video_sources', [0] * num_cameras)

        for i in range(num_cameras):
            source = video_sources[i] if i < len(video_sources) else i
            stop_event = mp.Event()
            camera = CameraStream(
                i, source, self.frame_queue, config,
                stop_event=stop_event,
            )
            self.cameras.append(camera)
            self.stop_events.append(stop_event)
    
    def start(self):
        if self.started:
            self.logger.info("Camera processes already running")
            return

        self.logger.info(f"Starting {len(self.cameras)} camera processes")
        
        for camera in self.cameras:
            process = mp.Process(
                target=camera.run,
                name=f'Camera_{camera.camera_id}'
            )
            process.start()
            self.processes.append(process)
        
        self.started = True
        self.logger.info("All camera processes started")
    
    def stop(self):
        if not self.started:
            self.logger.info("Camera processes already stopped")
            return

        self.logger.info("Stopping all camera processes")

        # 1. Signal every child process to exit via its shared mp.Event.
        for event in self.stop_events:
            event.set()

        # 2. Give each child a short grace period to exit cleanly.
        #    The loop uses a per-iteration budget instead of serial 5s joins,
        #    so 8 cameras no longer cost 8*5s = 40s on shutdown.
        deadline = time.time() + 2.0  # total wall-clock budget
        for process in self.processes:
            remaining = max(deadline - time.time(), 0.1)
            process.join(timeout=remaining)
            if process.is_alive():
                self.logger.warning(f"Terminating process {process.name}")
                process.terminate()

        # 3. Reap any stragglers quickly.
        for process in self.processes:
            if process.is_alive():
                process.join(timeout=0.5)

        self.processes = []
        self.started = False
        self.logger.info("All camera processes stopped")
    
    def get_frame_queue(self):
        return self.frame_queue
