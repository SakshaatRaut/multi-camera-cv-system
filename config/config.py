# System Configuration
"""
System Configuration
All configurable parameters for the multi-camera CV system
"""

# System Configuration
SYSTEM_CONFIG = {
    # Camera Settings
    'num_cameras': 4,
    'video_sources': [],  # Will be set at runtime
    'frame_width': 640,
    'frame_height': 480,
    'target_fps': 30,
    
    # Buffer Settings
    'buffer_size': 10,
    'max_queue_size': 50,
    'drop_frames_on_full': True,
    
    # Detection Settings
    'model_name': 'yolov8n.pt',
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'batch_size': 4,
    
    # Pipeline Settings
    'num_workers': 4,
    'batch_timeout': 0.05,  # seconds
    
    # Profiling Settings
    'enable_profiling': True,
    'profile_interval': 1.0,  # seconds
    'save_stats': True,
    'stats_file': 'results/system_stats.csv',
    
    # Output Settings
    'output_dir': 'results/',
    'save_visualizations': True,
}

# YOLO Classes (COCO dataset - 80 objects)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]