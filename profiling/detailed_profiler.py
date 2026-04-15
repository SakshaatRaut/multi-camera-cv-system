# Enhanced profiler with stage-by-stage timing breakdown for bottleneck identification
"""
Detailed profiler for stage-by-stage performance analysis
Implements Experiment E4: Bottleneck Identification
"""

import time
from collections import defaultdict
from utils.logger import get_logger


class DetailedProfiler:
    """
    Tracks time spent in each pipeline stage to identify bottlenecks.
    
    Stages:
    1. Frame Loading (I/O bound)
    2. Preprocessing (CPU bound)
    3. Inference (GPU/CPU bound)
    4. Postprocessing (CPU bound)
    5. Output/Display (I/O bound)
    """
    
    def __init__(self):
        self.logger = get_logger('DetailedProfiler')
        
        # Stage timings
        self.stage_times = defaultdict(list)
        self.total_frames = 0
        
        # Stage names
        self.stages = [
            'frame_loading',
            'preprocessing', 
            'inference',
            'postprocessing',
            'output'
        ]
    
    def record_stage(self, stage_name, duration):
        """Record time spent in a stage"""
        self.stage_times[stage_name].append(duration)
    
    def get_bottleneck_analysis(self):
        """
        Identify which stage is the bottleneck.
        
        Returns:
            Dictionary with stage percentages and bottleneck identification
        """
        total_time = sum(sum(times) for times in self.stage_times.values())
        
        if total_time == 0:
            return {}
        
        analysis = {
            'total_time': total_time,
            'stages': {}
        }
        
        max_percent = 0
        bottleneck = None
        
        for stage in self.stages:
            stage_total = sum(self.stage_times.get(stage, [0]))
            percent = (stage_total / total_time) * 100
            
            analysis['stages'][stage] = {
                'total_time': stage_total,
                'percent': percent,
                'avg_time': stage_total / len(self.stage_times.get(stage, [1]))
            }
            
            if percent > max_percent:
                max_percent = percent
                bottleneck = stage
        
        analysis['bottleneck'] = {
            'stage': bottleneck,
            'percent': max_percent
        }
        
        return analysis
    
    def print_bottleneck_report(self):
        """Print detailed bottleneck analysis"""
        analysis = self.get_bottleneck_analysis()
        
        if not analysis:
            print("No data collected")
            return
        
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS (Experiment E4)")
        print("="*70)
        print(f"\nTotal Pipeline Time: {analysis['total_time']:.3f}s\n")
        
        print("Stage Breakdown:")
        print("-" * 70)
        print(f"{'Stage':<20} {'Time (s)':<12} {'Percentage':<12} {'Avg/Frame (ms)'}")
        print("-" * 70)
        
        for stage in self.stages:
            if stage in analysis['stages']:
                info = analysis['stages'][stage]
                print(f"{stage:<20} {info['total_time']:<12.3f} "
                      f"{info['percent']:<12.1f}% {info['avg_time']*1000:.2f}")
        
        print("-" * 70)
        print(f"\n🔴 BOTTLENECK IDENTIFIED: {analysis['bottleneck']['stage']}")
        print(f"   Takes {analysis['bottleneck']['percent']:.1f}% of total time")
        
        # Provide recommendations
        bottleneck_stage = analysis['bottleneck']['stage']
        print(f"\n💡 RECOMMENDATIONS:")
        
        if bottleneck_stage == 'frame_loading':
            print("   - Use faster storage (SSD instead of HDD)")
            print("   - Increase buffer size")
            print("   - Use video compression with lower decode complexity")
        elif bottleneck_stage == 'inference':
            print("   - Increase batch size for better GPU utilization")
            print("   - Use smaller YOLO model (n instead of m/l)")
            print("   - Ensure GPU is being used (not CPU)")
        elif bottleneck_stage == 'preprocessing':
            print("   - Optimize NumPy operations")
            print("   - Reduce resize operations")
            print("   - Use GPU for preprocessing if available")
        elif bottleneck_stage == 'postprocessing':
            print("   - Minimize drawing operations")
            print("   - Reduce number of detection classes")
            print("   - Optimize bounding box rendering")
        
        print("="*70 + "\n")


# Usage example in pipeline_manager.py:
"""
# In pipeline_manager.py, add timing for each stage:

def _process_batch(self, frames, metadata):
    detailed_profiler = DetailedProfiler()
    
    for frame, meta in zip(frames, metadata):
        # Stage 1: Frame loading (already done by camera)
        # Stage 2: Preprocessing
        t1 = time.time()
        preprocessed = self._preprocess_frame(frame)
        detailed_profiler.record_stage('preprocessing', time.time() - t1)
        
        # Stage 3: Inference
        t2 = time.time()
        result = self.detector.detect_single(preprocessed)
        detailed_profiler.record_stage('inference', time.time() - t2)
        
        # Stage 4: Postprocessing
        t3 = time.time()
        annotated = self.detector.draw_detections(frame, result['detections'])
        detailed_profiler.record_stage('postprocessing', time.time() - t3)
    
    # Print periodic analysis
    if self.total_frames % 100 == 0:
        detailed_profiler.print_bottleneck_report()
"""