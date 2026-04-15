# Amdahl's Law speedup prediction and comparison (Experiment E3)

"""
Amdahl's Law Analysis for Multi-Stream Scaling
Implements theoretical prediction vs empirical measurement comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.logger import get_logger


class AmdahlsLawAnalyzer:
    """
    Analyzes multi-stream scaling using Amdahl's Law.
    
    Amdahl's Law: Speedup = 1 / ((1 - P) + P/N)
    where:
        P = fraction of program that can be parallelized
        N = number of processors (streams in our case)
    """
    
    def __init__(self):
        self.logger = get_logger('AmdahlsLaw')
        self.measurements = {}  # {num_streams: fps}
    
    def add_measurement(self, num_streams, fps, latency=None):
        """Record empirical measurement"""
        self.measurements[num_streams] = {
            'fps': fps,
            'latency': latency,
            'speedup': None  # Will be calculated
        }
    
    def calculate_speedup(self, baseline_streams=1):
        """
        Calculate empirical speedup relative to baseline.
        
        Args:
            baseline_streams: Number of streams to use as baseline (usually 1)
        """
        if baseline_streams not in self.measurements:
            self.logger.error(f"No measurement for baseline ({baseline_streams} streams)")
            return
        
        baseline_fps = self.measurements[baseline_streams]['fps']
        
        for num_streams, data in self.measurements.items():
            data['speedup'] = data['fps'] / baseline_fps
    
    def predict_speedup(self, parallel_fraction, num_streams_list):
        """
        Predict speedup using Amdahl's Law.
        
        Args:
            parallel_fraction: Fraction of work that can be parallelized (0-1)
            num_streams_list: List of stream counts to predict for
            
        Returns:
            Dictionary of {num_streams: predicted_speedup}
        """
        predictions = {}
        
        for N in num_streams_list:
            # Amdahl's Law formula
            serial_fraction = 1 - parallel_fraction
            speedup = 1 / (serial_fraction + (parallel_fraction / N))
            predictions[N] = speedup
        
        return predictions
    
    def estimate_parallel_fraction(self):
        """
        Estimate parallel fraction from empirical data.
        
        Uses measurements to back-calculate P from Amdahl's Law.
        """
        if len(self.measurements) < 2:
            self.logger.warning("Need at least 2 measurements to estimate P")
            return 0.9  # Default assumption
        
        # Use measurement with highest stream count
        max_streams = max(self.measurements.keys())
        if max_streams == 1:
            return 0.9
        
        empirical_speedup = self.measurements[max_streams]['speedup']
        N = max_streams
        
        # Solve Amdahl's Law for P:
        # S = 1 / ((1-P) + P/N)
        # S * ((1-P) + P/N) = 1
        # S - S*P + S*P/N = 1
        # S*P/N - S*P = 1 - S
        # S*P*(1/N - 1) = 1 - S
        # P = (1 - S) / (S*(1/N - 1))
        
        if empirical_speedup <= 1:
            return 0.5  # Poor parallelism
        
        denominator = empirical_speedup * (1/N - 1)
        if denominator == 0:
            return 0.9
        
        P = (1 - empirical_speedup) / denominator
        P = max(0.0, min(1.0, P))  # Clamp to [0, 1]
        
        return P
    
    def generate_report(self, output_file='results/amdahls_law_analysis.txt'):
        """Generate comprehensive Amdahl's Law analysis report"""
        self.calculate_speedup()
        P = self.estimate_parallel_fraction()
        
        stream_counts = sorted(self.measurements.keys())
        predictions = self.predict_speedup(P, stream_counts)
        
        # Generate report
        report = []
        report.append("="*70)
        report.append("AMDAHL'S LAW ANALYSIS (Experiment E3)")
        report.append("="*70)
        report.append("")
        report.append(f"Estimated Parallel Fraction (P): {P:.3f}")
        report.append(f"Serial Fraction (1-P): {1-P:.3f}")
        report.append("")
        report.append("Speedup Analysis:")
        report.append("-"*70)
        report.append(f"{'Streams':<10} {'Measured FPS':<15} {'Speedup':<15} "
                     f"{'Predicted':<15} {'Error'}")
        report.append("-"*70)
        
        for num_streams in stream_counts:
            measured = self.measurements[num_streams]
            predicted_speedup = predictions[num_streams]
            actual_speedup = measured['speedup']
            error = abs(predicted_speedup - actual_speedup) / predicted_speedup * 100
            
            report.append(f"{num_streams:<10} {measured['fps']:<15.2f} "
                         f"{actual_speedup:<15.2f} {predicted_speedup:<15.2f} "
                         f"{error:.1f}%")
        
        report.append("-"*70)
        report.append("")
        
        # Analysis
        report.append("INTERPRETATION:")
        if P > 0.9:
            report.append("✓ Excellent parallelization (P > 0.9)")
            report.append("  System scales well with multiple streams")
        elif P > 0.7:
            report.append("⚠ Good parallelization (0.7 < P < 0.9)")
            report.append("  Some serial bottlenecks present")
        else:
            report.append("✗ Poor parallelization (P < 0.7)")
            report.append("  Significant serial bottleneck limits scaling")
        
        report.append("")
        report.append("THEORETICAL LIMITS:")
        max_speedup = 1 / (1 - P) if P < 1 else float('inf')
        report.append(f"  Maximum possible speedup (infinite streams): {max_speedup:.2f}x")
        report.append(f"  Speedup at 8 streams: {predictions.get(8, 'N/A'):.2f}x")
        report.append(f"  Speedup at 16 streams: {predictions.get(16, 'N/A'):.2f}x")
        
        report.append("")
        report.append("="*70)
        
        # Print and save
        report_text = '\n'.join(report)
        print(report_text)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Amdahl's Law report saved to {output_file}")
        
        return P, predictions
    
    def plot_speedup_curve(self, output_file='results/amdahls_law_plot.png'):
        """Generate speedup vs stream count plot"""
        self.calculate_speedup()
        P = self.estimate_parallel_fraction()
        
        # Empirical data
        stream_counts = sorted(self.measurements.keys())
        empirical_speedups = [self.measurements[n]['speedup'] for n in stream_counts]
        
        # Theoretical predictions
        theory_streams = np.arange(1, max(stream_counts)*2 + 1)
        theory_speedups = [1 / ((1-P) + P/N) for N in theory_streams]
        
        # Ideal linear speedup
        ideal_speedups = theory_streams
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(theory_streams, theory_speedups, 'b-', linewidth=2, 
                label=f'Amdahl\'s Law (P={P:.2f})')
        plt.plot(theory_streams, ideal_speedups, 'g--', linewidth=1, 
                label='Ideal Linear Speedup')
        plt.scatter(stream_counts, empirical_speedups, color='red', s=100, 
                   zorder=5, label='Measured')
        
        plt.xlabel('Number of Streams', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title('Multi-Stream Scaling: Amdahl\'s Law vs Empirical', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Speedup plot saved to {output_file}")


# Example usage script
if __name__ == '__main__':
    # Example: Run system with different stream counts and analyze
    analyzer = AmdahlsLawAnalyzer()
    
    # Add measurements (these would come from actual runs)
    analyzer.add_measurement(1, fps=30.0, latency=0.033)
    analyzer.add_measurement(2, fps=55.0, latency=0.036)
    analyzer.add_measurement(4, fps=95.0, latency=0.042)
    
    # Generate analysis
    P, predictions = analyzer.generate_report()
    analyzer.plot_speedup_curve()