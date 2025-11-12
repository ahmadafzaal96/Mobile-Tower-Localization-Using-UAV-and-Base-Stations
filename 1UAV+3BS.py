# UAV-Based Cell Tower Localization Simulation -
## Proof of Concept with Accuracy Metrics

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class CellTowerLocalizationSimulator:
    """
    Simulator for UAV-based cell tower localization using RF parameters
    """
    
    def __init__(self, area_size: float = 1000.0):
        self.area_size = area_size
        self.towers = []
        self.ground_truth = None
        
        # Realistic noise models
        self.noise_models = {
            'toa': 5e-9,       # 5ns timing error (≈1.5m range error)
            'aoa': np.radians(1.5),  # 1.5 degree angle error
        }
        
        self.c = 3e8  # speed of light
        
    def setup_scenario(self):
        """Setup the simulation scenario"""
        # Reference towers (known positions)
        self.towers = [
            np.array([100, 100, 35]),    # Reference tower 1
            np.array([900, 200, 40]),    # Reference tower 2  
            np.array([150, 850, 30]),    # Reference tower 3
        ]
        
        # Target tower (unknown position - ground truth)
        self.ground_truth = np.array([650, 720, 32])
        
        print("Scenario Setup Complete:")
        print(f"Reference Towers: {[f'({t[0]}, {t[1]}, {t[2]})' for t in self.towers]}")
        print(f"Target Tower Ground Truth: ({self.ground_truth[0]}, {self.ground_truth[1]}, {self.ground_truth[2]})")
        
        return self.towers, self.ground_truth
    
    def generate_spiral_trajectory(self, num_points: int, altitude: float = 100.0):
        """Generate UAV trajectory"""
        t = np.linspace(0, 6*np.pi, num_points)
        radius = 100 + 350 * (t / (6*np.pi))  # Gradually increasing radius
        
        x = 500 + radius * np.cos(t)
        y = 500 + radius * np.sin(t)
        z = np.full(num_points, altitude)
        
        trajectory = np.column_stack([x, y, z])
        return trajectory
    
    def generate_measurements(self, uav_position: np.ndarray, tower_position: np.ndarray) -> Dict:
        """Generate noisy RF measurements"""
        true_distance = LA.norm(uav_position - tower_position)
        
        # Time of Arrival with noise
        toa_measurement = true_distance / self.c + np.random.normal(0, self.noise_models['toa'])
        
        # Angle of Arrival with noise (azimuth only for simplicity)
        dx = tower_position[0] - uav_position[0]
        dy = tower_position[1] - uav_position[1]
        
        true_azimuth = np.arctan2(dy, dx)
        aoa_measurement = true_azimuth + np.random.normal(0, self.noise_models['aoa'])
        
        return {
            'toa': toa_measurement,
            'aoa': aoa_measurement,
            'true_distance': true_distance
        }

class LocalizationAlgorithms:
    """WORKING localization algorithms without bounds issues"""
    
    def __init__(self, c: float = 3e8):
        self.c = c
    
    def simple_triangulation(self, uav_positions: List[np.ndarray], measurements: List[Dict]) -> np.ndarray:
        """Simple triangulation using intersection of bearing lines"""
        if len(uav_positions) < 2:
            return np.array([500, 500, 30])
        
        # Convert AOA measurements to lines and find intersection
        lines = []
        for uav_pos, meas in zip(uav_positions, measurements):
            angle = meas['aoa']
            # Create a line from UAV position in the direction of the angle
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            lines.append((uav_pos[:2], direction[:2]))  # Use only x,y for 2D triangulation
        
        # Find intersection point (simplified - average of closest points between lines)
        intersections = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                p1, d1 = lines[i]
                p2, d2 = lines[j]
                
                # Find closest point between two lines
                A = np.column_stack([d1, -d2])
                b = p2 - p1
                try:
                    t = np.linalg.solve(A, b)
                    point1 = p1 + t[0] * d1
                    point2 = p2 + t[1] * d2
                    avg_point = (point1 + point2) / 2
                    intersections.append(avg_point)
                except:
                    continue
        
        if intersections:
            # Average all intersection points and assume reasonable height
            avg_2d = np.mean(intersections, axis=0)
            return np.array([avg_2d[0], avg_2d[1], 30])
        else:
            return np.array([500, 500, 30])
    
    def range_based_localization(self, uav_positions: List[np.ndarray], measurements: List[Dict]) -> np.ndarray:
        """Range-based localization using least squares without bounds"""
        if len(uav_positions) < 3:
            return np.array([500, 500, 30])
        
        def residuals(params, uav_positions, ranges):
            x, y, z = params
            tower_pos = np.array([x, y, z])
            residuals = []
            
            for i, uav_pos in enumerate(uav_positions):
                predicted_range = LA.norm(tower_pos - uav_pos)
                measured_range = ranges[i] * self.c
                residuals.append(predicted_range - measured_range)
                
            return np.array(residuals)
        
        ranges = [m['toa'] for m in measurements]
        
        # Use centroid of UAV positions as initial guess
        x0 = np.mean(uav_positions, axis=0)
        
        try:
            result = least_squares(residuals, x0, args=(uav_positions, ranges), 
                                  method='lm')  # No bounds for LM
            return result.x
        except:
            return x0
    
    def hybrid_localization(self, uav_positions: List[np.ndarray], measurements: List[Dict]) -> np.ndarray:
        """Hybrid localization combining range and angle without bounds"""
        if len(uav_positions) < 2:
            return np.array([500, 500, 30])
        
        def residuals(params, uav_positions, measurements):
            x, y, z = params
            tower_pos = np.array([x, y, z])
            residuals = []
            
            for i, (uav_pos, meas) in enumerate(zip(uav_positions, measurements)):
                # Range residual
                predicted_range = LA.norm(tower_pos - uav_pos)
                measured_range = meas['toa'] * self.c
                residuals.append(predicted_range - measured_range)
                
                # Angle residual (weighted less)
                dx = tower_pos[0] - uav_pos[0]
                dy = tower_pos[1] - uav_pos[1]
                predicted_angle = np.arctan2(dy, dx)
                residuals.append(0.3 * (predicted_angle - meas['aoa']))  # Lower weight
                
            return np.array(residuals)
        
        x0 = np.mean(uav_positions, axis=0)
        
        try:
            result = least_squares(residuals, x0, args=(uav_positions, measurements), 
                                  method='lm')  # No bounds
            return result.x
        except:
            return x0

class AccuracyMetrics:
    """Accuracy metrics calculation"""
    
    def calculate_metrics(self, estimated_positions: List[np.ndarray], 
                         ground_truth: np.ndarray) -> Dict:
        """Calculate accuracy metrics"""
        errors = [LA.norm(est - ground_truth) for est in estimated_positions]
        errors = np.array(errors)
        
        metrics = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'cep_50': np.percentile(errors, 50),
            'cep_95': np.percentile(errors, 95),
        }
        
        return metrics

class Visualization:
    """Visualization tools"""
    
    @staticmethod
    def plot_scenario(towers: List[np.ndarray], ground_truth: np.ndarray, 
                     trajectory: np.ndarray, estimates: Dict = None):
        """Plot the complete scenario"""
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: 3D View
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot reference towers
        tower_array = np.array(towers)
        ax1.scatter(tower_array[:, 0], tower_array[:, 1], tower_array[:, 2], 
                   c='blue', s=100, marker='^', label='Reference Towers', alpha=0.7)
        
        # Plot ground truth
        ax1.scatter(ground_truth[0], ground_truth[1], ground_truth[2], 
                   c='red', s=200, marker='*', label='Target Tower (GT)')
        
        # Plot trajectory
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'g-', alpha=0.6, label='UAV Trajectory')
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   c='green', s=50, marker='o', label='Start')
        
        # Plot estimates if provided
        if estimates is not None:
            colors = ['orange', 'purple', 'brown']
            for i, (method, est) in enumerate(estimates.items()):
                ax1.scatter(est[0], est[1], est[2], c=colors[i], s=80, marker='s', label=method)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Scenario Overview')
        ax1.legend()
        
        # Plot 2: Top View
        ax2 = fig.add_subplot(132)
        ax2.scatter(tower_array[:, 0], tower_array[:, 1], c='blue', s=100, marker='^', alpha=0.7, label='Ref Towers')
        ax2.scatter(ground_truth[0], ground_truth[1], c='red', s=200, marker='*', label='Target (GT)')
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'g-', alpha=0.6, label='Trajectory')
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=50, marker='o', label='Start')
        
        if estimates is not None:
            colors = ['orange', 'purple', 'brown']
            for i, (method, est) in enumerate(estimates.items()):
                ax2.scatter(est[0], est[1], c=colors[i], s=80, marker='s', label=method)
                # Draw error lines
                ax2.plot([est[0], ground_truth[0]], [est[1], ground_truth[1]], 
                        color=colors[i], linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View with Estimates')
        ax2.grid(True)
        ax2.legend()
        ax2.axis('equal')
        
        # Plot 3: Error Analysis
        ax3 = fig.add_subplot(133)
        if estimates is not None:
            errors = [LA.norm(est - ground_truth) for est in estimates.values()]
            methods = list(estimates.keys())
            bars = ax3.bar(methods, errors, color=['orange', 'purple', 'brown'], alpha=0.7)
            ax3.set_ylabel('Localization Error (m)')
            ax3.set_title('Localization Error by Method')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, error in zip(bars, errors):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{error:.1f}m', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_error_distribution(final_errors: Dict, metrics_summary: Dict):
        """Plot error distributions and metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot of errors
        error_data = [final_errors[method] for method in final_errors.keys()]
        ax1.boxplot(error_data, labels=final_errors.keys())
        ax1.set_ylabel('Localization Error (m)')
        ax1.set_title('Error Distribution by Method')
        ax1.grid(True)
        
        # CDF plot
        ax2.set_xlabel('Localization Error (m)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function')
        for method, errors in final_errors.items():
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
            ax2.plot(sorted_errors, cdf, label=method, linewidth=2)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics table
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE METRICS")
        print("="*70)
        print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Median':<8} {'CEP50':<8} {'CEP95':<8}")
        print("-"*70)
        for method, metrics in metrics_summary.items():
            print(f"{method:<15} {metrics['mean_error']:<8.1f} {metrics['std_error']:<8.1f} "
                  f"{metrics['median_error']:<8.1f} {metrics['cep_50']:<8.1f} {metrics['cep_95']:<8.1f}")

# Main Simulation Execution
print("# UAV-Based Cell Tower Localization Simulation - WORKING VERSION")
print("## Proof of Concept with Accuracy Metrics\n")

# Initialize simulator
simulator = CellTowerLocalizationSimulator(area_size=1000.0)
localizer = LocalizationAlgorithms()
metrics_calc = AccuracyMetrics()
viz = Visualization()

# Setup scenario
reference_towers, target_tower = simulator.setup_scenario()

# Generate UAV trajectory
print("\nGenerating UAV trajectory...")
trajectory = simulator.generate_spiral_trajectory(num_points=20, altitude=100.0)
print(f"Trajectory points: {len(trajectory)}")

# Run Monte Carlo simulations
print("\nRunning Monte Carlo simulations...")
num_monte_carlo = 50
methods = ['Triangulation', 'Range-Based', 'Hybrid']
final_estimates = {method: [] for method in methods}
final_errors = {method: [] for method in methods}

for mc_run in range(num_monte_carlo):
    if (mc_run + 1) % 10 == 0:
        print(f"Completed {mc_run + 1}/{num_monte_carlo} Monte Carlo runs...")
    
    # For each Monte Carlo run, collect measurements from UAV to TARGET tower
    target_measurements = []
    for uav_pos in trajectory:
        # Measure the target tower from UAV position
        meas = simulator.generate_measurements(uav_pos, target_tower)
        target_measurements.append(meas)
    
    # Run localization using all measurements
    current_positions = trajectory
    current_measurements = target_measurements
    
    # Triangulation Localization
    triang_est = localizer.simple_triangulation(current_positions, current_measurements)
    final_estimates['Triangulation'].append(triang_est)
    final_errors['Triangulation'].append(LA.norm(triang_est - target_tower))
    
    # Range-Based Localization
    range_est = localizer.range_based_localization(current_positions, current_measurements)
    final_estimates['Range-Based'].append(range_est)
    final_errors['Range-Based'].append(LA.norm(range_est - target_tower))
    
    # Hybrid Localization
    hybrid_est = localizer.hybrid_localization(current_positions, current_measurements)
    final_estimates['Hybrid'].append(hybrid_est)
    final_errors['Hybrid'].append(LA.norm(hybrid_est - target_tower))

print("Monte Carlo simulation completed!")

# Calculate comprehensive metrics
print("\n## Accuracy Metrics Results")
print("="*50)

metrics_summary = {}
for method in methods:
    if final_estimates[method]:
        metrics = metrics_calc.calculate_metrics(final_estimates[method], target_tower)
        metrics_summary[method] = metrics
        
        print(f"\n{method}:")
        print(f"  Mean Error: {metrics['mean_error']:.2f} m")
        print(f"  STD: {metrics['std_error']:.2f} m")
        print(f"  RMSE: {metrics['rmse']:.2f} m")
        print(f"  Median: {metrics['median_error']:.2f} m")
        print(f"  CEP50: {metrics['cep_50']:.2f} m")
        print(f"  CEP95: {metrics['cep_95']:.2f} m")

# Visualization
print("\n## Simulation Visualizations")
print("="*50)

# Plot scenario with sample estimates from first run
print("\n1. Scenario Overview with Sample Estimates:")
sample_estimates = {
    'Triangulation': final_estimates['Triangulation'][0],
    'Range-Based': final_estimates['Range-Based'][0], 
    'Hybrid': final_estimates['Hybrid'][0]
}
viz.plot_scenario(reference_towers, target_tower, trajectory, sample_estimates)

# Plot error distributions
print("\n2. Error Distribution Analysis:")
viz.plot_error_distribution(final_errors, metrics_summary)

# Final summary
print("\n## SIMULATION SUMMARY")
print("="*60)
print("UAV-Based Cell Tower Localization Proof of Concept")
print("RF Parameters: TOA, AOA with realistic noise models")
print(f"Monte Carlo Runs: {num_monte_carlo}")
print(f"Measurement Points: {len(trajectory)}")
print("\nKEY FINDINGS:")

for method in methods:
    if method in metrics_summary:
        mean_err = metrics_summary[method]['mean_error']
        cep95 = metrics_summary[method]['cep_95']
        print(f"  {method:15} | Mean Error: {mean_err:5.1f}m | CEP95: {cep95:5.1f}m")

if metrics_summary:
    best_method = min(metrics_summary.keys(), key=lambda x: metrics_summary[x]['mean_error'])
    best_error = metrics_summary[best_method]['mean_error']
    print(f"\nRECOMMENDATION: {best_method} provides the best accuracy with {best_error:.1f}m mean error")
    
    if best_error < 20:
        print("✅ EXCELLENT: This successfully demonstrates high-accuracy UAV-based cell tower localization!")
        print("   The technique is feasible for practical applications.")
    elif best_error < 50:
        print("✅ GOOD: This demonstrates the feasibility of UAV-based cell tower localization.")
        print("   Accuracy is sufficient for many applications.")
    elif best_error < 100:
        print("⚠️  MODERATE: Results show potential but need improvement in accuracy.")
        print("   Further optimization or additional sensors may be needed.")
    else:
        print("❌ POOR: Current accuracy is insufficient for practical applications.")
        print("   Fundamental improvements in algorithms or sensor quality are required.")
