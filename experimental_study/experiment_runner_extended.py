"""
Extended Experimental Study Runner for Neural-LSH
==================================================
Extended version with novel parameter combinations and analysis.
Uses ExperimentRunner from experiment_runner.py but runs different scenarios.

Focuses on:
  1. Edge cases (very high k, large m, deep networks)
  2. Speed-oriented configurations
  3. Quality-oriented configurations
  4. Comparative analysis between novel and standard parameters
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experimental_study.experiment_runner import (
    ExperimentRunner, ExperimentConfig, ExperimentResult
)


@dataclass
class ScenarioReport:
    """Report for a scenario execution."""
    scenario_name: str
    description: str
    configs_tested: int
    results: List[ExperimentResult]
    best_result: Optional[ExperimentResult] = None
    worst_result: Optional[ExperimentResult] = None
    avg_recall: float = 0.0
    avg_qps: float = 0.0


class ExtendedExperimentRunner:
    """Extended experiment runner with novel parameter combinations."""
    
    def __init__(self, workspace_dir: str = None, results_dir: str = 'results'):
        """Initialize the extended runner."""
        if workspace_dir is None:
            workspace_dir = str(Path.cwd())
        self.workspace_dir = Path(workspace_dir)
        self.results_dir = self.workspace_dir / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runner = ExperimentRunner(str(self.workspace_dir), results_dir)
        self.scenario_reports: List[ScenarioReport] = []
    
    def scenario_speed_optimized(self, data_type: str = 'sift',
                                query_file: str = 'dataset/sift_query.fvecs') -> ScenarioReport:
        """
        SCENARIO 1: Speed-Optimized Configurations
        Focus: Fast index building and search, minimal memory overhead
        
        Configuration strategy:
        - Low k (fewer neighbors = smaller graph)
        - Low m (fewer partitions = faster partitioning)
        - Shallow networks (fewer layers = faster inference)
        - Few epochs (minimal training time)
        - Large batch size (faster SGD)
        """
        print("\n" + "="*80)
        print("SCENARIO 1: SPEED-OPTIMIZED CONFIGURATIONS")
        print("="*80)
        print("Focus: Minimize total time (build + search)")
        print("Strategy: Low k, low m, shallow networks, minimal training\n")
        
        configs = [
            # Minimal configuration
            ExperimentConfig(
                knn=5, m=30, layers=1, nodes=16,
                epochs=5, batch_size=128, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Minimal + slightly wider
            ExperimentConfig(
                knn=5, m=40, layers=1, nodes=32,
                epochs=5, batch_size=256, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Minimal + slightly deeper
            ExperimentConfig(
                knn=10, m=50, layers=2, nodes=32,
                epochs=10, batch_size=256, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
        ]
        
        results = []
        print(f"Running {len(configs)} configurations...\n")
        for i, config in enumerate(configs, 1):
            try:
                print(f"[{i}/{len(configs)}] k={config.knn}, m={config.m}, "
                      f"layers={config.layers}, nodes={config.nodes}")
                result = self.runner.run_experiment(config, query_file)
                results.append(result)
            except Exception as e:
                print(f"⚠ Failed: {e}\n")
        
        report = self._create_scenario_report(
            'SPEED_OPTIMIZED', 
            'Fast index building and search with minimal resources',
            configs, results
        )
        self.scenario_reports.append(report)
        return report
    
    def scenario_quality_optimized(self, data_type: str = 'sift',
                                  query_file: str = 'dataset/sift_query.fvecs') -> ScenarioReport:
        """
        SCENARIO 2: Quality-Optimized Configurations
        Focus: High recall and accuracy, time is secondary
        
        Configuration strategy:
        - High k (more neighbors = richer graph)
        - High m (more partitions = finer granularity)
        - Deep networks (more layers = better expressiveness)
        - Many epochs (thorough training)
        - Small batch size (more updates per epoch)
        """
        print("\n" + "="*80)
        print("SCENARIO 2: QUALITY-OPTIMIZED CONFIGURATIONS")
        print("="*80)
        print("Focus: Maximize recall and accuracy")
        print("Strategy: High k, high m, deep networks, thorough training\n")
        
        configs = [
            # Quality focused
            ExperimentConfig(
                knn=25, m=150, layers=3, nodes=128,
                epochs=50, batch_size=16, learning_rate=0.0005,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Very high quality
            ExperimentConfig(
                knn=30, m=200, layers=4, nodes=128,
                epochs=50, batch_size=16, learning_rate=0.0005,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Extreme quality
            ExperimentConfig(
                knn=35, m=250, layers=5, nodes=256,
                epochs=100, batch_size=8, learning_rate=0.0005,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
        ]
        
        results = []
        print(f"Running {len(configs)} configurations...\n")
        for i, config in enumerate(configs, 1):
            try:
                print(f"[{i}/{len(configs)}] k={config.knn}, m={config.m}, "
                      f"layers={config.layers}, nodes={config.nodes}")
                result = self.runner.run_experiment(config, query_file)
                results.append(result)
            except Exception as e:
                print(f"⚠ Failed: {e}\n")
        
        report = self._create_scenario_report(
            'QUALITY_OPTIMIZED',
            'High recall and accuracy with more resources',
            configs, results
        )
        self.scenario_reports.append(report)
        return report
    
    def scenario_balanced_fast(self, data_type: str = 'sift',
                              query_file: str = 'dataset/sift_query.fvecs') -> ScenarioReport:
        """
        SCENARIO 3: Balanced Fast Configurations
        Focus: Good balance between speed and quality
        
        Configuration strategy:
        - Medium k (15-20)
        - Medium m (80-120)
        - Medium depth (2-3 layers)
        - Medium training (20-30 epochs)
        - Medium batch size (32-64)
        """
        print("\n" + "="*80)
        print("SCENARIO 3: BALANCED FAST CONFIGURATIONS")
        print("="*80)
        print("Focus: Practical balance between speed and quality")
        print("Strategy: Medium k, m, layers, and training\n")
        
        configs = [
            # Balanced v1
            ExperimentConfig(
                knn=15, m=80, layers=2, nodes=64,
                epochs=25, batch_size=32, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Balanced v2
            ExperimentConfig(
                knn=20, m=100, layers=2, nodes=64,
                epochs=30, batch_size=32, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Balanced v3 (slightly more complex)
            ExperimentConfig(
                knn=20, m=120, layers=3, nodes=96,
                epochs=30, batch_size=32, learning_rate=0.0005,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
        ]
        
        results = []
        print(f"Running {len(configs)} configurations...\n")
        for i, config in enumerate(configs, 1):
            try:
                print(f"[{i}/{len(configs)}] k={config.knn}, m={config.m}, "
                      f"layers={config.layers}, nodes={config.nodes}")
                result = self.runner.run_experiment(config, query_file)
                results.append(result)
            except Exception as e:
                print(f"⚠ Failed: {e}\n")
        
        report = self._create_scenario_report(
            'BALANCED_FAST',
            'Practical balance between performance and quality',
            configs, results
        )
        self.scenario_reports.append(report)
        return report
    
    def scenario_extreme_parameters(self, data_type: str = 'sift',
                                   query_file: str = 'dataset/sift_query.fvecs') -> ScenarioReport:
        """
        SCENARIO 4: Extreme Parameter Configurations
        Focus: Test boundaries and edge cases
        
        Tests what happens at extreme values:
        - Very low k (1-3)
        - Very high k (40+)
        - Very large m (300+)
        - Very deep networks (6+ layers)
        """
        print("\n" + "="*80)
        print("SCENARIO 4: EXTREME PARAMETER CONFIGURATIONS")
        print("="*80)
        print("Focus: Test parameter boundaries and edge cases")
        print("Strategy: Extreme values to understand limits\n")
        
        configs = [
            # Minimal graph
            ExperimentConfig(
                knn=3, m=50, layers=2, nodes=32,
                epochs=20, batch_size=32, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Maximal graph
            ExperimentConfig(
                knn=50, m=300, layers=2, nodes=64,
                epochs=25, batch_size=32, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Very deep network
            ExperimentConfig(
                knn=15, m=100, layers=7, nodes=32,
                epochs=30, batch_size=32, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
            # Very wide network
            ExperimentConfig(
                knn=15, m=100, layers=1, nodes=512,
                epochs=25, batch_size=64, learning_rate=0.001,
                data_type=data_type, data_file=f'dataset/{data_type}_base.fvecs'
            ),
        ]
        
        results = []
        print(f"Running {len(configs)} configurations...\n")
        for i, config in enumerate(configs, 1):
            try:
                print(f"[{i}/{len(configs)}] k={config.knn}, m={config.m}, "
                      f"layers={config.layers}, nodes={config.nodes}")
                result = self.runner.run_experiment(config, query_file)
                results.append(result)
            except Exception as e:
                print(f"⚠ Failed: {e}\n")
        
        report = self._create_scenario_report(
            'EXTREME_PARAMETERS',
            'Edge cases testing parameter boundaries',
            configs, results
        )
        self.scenario_reports.append(report)
        return report
    
    def scenario_dataset_comparison(self, query_file_sift: str = 'dataset/sift_query.fvecs',
                                   query_file_mnist: str = 'dataset/t10k-images.idx3-ubyte') -> ScenarioReport:
        """
        SCENARIO 5: Dataset Comparison
        Focus: Compare performance across SIFT and MNIST datasets
        
        Uses same configuration but different datasets to understand
        how dataset characteristics affect performance.
        """
        print("\n" + "="*80)
        print("SCENARIO 5: DATASET COMPARISON")
        print("="*80)
        print("Focus: Compare SIFT vs MNIST performance")
        print("Strategy: Same config, different datasets\n")
        
        configs = [
            # SIFT dataset
            ExperimentConfig(
                knn=15, m=100, layers=2, nodes=64,
                epochs=25, batch_size=32, learning_rate=0.001,
                data_type='sift', data_file='dataset/sift_base.fvecs'
            ),
            # MNIST dataset
            ExperimentConfig(
                knn=15, m=100, layers=2, nodes=64,
                epochs=25, batch_size=32, learning_rate=0.001,
                data_type='mnist', data_file='dataset/train-images.idx3-ubyte'
            ),
        ]
        
        results = []
        query_files = [query_file_sift, query_file_mnist]
        print(f"Running {len(configs)} configurations...\n")
        
        for i, (config, query_file) in enumerate(zip(configs, query_files), 1):
            try:
                print(f"[{i}/{len(configs)}] {config.data_type.upper()} dataset")
                result = self.runner.run_experiment(config, query_file)
                results.append(result)
            except Exception as e:
                print(f"⚠ Failed: {e}\n")
        
        report = self._create_scenario_report(
            'DATASET_COMPARISON',
            'Compare SIFT vs MNIST datasets',
            configs, results
        )
        self.scenario_reports.append(report)
        return report
    
    def _create_scenario_report(self, name: str, description: str,
                               configs: List[ExperimentConfig],
                               results: List[ExperimentResult]) -> ScenarioReport:
        """Create a report for a scenario."""
        best_result = None
        worst_result = None
        avg_recall = 0.0
        avg_qps = 0.0
        
        if results:
            best_result = max(results, key=lambda r: r.avg_recall)
            worst_result = min(results, key=lambda r: r.avg_recall)
            avg_recall = sum(r.avg_recall for r in results) / len(results)
            avg_qps = sum(r.avg_qps for r in results) / len(results)
        
        return ScenarioReport(
            scenario_name=name,
            description=description,
            configs_tested=len(configs),
            results=results,
            best_result=best_result,
            worst_result=worst_result,
            avg_recall=avg_recall,
            avg_qps=avg_qps
        )
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report of all scenarios."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENTAL REPORT")
        print("="*80)
        
        report_lines = [
            "="*80,
            "NEURAL-LSH EXTENDED EXPERIMENTAL STUDY",
            "="*80,
            f"Total Scenarios: {len(self.scenario_reports)}",
            f"Total Configurations Tested: {sum(r.configs_tested for r in self.scenario_reports)}",
            "",
        ]
        
        for report in self.scenario_reports:
            report_lines.extend([
                f"\n{'='*80}",
                f"SCENARIO: {report.scenario_name}",
                f"{'='*80}",
                f"Description: {report.description}",
                f"Configurations Tested: {report.configs_tested}",
                f"Average Recall: {report.avg_recall:.4f}",
                f"Average QPS: {report.avg_qps:.2f}",
            ])
            
            if report.best_result:
                report_lines.extend([
                    f"\nBest Result:",
                    f"  Recall: {report.best_result.avg_recall:.4f}",
                    f"  QPS: {report.best_result.avg_qps:.2f}",
                    f"  Build Time: {report.best_result.build_time:.2f}s",
                    f"  Config: k={report.best_result.config.knn}, "
                    f"m={report.best_result.config.m}, "
                    f"layers={report.best_result.config.layers}, "
                    f"nodes={report.best_result.config.nodes}",
                ])
            
            if report.worst_result:
                report_lines.extend([
                    f"\nWorst Result:",
                    f"  Recall: {report.worst_result.avg_recall:.4f}",
                    f"  QPS: {report.worst_result.avg_qps:.2f}",
                    f"  Build Time: {report.worst_result.build_time:.2f}s",
                    f"  Config: k={report.worst_result.config.knn}, "
                    f"m={report.worst_result.config.m}, "
                    f"layers={report.worst_result.config.layers}, "
                    f"nodes={report.worst_result.config.nodes}",
                ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / 'extended_experimental_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {report_file}")
        return report_text
    
    def run_all_scenarios(self, data_type: str = 'sift') -> None:
        """Run all experimental scenarios."""
        print("\n" + "="*80)
        print("EXTENDED EXPERIMENTAL STUDY - RUNNING ALL SCENARIOS")
        print("="*80)
        
        start_time = time.time()
        
        try:
            self.scenario_speed_optimized(data_type)
            self.scenario_balanced_fast(data_type)
            self.scenario_quality_optimized(data_type)
            self.scenario_extreme_parameters(data_type)
            self.scenario_dataset_comparison()
        except Exception as e:
            print(f"⚠ Error during scenario execution: {e}")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        self.generate_comprehensive_report()
        self.runner.save_results_summary()
        
        print(f"\n✓ All scenarios completed in {elapsed_time/60:.1f} minutes")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extended experimental study for Neural-LSH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios
  python3 experiment_runner_extended.py --all
  
  # Run specific scenario
  python3 experiment_runner_extended.py --scenario speed
  
  # Run with MNIST
  python3 experiment_runner_extended.py --all --data mnist
  
  # Run only dataset comparison
  python3 experiment_runner_extended.py --scenario dataset
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all scenarios')
    parser.add_argument('--scenario', 
                       choices=['speed', 'balanced', 'quality', 'extreme', 'dataset'],
                       help='Specific scenario to run')
    parser.add_argument('--data', choices=['sift', 'mnist'], default='sift',
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    runner = ExtendedExperimentRunner()
    
    scenario_map = {
        'speed': runner.scenario_speed_optimized,
        'balanced': runner.scenario_balanced_fast,
        'quality': runner.scenario_quality_optimized,
        'extreme': runner.scenario_extreme_parameters,
        'dataset': runner.scenario_dataset_comparison,
    }
    
    if args.all:
        runner.run_all_scenarios(args.data)
    elif args.scenario:
        if args.scenario == 'dataset':
            scenario_map[args.scenario]()
        else:
            scenario_map[args.scenario](args.data)
    else:
        runner.run_all_scenarios(args.data)


if __name__ == '__main__':
    main()
