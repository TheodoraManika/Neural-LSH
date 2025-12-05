#!/usr/bin/env python3
"""
Master Experiment Orchestration Script
========================================
Coordinates all experimental study phases:
  1. Hyperparameter grid search
  2. Build phase analysis
  3. Search phase analysis
  4. Comparative analysis with old methods
  5. Generate comprehensive reports
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time


class ExperimentOrchestrator:
    """Orchestrates the entire experimental study."""
    
    def __init__(self, workspace_dir: str = '/Users/theodora/Desktop/uni/project/Neural-LSH'):
        self.workspace_dir = Path(workspace_dir)
        self.experimental_study_dir = self.workspace_dir / 'experimental_study'
        self.experimental_study_dir.mkdir(parents=True, exist_ok=True)
    
    def run_grid_search(self, config_args: dict = None) -> bool:
        """Run hyperparameter grid search."""
        print("\n" + "="*80)
        print("PHASE 1: Hyperparameter Grid Search")
        print("="*80 + "\n")
        
        cmd = ['python', 'experimental_study/experiment_runner.py']
        
        if config_args:
            if 'knn' in config_args:
                cmd.extend(['--knn'] + [str(x) for x in config_args['knn']])
            if 'm' in config_args:
                cmd.extend(['--m'] + [str(x) for x in config_args['m']])
            if 'layers' in config_args:
                cmd.extend(['--layers'] + [str(x) for x in config_args['layers']])
            if 'nodes' in config_args:
                cmd.extend(['--nodes'] + [str(x) for x in config_args['nodes']])
            if 'epochs' in config_args:
                cmd.extend(['--epochs'] + [str(x) for x in config_args['epochs']])
            if 'data' in config_args:
                cmd.extend(['--data', config_args['data']])
            if 'max_experiments' in config_args:
                cmd.extend(['--max-experiments', str(config_args['max_experiments'])])
        
        try:
            result = subprocess.run(cmd, cwd=str(self.workspace_dir), timeout=None)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("❌ Grid search timed out")
            return False
        except Exception as e:
            print(f"❌ Grid search failed: {e}")
            return False
    
    def generate_build_summary(self) -> bool:
        """Generate build phase summary."""
        print("\n" + "="*80)
        print("PHASE 2: Build Summary Report")
        print("="*80 + "\n")
        
        cmd = ['python', 'experimental_study/summary_report_generator.py', '--build']
        
        try:
            result = subprocess.run(cmd, cwd=str(self.workspace_dir))
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Build summary generation failed: {e}")
            return False
    
    def generate_search_summary(self) -> bool:
        """Generate search phase summary."""
        print("\n" + "="*80)
        print("PHASE 3: Search Summary Report")
        print("="*80 + "\n")
        
        cmd = ['python', 'experimental_study/summary_report_generator.py', '--search']
        
        try:
            result = subprocess.run(cmd, cwd=str(self.workspace_dir))
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Search summary generation failed: {e}")
            return False
    
    def generate_combined_analysis(self) -> bool:
        """Generate combined build + search analysis."""
        print("\n" + "="*80)
        print("PHASE 4: Combined Analysis Report")
        print("="*80 + "\n")
        
        cmd = ['python', 'experimental_study/summary_report_generator.py', '--combined']
        
        try:
            result = subprocess.run(cmd, cwd=str(self.workspace_dir))
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Combined analysis generation failed: {e}")
            return False
    
    def run_comparative_analysis(self, dataset: str = 'sift') -> bool:
        """Compare Neural-LSH with old methods."""
        print("\n" + "="*80)
        print("PHASE 5: Comparative Analysis (vs Old Methods)")
        print("="*80 + "\n")
        
        cmd = ['python', 'experimental_study/comparative_analysis.py', '--dataset', dataset]
        
        try:
            result = subprocess.run(cmd, cwd=str(self.workspace_dir))
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Comparative analysis failed: {e}")
            return False
    
    def run_full_study(self, config_args: dict = None) -> None:
        """Run the complete experimental study."""
        print("\n" + "="*80)
        print("NEURAL-LSH EXPERIMENTAL STUDY")
        print("Complete Hyperparameter Optimization and Comparative Analysis")
        print("="*80)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        start_time = time.time()
        results = {
            'grid_search': False,
            'build_summary': False,
            'search_summary': False,
            'combined_analysis': False,
            'comparative_sift': False,
            'comparative_mnist': False
        }
        
        # Phase 1: Grid Search
        if self.run_grid_search(config_args):
            results['grid_search'] = True
            print("✓ Grid search completed successfully")
        else:
            print("❌ Grid search failed - stopping execution")
            self._print_final_summary(results, start_time)
            sys.exit(1)
        
        # Phase 2: Build Summary
        if self.generate_build_summary():
            results['build_summary'] = True
            print("✓ Build summary generated")
        else:
            print("⚠ Build summary generation failed (continuing)")
        
        # Phase 3: Search Summary
        if self.generate_search_summary():
            results['search_summary'] = True
            print("✓ Search summary generated")
        else:
            print("⚠ Search summary generation failed (continuing)")
        
        # Phase 4: Combined Analysis
        if self.generate_combined_analysis():
            results['combined_analysis'] = True
            print("✓ Combined analysis generated")
        else:
            print("⚠ Combined analysis generation failed (continuing)")
        
        # Phase 5: Comparative Analysis (SIFT)
        if self.run_comparative_analysis('sift'):
            results['comparative_sift'] = True
            print("✓ SIFT comparative analysis completed")
        else:
            print("⚠ SIFT comparative analysis failed (continuing)")
        
        # Phase 5: Comparative Analysis (MNIST)
        if self.run_comparative_analysis('mnist'):
            results['comparative_mnist'] = True
            print("✓ MNIST comparative analysis completed")
        else:
            print("⚠ MNIST comparative analysis failed (continuing)")
        
        self._print_final_summary(results, start_time)
    
    def _print_final_summary(self, results: dict, start_time: float) -> None:
        """Print final summary of execution."""
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("EXPERIMENTAL STUDY SUMMARY")
        print("="*80 + "\n")
        
        print("Execution Results:")
        print(f"  ✓ Grid Search: {'PASSED' if results['grid_search'] else 'FAILED'}")
        print(f"  ✓ Build Summary: {'PASSED' if results['build_summary'] else 'SKIPPED/FAILED'}")
        print(f"  ✓ Search Summary: {'PASSED' if results['search_summary'] else 'SKIPPED/FAILED'}")
        print(f"  ✓ Combined Analysis: {'PASSED' if results['combined_analysis'] else 'SKIPPED/FAILED'}")
        print(f"  ✓ Comparative Analysis (SIFT): {'PASSED' if results['comparative_sift'] else 'SKIPPED/FAILED'}")
        print(f"  ✓ Comparative Analysis (MNIST): {'PASSED' if results['comparative_mnist'] else 'SKIPPED/FAILED'}")
        
        print(f"\nTotal elapsed time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "="*80)
        print("Generated Reports:")
        print("="*80)
        
        reports_dir = self.experimental_study_dir / 'results'
        reports = [
            ('Grid Search Results', 'grid_search_results.json'),
            ('Build Summary', 'BUILD_SUMMARY.md'),
            ('Search Summary', 'SEARCH_SUMMARY.md'),
            ('Combined Analysis', 'COMBINED_ANALYSIS.md'),
            ('Comparative Analysis', 'COMPARISON_REPORT.md'),
        ]
        
        for report_name, file_path in reports:
            full_path = reports_dir / file_path
            if full_path.exists():
                print(f"  ✓ {report_name}: {full_path}")
            else:
                print(f"  ○ {report_name}: (not generated)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Neural-LSH Experimental Study Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full study with default parameters
  python run_experiments.py
  
  # Run with custom hyperparameter ranges (fewer experiments)
  python run_experiments.py --knn 10 15 --m 50 100 --epochs 25 50
  
  # Limit to 5 experiments for quick testing
  python run_experiments.py --max-experiments 5
        """
    )
    
    parser.add_argument('--knn', type=int, nargs='+', default=[10, 15, 20],
                       help='k-NN values to test (default: 10 15 20)')
    parser.add_argument('--m', type=int, nargs='+', default=[50, 100, 200],
                       help='Partition counts to test (default: 50 100 200)')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4],
                       help='MLP layer counts (default: 2 3 4)')
    parser.add_argument('--nodes', type=int, nargs='+', default=[32, 64, 128],
                       help='MLP nodes per layer (default: 32 64 128)')
    parser.add_argument('--epochs', type=int, nargs='+', default=[25, 50],
                       help='Training epochs (default: 25 50)')
    parser.add_argument('--data', type=str, default='sift', choices=['sift', 'mnist'],
                       help='Dataset to use (default: sift)')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Max experiments to run (None = all)')
    parser.add_argument('--skip-grid-search', action='store_true',
                       help='Skip grid search, only generate reports')
    
    args = parser.parse_args()
    
    orchestrator = ExperimentOrchestrator()
    
    if args.skip_grid_search:
        print("Skipping grid search, generating reports from existing results...")
        orchestrator.generate_build_summary()
        orchestrator.generate_search_summary()
        orchestrator.generate_combined_analysis()
        orchestrator.run_comparative_analysis('sift')
        orchestrator.run_comparative_analysis('mnist')
    else:
        config = {
            'knn': args.knn,
            'm': args.m,
            'layers': args.layers,
            'nodes': args.nodes,
            'epochs': args.epochs,
            'data': args.data,
            'max_experiments': args.max_experiments
        }
        
        orchestrator.run_full_study(config)


if __name__ == '__main__':
    main()
