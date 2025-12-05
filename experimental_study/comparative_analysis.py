"""
Comparative Analysis: Neural-LSH vs Old Methods
================================================
Compares the best Neural-LSH configuration against LSH, Hypercube, IVFFlat, and IVFPQ
from the old_results.

Generates:
  - Side-by-side performance comparison
  - Ranking tables
  - Performance improvement metrics
  - Visualization data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MethodResult:
    """Result metrics for a specific method and configuration."""
    method: str
    dataset: str
    config_name: str
    recall: float
    af: float  # approximation factor
    qps: float  # queries per second
    approx_time: float
    true_time: float
    
    def efficiency_score(self) -> float:
        """
        Compute overall efficiency score.
        Combines recall and QPS: recall * QPS (higher is better)
        """
        return self.recall * self.qps if self.qps > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'dataset': self.dataset,
            'config': self.config_name,
            'recall': self.recall,
            'af': self.af,
            'qps': self.qps,
            'approx_time_ms': self.approx_time * 1000,
            'true_time_ms': self.true_time * 1000,
            'efficiency_score': self.efficiency_score()
        }


class ComparativeAnalyzer:
    """Analyzes and compares Neural-LSH with old methods."""
    
    def __init__(self, 
                 workspace_dir: str = '/Users/theodora/Desktop/uni/project/Neural-LSH',
                 old_results_file: str = 'old_results/REPORT_TESTS_SUMMARY.json',
                 nlsh_results_file: str = 'experimental_study/grid_search_results.json'):
        self.workspace_dir = Path(workspace_dir)
        self.old_results_file = self.workspace_dir / old_results_file
        self.nlsh_results_file = self.workspace_dir / nlsh_results_file
        self.old_results = self._load_old_results()
        self.nlsh_results = self._load_nlsh_results()
    
    def _load_old_results(self) -> Dict[str, Dict]:
        """Load old results from JSON file."""
        if not self.old_results_file.exists():
            print(f"⚠ Old results file not found: {self.old_results_file}")
            return {}
        
        with open(self.old_results_file, 'r') as f:
            return json.load(f)
    
    def _load_nlsh_results(self) -> Dict:
        """Load Neural-LSH grid search results."""
        if not self.nlsh_results_file.exists():
            print(f"⚠ Neural-LSH results file not found: {self.nlsh_results_file}")
            return {}
        
        with open(self.nlsh_results_file, 'r') as f:
            return json.load(f)
    
    def get_best_nlsh_configs(self, dataset: str = 'sift', top_k: int = 5) -> List[Dict]:
        """
        Get best Neural-LSH configurations for a specific dataset.
        
        Args:
            dataset: 'sift' or 'mnist'
            top_k: Return top K configurations
            
        Returns:
            List of top configurations sorted by recall
        """
        if not self.nlsh_results.get('results'):
            print(f"No Neural-LSH results available")
            return []
        
        results = self.nlsh_results['results']
        dataset_results = [r for r in results if dataset in r['config']['data_type']]
        
        # Sort by recall
        sorted_results = sorted(dataset_results, 
                               key=lambda r: r['avg_recall'], 
                               reverse=True)
        
        return sorted_results[:top_k]
    
    def get_best_old_method_configs(self, method: str, dataset: str = 'sift', 
                                   top_k: int = 5) -> List[MethodResult]:
        """
        Get best configurations for a specific old method and dataset.
        
        Args:
            method: 'lsh', 'hypercube', 'ivfflat', or 'ivfpq'
            dataset: 'sift' or 'mnist'
            top_k: Return top K configurations
            
        Returns:
            List of top configurations sorted by recall
        """
        if method not in self.old_results:
            print(f"⚠ Method '{method}' not found in old results")
            return []
        
        configs = self.old_results[method]
        method_results = []
        
        for config_name, metrics in configs.items():
            if dataset not in config_name:
                continue
            
            result = MethodResult(
                method=method,
                dataset=dataset,
                config_name=config_name,
                recall=metrics.get('recall', 0),
                af=metrics.get('af', 1.0),
                qps=metrics.get('qps', 0),
                approx_time=metrics.get('approx_time', 0),
                true_time=metrics.get('true_time', 0)
            )
            method_results.append(result)
        
        # Sort by recall
        sorted_results = sorted(method_results, key=lambda r: r.recall, reverse=True)
        return sorted_results[:top_k]
    
    def compare_best_configs(self, dataset: str = 'sift') -> Dict:
        """
        Compare the best Neural-LSH configuration against the best of each old method.
        
        Returns:
            Dictionary with comparison metrics
        """
        print(f"\n{'='*80}")
        print(f"Comparative Analysis: Neural-LSH vs Other Methods ({dataset.upper()})")
        print(f"{'='*80}\n")
        
        # Get best Neural-LSH
        nlsh_configs = self.get_best_nlsh_configs(dataset, top_k=1)
        if not nlsh_configs:
            print("❌ No Neural-LSH results available")
            return {}
        
        best_nlsh = nlsh_configs[0]
        nlsh_result = MethodResult(
            method='Neural-LSH',
            dataset=dataset,
            config_name=best_nlsh['config'].get('data_type', ''),
            recall=best_nlsh['avg_recall'],
            af=best_nlsh['avg_af'],
            qps=best_nlsh['avg_qps'],
            approx_time=best_nlsh['search_time'] / best_nlsh['n_queries'] if best_nlsh['n_queries'] > 0 else 0,
            true_time=0  # Not directly available
        )
        
        results = {
            'dataset': dataset,
            'neural_lsh': nlsh_result.to_dict(),
            'comparisons': {}
        }
        
        # Compare with each old method
        for method in ['lsh', 'hypercube', 'ivfflat', 'ivfpq']:
            best_configs = self.get_best_old_method_configs(method, dataset, top_k=1)
            if not best_configs:
                continue
            
            old_result = best_configs[0]
            
            # Calculate improvements
            recall_improvement = (nlsh_result.recall - old_result.recall) / old_result.recall * 100 if old_result.recall > 0 else 0
            qps_improvement = (nlsh_result.qps - old_result.qps) / old_result.qps * 100 if old_result.qps > 0 else 0
            efficiency_improvement = (nlsh_result.efficiency_score() - old_result.efficiency_score()) / old_result.efficiency_score() * 100 if old_result.efficiency_score() > 0 else 0
            
            results['comparisons'][method] = {
                'best_config': old_result.to_dict(),
                'improvements': {
                    'recall': recall_improvement,
                    'qps': qps_improvement,
                    'efficiency_score': efficiency_improvement
                }
            }
        
        return results
    
    def generate_comparison_report(self, output_file: Path = None) -> None:
        """Generate comprehensive comparison report in Markdown format."""
        if output_file is None:
            output_file = self.workspace_dir / 'experimental_study' / 'COMPARISON_REPORT.md'
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Neural-LSH Comparative Analysis Report\n\n")
            f.write(f"**Generated**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Analyze each dataset
            for dataset in ['sift', 'mnist']:
                comparison = self.compare_best_configs(dataset)
                if not comparison:
                    continue
                
                f.write(f"\n## {dataset.upper()} Dataset Analysis\n\n")
                
                # Neural-LSH best config
                nlsh = comparison['neural_lsh']
                f.write(f"### Best Neural-LSH Configuration\n\n")
                f.write(f"- **Recall**: {nlsh['recall']:.4f}\n")
                f.write(f"- **Approximation Factor**: {nlsh['af']:.4f}\n")
                f.write(f"- **Queries/Second**: {nlsh['qps']:.2f}\n")
                f.write(f"- **Efficiency Score**: {nlsh['efficiency_score']:.2f}\n\n")
                
                # Comparisons with each method
                f.write(f"### Performance Comparison Table\n\n")
                f.write("| Method | Recall | AF | QPS | Efficiency Score | Recall Improvement | QPS Improvement |\n")
                f.write("|--------|--------|----|----|---|---|---|\n")
                f.write(f"| **Neural-LSH** | **{nlsh['recall']:.4f}** | **{nlsh['af']:.4f}** | **{nlsh['qps']:.2f}** | **{nlsh['efficiency_score']:.2f}** | - | - |\n")
                
                for method, comp_data in comparison['comparisons'].items():
                    old = comp_data['best_config']
                    imp = comp_data['improvements']
                    
                    recall_arrow = "↑" if imp['recall'] > 0 else "↓"
                    qps_arrow = "↑" if imp['qps'] > 0 else "↓"
                    
                    f.write(f"| {method.upper()} | {old['recall']:.4f} | {old['af']:.4f} | {old['qps']:.2f} | {old['efficiency_score']:.2f} | {recall_arrow} {abs(imp['recall']):.1f}% | {qps_arrow} {abs(imp['qps']):.1f}% |\n")
                
                # Detailed comparison text
                f.write(f"\n### Detailed Analysis\n\n")
                for method, comp_data in comparison['comparisons'].items():
                    old = comp_data['best_config']
                    imp = comp_data['improvements']
                    
                    f.write(f"\n#### vs {method.upper()}\n\n")
                    f.write(f"**Configuration**: {old['config']}\n\n")
                    f.write(f"**Performance Metrics**:\n")
                    f.write(f"- Recall: Neural-LSH {nlsh['recall']:.4f} vs {method.upper()} {old['recall']:.4f} ({imp['recall']:+.1f}%)\n")
                    f.write(f"- QPS: Neural-LSH {nlsh['qps']:.2f} vs {method.upper()} {old['qps']:.2f} ({imp['qps']:+.1f}%)\n")
                    f.write(f"- Efficiency Score: Neural-LSH {nlsh['efficiency_score']:.2f} vs {method.upper()} {old['efficiency_score']:.2f} ({imp['efficiency_score']:+.1f}%)\n\n")
        
        print(f"✓ Comparison report saved to {output_file}")


def main():
    """Run comparative analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural-LSH Comparative Analysis')
    parser.add_argument('--dataset', type=str, default='sift', choices=['sift', 'mnist'],
                       help='Dataset to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for report')
    
    args = parser.parse_args()
    
    analyzer = ComparativeAnalyzer()
    
    if args.output:
        analyzer.generate_comparison_report(Path(args.output))
    else:
        analyzer.generate_comparison_report()
    
    # Also generate dataset-specific comparisons
    comparison = analyzer.compare_best_configs(args.dataset)
    
    if comparison:
        print(f"\n{'='*80}")
        print(f"SUMMARY: {args.dataset.upper()} Dataset")
        print(f"{'='*80}\n")
        
        nlsh = comparison['neural_lsh']
        print(f"Best Neural-LSH Configuration:")
        print(f"  - Recall: {nlsh['recall']:.4f}")
        print(f"  - AF: {nlsh['af']:.4f}")
        print(f"  - QPS: {nlsh['qps']:.2f}")
        print(f"  - Efficiency Score: {nlsh['efficiency_score']:.2f}\n")
        
        print(f"Comparison with Old Methods:")
        for method, comp_data in comparison['comparisons'].items():
            imp = comp_data['improvements']
            print(f"  {method.upper():10s}: Recall {imp['recall']:+6.1f}%, QPS {imp['qps']:+6.1f}%")


if __name__ == '__main__':
    main()
