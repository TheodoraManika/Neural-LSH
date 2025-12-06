"""
Summary Report Generator for Neural-LSH
========================================
Generates comprehensive summary reports for:
  1. Build phase metrics (index construction statistics)
  2. Search phase metrics (query performance)
  3. Combined analysis with cost-benefit trade-offs
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from dataclasses import dataclass
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BuildMetrics:
    """Metrics from the index building phase."""
    config_name: str
    build_time: float  # seconds
    index_size: float  # MB
    num_points: int
    num_partitions: int
    
    @property
    def index_efficiency(self) -> float:
        """Index size per 1000 points (KB/1000 points)."""
        return (self.index_size * 1024) / (self.num_points / 1000) if self.num_points > 0 else 0
    
    @property
    def build_time_per_point(self) -> float:
        """Build time per point (microseconds/point)."""
        return (self.build_time * 1e6) / self.num_points if self.num_points > 0 else 0


@dataclass
class SearchMetrics:
    """Metrics from the search phase."""
    config_name: str
    avg_recall: float
    avg_af: float
    avg_qps: float  # queries per second
    search_time: float  # seconds for all queries
    num_queries: int
    
    @property
    def time_per_query(self) -> float:
        """Average time per query (milliseconds)."""
        return (self.search_time * 1000) / self.num_queries if self.num_queries > 0 else 0
    
    @property
    def efficiency_score(self) -> float:
        """Efficiency: recall * QPS (higher is better)."""
        return self.avg_recall * self.avg_qps


class SummaryReportGenerator:
    """Generates comprehensive summary reports."""
    
    def __init__(self, workspace_dir: str = None,
                 results_file: str = 'experimental_study/grid_search_results.json'):
        # Use current directory if not specified
        if workspace_dir is None:
            workspace_dir = str(Path.cwd())
        self.workspace_dir = Path(workspace_dir)
        self.results_file = self.workspace_dir / results_file
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """Load grid search results."""
        if not self.results_file.exists():
            print(f"⚠ Results file not found: {self.results_file}")
            return {}
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def extract_build_metrics(self) -> List[BuildMetrics]:
        """Extract build metrics from all results."""
        metrics = []
        
        for result in self.results.get('results', []):
            config = result['config']
            metric = BuildMetrics(
                config_name=f"k{config['knn']}_m{config['m']}_l{config['layers']}_n{config['nodes']}_e{config['epochs']}",
                build_time=result['build_time'],
                index_size=result['index_size'],
                num_points=100000,  # SIFT has 1M points, but we typically use 100K for experiments
                num_partitions=config['m']
            )
            metrics.append(metric)
        
        return metrics
    
    def extract_search_metrics(self) -> List[SearchMetrics]:
        """Extract search metrics from all results."""
        metrics = []
        
        for result in self.results.get('results', []):
            config = result['config']
            metric = SearchMetrics(
                config_name=f"k{config['knn']}_m{config['m']}_l{config['layers']}_n{config['nodes']}_e{config['epochs']}",
                avg_recall=result['avg_recall'],
                avg_af=result['avg_af'],
                avg_qps=result['avg_qps'],
                search_time=result['search_time'],
                num_queries=result['n_queries']
            )
            metrics.append(metric)
        
        return metrics
    
    def generate_build_summary(self, output_file: Path = None) -> str:
        """Generate build phase summary report."""
        if output_file is None:
            output_file = self.workspace_dir / 'experimental_study' / 'BUILD_SUMMARY.md'
        
        metrics = self.extract_build_metrics()
        if not metrics:
            return "No build metrics available"
        
        # Sort by build time
        sorted_metrics = sorted(metrics, key=lambda m: m.build_time)
        
        build_times = [m.build_time for m in metrics]
        index_sizes = [m.index_size for m in metrics]
        
        with open(output_file, 'w') as f:
            f.write("# Neural-LSH Build Phase Summary Report\n\n")
            f.write(f"**Generated**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Configurations**: {len(metrics)}\n\n")
            
            # Statistics
            f.write("## Build Time Statistics\n\n")
            f.write(f"- **Minimum**: {min(build_times):.2f}s\n")
            f.write(f"- **Maximum**: {max(build_times):.2f}s\n")
            f.write(f"- **Mean**: {statistics.mean(build_times):.2f}s\n")
            f.write(f"- **Median**: {statistics.median(build_times):.2f}s\n")
            f.write(f"- **Std Dev**: {statistics.stdev(build_times) if len(build_times) > 1 else 0:.2f}s\n\n")
            
            f.write("## Index Size Statistics\n\n")
            f.write(f"- **Minimum**: {min(index_sizes):.2f} MB\n")
            f.write(f"- **Maximum**: {max(index_sizes):.2f} MB\n")
            f.write(f"- **Mean**: {statistics.mean(index_sizes):.2f} MB\n")
            f.write(f"- **Median**: {statistics.median(index_sizes):.2f} MB\n")
            f.write(f"- **Std Dev**: {statistics.stdev(index_sizes) if len(index_sizes) > 1 else 0:.2f} MB\n\n")
            
            # Fastest builds
            f.write("## Fastest Build Configurations\n\n")
            f.write("| Rank | Configuration | Build Time (s) | Index Size (MB) | Time/Point (µs) |\n")
            f.write("|------|---|---|---|---|\n")
            
            for rank, metric in enumerate(sorted_metrics[:10], 1):
                f.write(f"| {rank} | {metric.config_name} | {metric.build_time:.2f} | "
                       f"{metric.index_size:.2f} | {metric.build_time_per_point:.2f} |\n")
            
            # Smallest indexes
            f.write("\n## Smallest Index Configurations\n\n")
            sorted_by_size = sorted(metrics, key=lambda m: m.index_size)
            f.write("| Rank | Configuration | Index Size (MB) | Build Time (s) | Size Ratio |\n")
            f.write("|------|---|---|---|---|\n")
            
            min_size = min(index_sizes)
            for rank, metric in enumerate(sorted_by_size[:10], 1):
                f.write(f"| {rank} | {metric.config_name} | {metric.index_size:.2f} | "
                       f"{metric.build_time:.2f} | {metric.index_size/min_size:.2f}x |\n")
            
            # All configurations detailed view
            f.write("\n## All Configurations\n\n")
            f.write("| Configuration | Build Time (s) | Index Size (MB) | Efficiency (KB/1k pts) |\n")
            f.write("|---|---|---|---|\n")
            
            for metric in sorted_metrics:
                f.write(f"| {metric.config_name} | {metric.build_time:.2f} | "
                       f"{metric.index_size:.2f} | {metric.index_efficiency:.2f} |\n")
        
        print(f"✓ Build summary saved to {output_file}")
        return str(output_file)
    
    def generate_search_summary(self, output_file: Path = None) -> str:
        """Generate search phase summary report."""
        if output_file is None:
            output_file = self.workspace_dir / 'experimental_study' / 'SEARCH_SUMMARY.md'
        
        metrics = self.extract_search_metrics()
        if not metrics:
            return "No search metrics available"
        
        # Sort by recall
        sorted_metrics = sorted(metrics, key=lambda m: m.avg_recall, reverse=True)
        
        recalls = [m.avg_recall for m in metrics]
        qps_values = [m.avg_qps for m in metrics]
        efficiency_scores = [m.efficiency_score for m in metrics]
        
        with open(output_file, 'w') as f:
            f.write("# Neural-LSH Search Phase Summary Report\n\n")
            f.write(f"**Generated**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Configurations**: {len(metrics)}\n")
            f.write(f"**Total Queries Executed**: {sum(m.num_queries for m in metrics)}\n\n")
            
            # Statistics
            f.write("## Recall Statistics\n\n")
            f.write(f"- **Minimum**: {min(recalls):.4f}\n")
            f.write(f"- **Maximum**: {max(recalls):.4f}\n")
            f.write(f"- **Mean**: {statistics.mean(recalls):.4f}\n")
            f.write(f"- **Median**: {statistics.median(recalls):.4f}\n")
            f.write(f"- **Std Dev**: {statistics.stdev(recalls) if len(recalls) > 1 else 0:.4f}\n\n")
            
            f.write("## Query Performance (QPS) Statistics\n\n")
            f.write(f"- **Minimum**: {min(qps_values):.2f} QPS\n")
            f.write(f"- **Maximum**: {max(qps_values):.2f} QPS\n")
            f.write(f"- **Mean**: {statistics.mean(qps_values):.2f} QPS\n")
            f.write(f"- **Median**: {statistics.median(qps_values):.2f} QPS\n\n")
            
            f.write("## Efficiency Score Statistics (Recall × QPS)\n\n")
            f.write(f"- **Minimum**: {min(efficiency_scores):.2f}\n")
            f.write(f"- **Maximum**: {max(efficiency_scores):.2f}\n")
            f.write(f"- **Mean**: {statistics.mean(efficiency_scores):.2f}\n\n")
            
            # Best recall configurations
            f.write("## Best Recall Configurations\n\n")
            f.write("| Rank | Configuration | Recall | QPS | AF | Time/Query (ms) | Efficiency |\n")
            f.write("|------|---|---|---|---|---|---|\n")
            
            for rank, metric in enumerate(sorted_metrics[:10], 1):
                f.write(f"| {rank} | {metric.config_name} | {metric.avg_recall:.4f} | "
                       f"{metric.avg_qps:.2f} | {metric.avg_af:.4f} | {metric.time_per_query:.2f} | "
                       f"{metric.efficiency_score:.2f} |\n")
            
            # Fastest queries
            f.write("\n## Fastest Query Configurations\n\n")
            sorted_by_speed = sorted(metrics, key=lambda m: m.avg_qps, reverse=True)
            f.write("| Rank | Configuration | QPS | Recall | AF | Efficiency |\n")
            f.write("|------|---|---|---|---|---|\n")
            
            for rank, metric in enumerate(sorted_by_speed[:10], 1):
                f.write(f"| {rank} | {metric.config_name} | {metric.avg_qps:.2f} | "
                       f"{metric.avg_recall:.4f} | {metric.avg_af:.4f} | {metric.efficiency_score:.2f} |\n")
            
            # Best efficiency (recall × QPS)
            f.write("\n## Best Efficiency Configurations\n\n")
            sorted_by_efficiency = sorted(metrics, key=lambda m: m.efficiency_score, reverse=True)
            f.write("| Rank | Configuration | Recall | QPS | Efficiency | AF |\n")
            f.write("|------|---|---|---|---|---|\n")
            
            for rank, metric in enumerate(sorted_by_efficiency[:10], 1):
                f.write(f"| {rank} | {metric.config_name} | {metric.avg_recall:.4f} | "
                       f"{metric.avg_qps:.2f} | {metric.efficiency_score:.2f} | {metric.avg_af:.4f} |\n")
        
        print(f"✓ Search summary saved to {output_file}")
        return str(output_file)
    
    def generate_combined_report(self, output_file: Path = None) -> str:
        """Generate combined build + search analysis."""
        if output_file is None:
            output_file = self.workspace_dir / 'experimental_study' / 'COMBINED_ANALYSIS.md'
        
        build_metrics = self.extract_build_metrics()
        search_metrics = self.extract_search_metrics()
        
        if not build_metrics or not search_metrics:
            return "Insufficient data for combined analysis"
        
        # Create lookup for easier correlation
        build_dict = {m.config_name: m for m in build_metrics}
        search_dict = {m.config_name: m for m in search_metrics}
        
        with open(output_file, 'w') as f:
            f.write("# Neural-LSH Combined Build + Search Analysis\n\n")
            f.write(f"**Generated**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Configurations Analyzed**: {len(build_metrics)}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes the cost-benefit trade-offs between index construction time/size and query performance.\n\n")
            
            # Create Pareto frontier analysis
            f.write("## Pareto Efficiency Analysis\n\n")
            f.write("Identifying configurations that are not dominated in both dimensions:\n\n")
            
            # Pareto frontier: fast build AND high recall
            f.write("### Fast Build with High Recall\n\n")
            f.write("Configurations with low build time and high recall (good for hot deployment):\n\n")
            f.write("| Configuration | Build Time (s) | Recall | QPS | Efficiency |\n")
            f.write("|---|---|---|---|---|\n")
            
            for build in sorted(build_metrics, key=lambda m: m.build_time)[:5]:
                search = search_dict.get(build.config_name)
                if search:
                    f.write(f"| {build.config_name} | {build.build_time:.2f} | "
                           f"{search.avg_recall:.4f} | {search.avg_qps:.2f} | {search.efficiency_score:.2f} |\n")
            
            # Pareto frontier: compact index with high recall
            f.write("\n### Compact Index with High Recall\n\n")
            f.write("Configurations with small index size and high recall (good for storage-constrained systems):\n\n")
            f.write("| Configuration | Index Size (MB) | Recall | QPS | Efficiency |\n")
            f.write("|---|---|---|---|---|\n")
            
            for build in sorted(build_metrics, key=lambda m: m.index_size)[:5]:
                search = search_dict.get(build.config_name)
                if search:
                    f.write(f"| {build.config_name} | {build.index_size:.2f} | "
                           f"{search.avg_recall:.4f} | {search.avg_qps:.2f} | {search.efficiency_score:.2f} |\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            best_recall = max(search_metrics, key=lambda m: m.avg_recall)
            best_speed = max(search_metrics, key=lambda m: m.avg_qps)
            best_efficiency = max(search_metrics, key=lambda m: m.efficiency_score)
            
            build_best_recall = build_dict[best_recall.config_name]
            build_best_speed = build_dict[best_speed.config_name]
            build_best_efficiency = build_dict[best_efficiency.config_name]
            
            f.write(f"**For Maximum Recall**:\n")
            f.write(f"- Configuration: {best_recall.config_name}\n")
            f.write(f"- Recall: {best_recall.avg_recall:.4f}\n")
            f.write(f"- Build Time: {build_best_recall.build_time:.2f}s\n")
            f.write(f"- Index Size: {build_best_recall.index_size:.2f}MB\n")
            f.write(f"- QPS: {best_recall.avg_qps:.2f}\n\n")
            
            f.write(f"**For Maximum Query Speed**:\n")
            f.write(f"- Configuration: {best_speed.config_name}\n")
            f.write(f"- QPS: {best_speed.avg_qps:.2f}\n")
            f.write(f"- Recall: {best_speed.avg_recall:.4f}\n")
            f.write(f"- Build Time: {build_best_speed.build_time:.2f}s\n")
            f.write(f"- Index Size: {build_best_speed.index_size:.2f}MB\n\n")
            
            f.write(f"**For Best Efficiency Score (Recall × QPS)**:\n")
            f.write(f"- Configuration: {best_efficiency.config_name}\n")
            f.write(f"- Efficiency Score: {best_efficiency.efficiency_score:.2f}\n")
            f.write(f"- Recall: {best_efficiency.avg_recall:.4f}\n")
            f.write(f"- QPS: {best_efficiency.avg_qps:.2f}\n")
            f.write(f"- Build Time: {build_best_efficiency.build_time:.2f}s\n")
            f.write(f"- Index Size: {build_best_efficiency.index_size:.2f}MB\n")
        
        print(f"✓ Combined analysis saved to {output_file}")
        return str(output_file)


def main():
    """Generate all summary reports."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural-LSH Summary Report Generator')
    parser.add_argument('--build', action='store_true', help='Generate build summary')
    parser.add_argument('--search', action='store_true', help='Generate search summary')
    parser.add_argument('--combined', action='store_true', help='Generate combined analysis')
    parser.add_argument('--all', action='store_true', help='Generate all reports')
    
    args = parser.parse_args()
    
    # Default to all if no options specified
    if not any([args.build, args.search, args.combined]):
        args.all = True
    
    generator = SummaryReportGenerator()
    
    if args.build or args.all:
        generator.generate_build_summary()
    
    if args.search or args.all:
        generator.generate_search_summary()
    
    if args.combined or args.all:
        generator.generate_combined_report()
    
    print("\n✓ All summary reports generated successfully!")


if __name__ == '__main__':
    main()
