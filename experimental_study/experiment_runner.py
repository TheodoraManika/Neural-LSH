"""
Experimental Study Runner for Neural-LSH
==========================================
Orchestrates hyperparameter optimization experiments for Neural-LSH.

Experiments cover:
  1. MLP hyperparameters (layers, nodes, batch_size, learning_rate)
  2. KaHIP partitioning (m - number of buckets, imbalance factor)
  3. Graph construction (k - neighbors for k-NN graph)
  4. Training (epochs)
  
Outputs:
  - Build time and index metrics for each configuration
  - Search performance metrics (recall, QPS, AF, etc.)
  - Summary reports comparing all configurations
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
import numpy as np
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.args import parse_and_validate_build_args, parse_and_validate_search_args


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    # Graph parameters
    knn: int
    # Partitioning parameters
    m: int
    # MLP parameters
    layers: int
    nodes: int
    batch_size: int
    learning_rate: float
    epochs: int
    # Optional parameters with defaults
    imbalance: float = 0.03
    kahip_mode: int = 2
    # Data
    data_type: str = 'sift'
    data_file: str = 'dataset/sift_base.fvecs'
    
    def get_index_name(self) -> str:
        """Generate a unique index name from configuration."""
        return f"nlsh_{self.data_type}_k{self.knn}_m{self.m}_l{self.layers}_n{self.nodes}_e{self.epochs}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """Result metrics from a single experiment."""
    config: ExperimentConfig
    build_time: float
    index_size: float  # in MB
    search_time: float  # for all queries
    avg_recall: float
    avg_af: float  # approximation factor
    avg_qps: float  # queries per second
    n_queries: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result


class ExperimentRunner:
    """Manages experimental studies for Neural-LSH hyperparameter optimization."""
    
    def __init__(self, workspace_dir: str = None,
                 results_dir: str = 'experimental_study/results'):
        # Use current directory if not specified
        if workspace_dir is None:
            workspace_dir = str(Path.cwd())
        self.workspace_dir = Path(workspace_dir)
        self.results_dir = self.workspace_dir / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        
    def run_experiment(self, config: ExperimentConfig, 
                      query_file: str = 'dataset/sift_query.fvecs',
                      output_file: str = 'experimental_results.pkl') -> ExperimentResult:
        """
        Execute a single experiment: build index and run search.
        
        Returns:
            ExperimentResult with metrics from both build and search phases
        """
        print(f"\n{'='*80}")
        print(f"Running Experiment: {config.get_index_name()}")
        print(f"{'='*80}")
        print(f"Config: {config.to_dict()}")
        
        index_name = config.get_index_name()
        # Store index in cache/index_model folder
        index_cache_dir = Path.cwd() / 'cache' / 'index_model'
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = str(index_cache_dir / f'{index_name}.pkl')
        
        # ===== BUILD PHASE =====
        print(f"\n[BUILD PHASE]")
        build_cmd = [
            'python', 'src/nlsh_build.py',
            '-d', config.data_file,
            '-i', index_path,
            '-type', config.data_type,
            '--knn', str(config.knn),
            '-m', str(config.m),
            '--imbalance', str(config.imbalance),
            '--kahip_mode', str(config.kahip_mode),
            '--layers', str(config.layers),
            '--nodes', str(config.nodes),
            '--batch_size', str(config.batch_size),
            '--lr', str(config.learning_rate),
            '--epochs', str(config.epochs),
        ]
        
        build_start = time.time()
        try:
            result = subprocess.run(build_cmd, cwd=str(self.workspace_dir), 
                                   capture_output=True, text=True, timeout=3600)
            build_time = time.time() - build_start
            
            if result.returncode != 0:
                print(f"❌ Build failed for {index_name}")
                print(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Build failed: {result.stderr}")
            
            # Get index file size
            if os.path.exists(index_path):
                index_size = os.path.getsize(index_path) / (1024 * 1024)  # Convert to MB
                print(f"✓ Build completed in {build_time:.2f}s")
                print(f"  Index size: {index_size:.2f} MB")
            else:
                raise FileNotFoundError(f"Index file not created: {index_path}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ Build timeout for {index_name}")
            raise
        except Exception as e:
            print(f"❌ Build error: {e}")
            raise
        
        # ===== SEARCH PHASE =====
        print(f"\n[SEARCH PHASE]")
        search_output = str(self.results_dir / f"{index_name}_search.json")
        search_cmd = [
            'python', 'src/nlsh_search.py',
            '-d', config.data_file,
            '-q', query_file,
            '-i', index_path,
            '-o', search_output,
            '-type', config.data_type,  # Add data type
            '-N', '10',  # Return top 10 neighbors
            '-T', '2',   # Explore only 2 buckets for speed (was 5)
        ]
        
        search_start = time.time()
        try:
            result = subprocess.run(search_cmd, cwd=str(self.workspace_dir),
                                   capture_output=True, text=True, timeout=3600)
            search_time = time.time() - search_start
            
            if result.returncode != 0:
                print(f"❌ Search failed for {index_name}")
                print(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Search failed: {result.stderr}")
            
            print(f"✓ Search completed in {search_time:.2f}s")
            
            # Parse search results
            if os.path.exists(search_output):
                with open(search_output, 'r') as f:
                    search_results = json.load(f)
                
                # Extract metrics
                avg_recall = np.mean([r.get('recall', 0) for r in search_results])
                avg_af = np.mean([r.get('af', 1.0) for r in search_results])
                avg_qps = len(search_results) / search_time if search_time > 0 else 0
                
            else:
                print(f"⚠ Search output file not found: {search_output}")
                avg_recall = avg_af = avg_qps = 0
                search_results = []
            
        except subprocess.TimeoutExpired:
            print(f"❌ Search timeout for {index_name}")
            raise
        except Exception as e:
            print(f"❌ Search error: {e}")
            raise
        
        # ===== CREATE RESULT =====
        experiment_result = ExperimentResult(
            config=config,
            build_time=build_time,
            index_size=index_size,
            search_time=search_time,
            avg_recall=avg_recall,
            avg_af=avg_af,
            avg_qps=avg_qps,
            n_queries=len(search_results)
        )
        
        self.results.append(experiment_result)
        return experiment_result
    
    def run_grid_search(self, 
                       knn_values: List[int] = [10, 15, 20],
                       m_values: List[int] = [50, 100, 200],
                       layers_values: List[int] = [2, 3, 4],
                       nodes_values: List[int] = [32, 64, 128],
                       epochs_values: List[int] = [25, 50],
                       data_type: str = 'sift',
                       data_file: str = 'dataset/sift_base.fvecs',
                       query_file: str = 'dataset/sift_query.fvecs',
                       max_experiments: int = None) -> None:
        """
        Run grid search over hyperparameter space.
        
        Args:
            knn_values: k-NN graph parameter values
            m_values: Number of partitions
            layers_values: MLP layer counts
            nodes_values: MLP hidden node counts
            epochs_values: Training epochs
            data_type: 'sift' or 'mnist'
            data_file: Path to data file
            query_file: Path to query file
            max_experiments: Limit total experiments (None = run all)
        """
        # Fixed parameters for experiments
        batch_size = 32
        learning_rate = 0.001
        imbalance = 0.03
        kahip_mode = 2
        
        # Generate all combinations
        configs = []
        for knn, m, layers, nodes, epochs in product(
            knn_values, m_values, layers_values, nodes_values, epochs_values
        ):
            config = ExperimentConfig(
                knn=knn, m=m, imbalance=imbalance, kahip_mode=kahip_mode,
                layers=layers, nodes=nodes, batch_size=batch_size,
                learning_rate=learning_rate, epochs=epochs,
                data_type=data_type, data_file=data_file
            )
            configs.append(config)
        
        if max_experiments:
            configs = configs[:max_experiments]
        
        print(f"\n{'='*80}")
        print(f"Starting Grid Search: {len(configs)} configurations")
        print(f"{'='*80}\n")
        
        failed_configs = []
        for i, config in enumerate(configs, 1):
            try:
                print(f"\n[{i}/{len(configs)}]")
                self.run_experiment(config, query_file)
            except Exception as e:
                print(f"⚠ Experiment failed: {e}")
                failed_configs.append((config, str(e)))
                continue
        
        # Save all results
        self.save_results_summary()
        
        if failed_configs:
            print(f"\n⚠ {len(failed_configs)} experiments failed:")
            for config, error in failed_configs:
                print(f"  - {config.get_index_name()}: {error}")
    
    def save_results_summary(self) -> None:
        """Save comprehensive results summary to JSON and Markdown."""
        if not self.results:
            print("No results to save")
            return
        
        # Sort by recall (best first)
        sorted_results = sorted(self.results, key=lambda r: r.avg_recall, reverse=True)
        
        # JSON summary
        json_file = self.results_dir / 'grid_search_results.json'
        json_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_experiments': len(self.results),
            'best_recall': sorted_results[0].avg_recall if sorted_results else 0,
            'results': [r.to_dict() for r in sorted_results]
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n✓ Results saved to {json_file}")
        
        # Markdown summary
        md_file = self.results_dir / 'grid_search_summary.md'
        self._write_markdown_summary(sorted_results, md_file)
        print(f"✓ Summary saved to {md_file}")
    
    def _write_markdown_summary(self, results: List[ExperimentResult], 
                               output_file: Path) -> None:
        """Write markdown summary of results."""
        with open(output_file, 'w') as f:
            f.write("# Neural-LSH Grid Search Results\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Experiments**: {len(results)}\n\n")
            
            # Top configurations table
            f.write("## Top 10 Configurations by Recall\n\n")
            f.write("| Rank | Configuration | Recall | AF | QPS | Build Time | Index Size |\n")
            f.write("|------|---|---|---|---|---|---|\n")
            
            for rank, result in enumerate(results[:10], 1):
                config_str = (f"k={result.config.knn}, m={result.config.m}, "
                             f"l={result.config.layers}, n={result.config.nodes}, "
                             f"e={result.config.epochs}")
                f.write(f"| {rank} | {config_str} | {result.avg_recall:.3f} | "
                       f"{result.avg_af:.3f} | {result.avg_qps:.2f} | "
                       f"{result.build_time:.1f}s | {result.index_size:.1f}MB |\n")
            
            # Detailed results
            f.write("\n## All Results\n\n")
            for rank, result in enumerate(results, 1):
                f.write(f"\n### {rank}. {result.config.get_index_name()}\n\n")
                f.write(f"**Configuration**:\n")
                f.write(f"  - k-NN graph: k={result.config.knn}\n")
                f.write(f"  - Partitioning: m={result.config.m}, imbalance={result.config.imbalance}\n")
                f.write(f"  - MLP: {result.config.layers} layers, {result.config.nodes} nodes/layer\n")
                f.write(f"  - Training: {result.config.epochs} epochs, batch_size={result.config.batch_size}\n\n")
                f.write(f"**Performance**:\n")
                f.write(f"  - Recall: {result.avg_recall:.4f}\n")
                f.write(f"  - Approximation Factor: {result.avg_af:.4f}\n")
                f.write(f"  - QPS: {result.avg_qps:.2f}\n")
                f.write(f"  - Build Time: {result.build_time:.2f}s\n")
                f.write(f"  - Index Size: {result.index_size:.2f}MB\n")
                f.write(f"  - Search Time: {result.search_time:.2f}s ({result.n_queries} queries)\n")


def main():
    """Run grid search experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural-LSH Hyperparameter Grid Search')
    parser.add_argument('--knn', type=int, nargs='+', default=[10, 15, 20],
                       help='k-NN values to test')
    parser.add_argument('--m', type=int, nargs='+', default=[50, 100, 200],
                       help='Number of partitions (m) to test')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4],
                       help='MLP layer counts to test')
    parser.add_argument('--nodes', type=int, nargs='+', default=[32, 64, 128],
                       help='MLP node counts per layer to test')
    parser.add_argument('--epochs', type=int, nargs='+', default=[25, 50],
                       help='Training epoch counts to test')
    parser.add_argument('--data', type=str, default='sift',
                       help='Dataset: sift or mnist')
    parser.add_argument('--data-file', type=str, default='dataset/sift_base.fvecs',
                       help='Path to data file')
    parser.add_argument('--query-file', type=str, default='dataset/sift_query.fvecs',
                       help='Path to query file')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Maximum number of experiments to run')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    runner.run_grid_search(
        knn_values=args.knn,
        m_values=args.m,
        layers_values=args.layers,
        nodes_values=args.nodes,
        epochs_values=args.epochs,
        data_type=args.data,
        data_file=args.data_file,
        query_file=args.query_file,
        max_experiments=args.max_experiments
    )


if __name__ == '__main__':
    main()
