# Neural-LSH Experimental Study

Complete framework for hyperparameter optimization and comparative analysis of Neural-LSH.

## Overview

This experimental study implements a comprehensive evaluation of Neural-LSH hyperparameters with comparison against baseline methods from the first assignment (LSH, Hypercube, IVFFlat, IVFPQ).

## Phases

### Phase 1: Hyperparameter Grid Search
Explores the hyperparameter space:

- **Graph parameters**: k ∈ {10, 15, 20}
- **Partitioning**: m ∈ {50, 100, 200}
- **MLP architecture**: layers ∈ {2, 3, 4}, nodes ∈ {32, 64, 128}
- **Training**: epochs ∈ {25, 50}

For each configuration:
1. **Build phase**: Constructs the index with Neural-LSH
   - Measures: build time, index size
2. **Search phase**: Runs queries on the built index
   - Measures: recall, approximation factor (AF), queries/second (QPS)

### Phase 2: Build Summary Analysis
Analyzes index construction metrics:
- Build time distribution across configurations
- Index size distribution
- Identifies fastest builds and most compact indexes

### Phase 3: Search Summary Analysis
Analyzes query performance metrics:
- Recall distribution
- QPS distribution
- Efficiency score (recall × QPS)
- Pareto-optimal configurations

### Phase 4: Combined Analysis
Cost-benefit trade-off analysis:
- Fast builds with high recall
- Compact indexes with high recall
- Recommendations for different deployment scenarios

### Phase 5: Comparative Analysis
Compares best Neural-LSH configurations against:
- LSH (Locality-Sensitive Hashing)
- Hypercube (data-oblivious partitioning)
- IVFFlat (Inverted File with flat quantization)
- IVFPQ (Inverted File with product quantization)

Metrics: relative improvements in recall, QPS, and efficiency.

## Scripts

### `experiment_runner.py`
Hyperparameter grid search orchestrator.

```bash
python experiment_runner.py [--knn KNN ...] [--m M ...] [--layers LAYERS ...]
                            [--nodes NODES ...] [--epochs EPOCHS ...]
                            [--data {sift,mnist}] [--max-experiments N]
```

**Outputs**:
- `experimental_study/results/grid_search_results.json` - All results in JSON format
- `experimental_study/results/grid_search_summary.md` - Top 10 configurations by recall

### `summary_report_generator.py`
Generates comprehensive analysis reports.

```bash
python summary_report_generator.py [--build] [--search] [--combined] [--all]
```

**Outputs**:
- `BUILD_SUMMARY.md` - Index construction analysis
- `SEARCH_SUMMARY.md` - Query performance analysis
- `COMBINED_ANALYSIS.md` - Cost-benefit trade-off analysis with recommendations

### `comparative_analysis.py`
Compares Neural-LSH with baseline methods.

```bash
python comparative_analysis.py [--dataset {sift,mnist}] [--output PATH]
```

**Outputs**:
- `COMPARISON_REPORT.md` - Side-by-side comparison with old methods
- Console output with improvement percentages

### `run_experiments.py` (Master Orchestrator)
Coordinates all experimental phases end-to-end.

```bash
# Run complete study with default parameters
python run_experiments.py

# Run with custom hyperparameter ranges
python run_experiments.py --knn 10 15 --m 50 100 --epochs 25 50 --max-experiments 20

# Quick test: only 5 experiments
python run_experiments.py --max-experiments 5

# Generate reports only from existing results
python run_experiments.py --skip-grid-search
```

## Directory Structure

```
experimental_study/
├── experiment_runner.py           # Grid search orchestrator
├── summary_report_generator.py    # Report generation
├── comparative_analysis.py        # Comparative analysis
├── run_experiments.py             # Master orchestrator
├── README.md                      # This file
└── results/                       # Generated outputs
    ├── grid_search_results.json   # All experiment results
    ├── grid_search_summary.md     # Top configurations
    ├── BUILD_SUMMARY.md           # Build phase analysis
    ├── SEARCH_SUMMARY.md          # Search phase analysis
    ├── COMBINED_ANALYSIS.md       # Cost-benefit analysis
    ├── COMPARISON_REPORT.md       # Comparative analysis
    └── *_search.json              # Per-configuration search results
```

## Example Usage

### Run Complete Experimental Study

```bash
cd /Users/theodora/Desktop/uni/project/Neural-LSH
python experimental_study/run_experiments.py
```

This will:
1. Run grid search over all hyperparameter combinations
2. Generate build summary report
3. Generate search summary report
4. Generate combined analysis with recommendations
5. Compare against LSH, Hypercube, IVFFlat, IVFPQ

**Estimated time**: 4-12 hours depending on configuration space size

### Run Reduced Experiment Set (Quick Test)

```bash
python experimental_study/run_experiments.py \
  --knn 10 15 \
  --m 50 100 \
  --layers 2 3 \
  --nodes 32 64 \
  --epochs 25 \
  --max-experiments 10
```

**Estimated time**: 30 minutes - 2 hours

### Generate Reports from Existing Results

```bash
python experimental_study/run_experiments.py --skip-grid-search
```

This will regenerate all reports from the existing `grid_search_results.json`.

### Individual Commands

Run individual phases separately:

```bash
# Grid search only
python experimental_study/experiment_runner.py --knn 10 15 20 --m 50 100 200

# Build summary only
python experimental_study/summary_report_generator.py --build

# Search summary only
python experimental_study/summary_report_generator.py --search

# Combined analysis only
python experimental_study/summary_report_generator.py --combined

# Comparative analysis only (SIFT)
python experimental_study/comparative_analysis.py --dataset sift

# Comparative analysis only (MNIST)
python experimental_study/comparative_analysis.py --dataset mnist
```

## Output Reports

### 1. `grid_search_results.json`
Complete results for all configurations including:
- Configuration parameters
- Build time and index size
- Search metrics (recall, AF, QPS)
- Number of queries executed

### 2. `grid_search_summary.md`
Top 10 configurations by recall with:
- Configuration parameters
- Performance metrics
- Build and search statistics

### 3. `BUILD_SUMMARY.md`
Index construction analysis:
- Build time statistics (min, max, mean, median, std dev)
- Index size statistics
- Fastest 10 builds
- Smallest 10 indexes
- All configurations detailed table

### 4. `SEARCH_SUMMARY.md`
Query performance analysis:
- Recall statistics
- QPS statistics
- Efficiency score statistics
- Top 10 configurations by recall
- Top 10 configurations by speed
- Top 10 configurations by efficiency score

### 5. `COMBINED_ANALYSIS.md`
Cost-benefit trade-off analysis with recommendations:
- Pareto efficiency analysis
- Fast build with high recall configurations
- Compact index with high recall configurations
- Recommendations for different deployment scenarios (maximum recall, maximum speed, best efficiency)

### 6. `COMPARISON_REPORT.md`
Neural-LSH vs old methods:
- Best Neural-LSH configuration details
- Side-by-side performance comparison table
- Per-method detailed analysis with improvement percentages
- Separate sections for SIFT and MNIST datasets

## Performance Metrics

### Build Metrics
- **Build Time**: Total time to construct the index (seconds)
- **Index Size**: Size of the serialized index file (MB)
- **Time per Point**: Build time divided by number of points (microseconds)
- **Index Efficiency**: Index size per 1000 points (KB/1k pts)

### Search Metrics
- **Recall**: Fraction of true top-k neighbors found (0-1)
- **Approximation Factor (AF)**: Ratio of approximate to exact distance (≥1)
- **QPS**: Queries per second (throughput)
- **Time per Query**: Average query execution time (milliseconds)
- **Efficiency Score**: Recall × QPS (quality-speed trade-off)

### Comparison Metrics
- **Recall Improvement**: (Neural-LSH recall - baseline recall) / baseline recall × 100%
- **QPS Improvement**: (Neural-LSH QPS - baseline QPS) / baseline QPS × 100%
- **Efficiency Improvement**: (Neural-LSH efficiency - baseline efficiency) / baseline efficiency × 100%

## Interpretation Guide

### Reading the Reports

**Grid Search Summary**: Find the top configurations by recall to identify the best accuracy achievable.

**Build Summary**: 
- If build time is critical, check the "Fastest Build Configurations" section
- If storage is constrained, check the "Smallest Index Configurations" section

**Search Summary**:
- "Best Recall Configurations": Maximum accuracy at the cost of speed
- "Fastest Query Configurations": Maximum throughput at the cost of accuracy
- "Best Efficiency Configurations": Best balance of accuracy and speed

**Combined Analysis**:
- **Fast build with high recall**: Good for one-time setup cost, then repeated queries
- **Compact index with high recall**: Good for memory/storage-constrained systems
- Recommendations section provides specific configurations for different use cases

**Comparison Report**:
- Look at the improvement percentages to see how Neural-LSH compares
- Positive percentages indicate Neural-LSH is better, negative indicate worse
- The comparison is always against the best configuration of each baseline method

## Notes

- All experiments use the same random seed (1) for reproducibility
- SIFT dataset: 1,000,000 points, 128 dimensions
- MNIST dataset: 60,000 training images, 784 dimensions (28×28)
- Default batch size: 32, learning rate: 0.001
- All metrics averaged across all query points in the dataset

## Troubleshooting

### Grid Search Takes Too Long
Use `--max-experiments` to limit the number of configurations:
```bash
python run_experiments.py --max-experiments 5
```

Or reduce the hyperparameter ranges:
```bash
python run_experiments.py --knn 10 15 --m 50 100 --epochs 25
```

### Missing Old Results
Ensure `old_results/REPORT_TESTS_SUMMARY.json` exists for comparative analysis. This file contains results from the baseline methods.

### Out of Memory
Large datasets (SIFT with 1M points) may cause memory issues. Either:
1. Use a smaller dataset subset
2. Increase system swap space
3. Reduce batch size in MLP training
