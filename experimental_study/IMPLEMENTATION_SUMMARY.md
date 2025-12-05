# Neural-LSH Experimental Study - Implementation Summary

## Overview

A comprehensive framework for hyperparameter optimization and comparative analysis of Neural-LSH has been implemented. This framework automates the experimental evaluation process specified in your assignment.

## What Was Created

### 1. **Master Orchestrator** (`run_experiments.py`)
- Coordinates all experimental phases end-to-end
- Can run complete study or individual phases
- Provides progress tracking and final summary

### 2. **Grid Search Engine** (`experiment_runner.py`)
- Executes hyperparameter grid search over:
  - Graph construction: k ∈ {10, 15, 20}
  - Partitioning: m ∈ {50, 100, 200}
  - MLP architecture: layers ∈ {2, 3, 4}, nodes ∈ {32, 64, 128}
  - Training: epochs ∈ {25, 50}
- For each configuration, runs both build and search phases
- Saves results to JSON for further analysis

### 3. **Summary Report Generator** (`summary_report_generator.py`)
Generates three comprehensive Markdown reports:
- **BUILD_SUMMARY.md**: Index construction analysis
  - Build time distribution
  - Index size analysis
  - Fastest builds and smallest indexes
- **SEARCH_SUMMARY.md**: Query performance analysis
  - Recall distribution
  - QPS and efficiency metrics
  - Top configurations by various criteria
- **COMBINED_ANALYSIS.md**: Cost-benefit analysis
  - Pareto efficiency frontier
  - Recommendations for different deployment scenarios

### 4. **Comparative Analysis** (`comparative_analysis.py`)
- Compares best Neural-LSH configuration against:
  - LSH (Locality-Sensitive Hashing)
  - Hypercube (data-oblivious partitioning)
  - IVFFlat (Inverted File with flat quantization)
  - IVFPQ (Inverted File with product quantization)
- Generates performance improvement metrics
- Creates side-by-side comparison tables

### 5. **Documentation**
- **README.md**: Comprehensive guide with all usage instructions
- **QUICKSTART.md**: Quick reference for common tasks

## How to Use

### Quick Start (Choose One)

#### Option A: Full Experimental Study
```bash
cd /Users/theodora/Desktop/uni/project/Neural-LSH
python experimental_study/run_experiments.py
```
Runs complete grid search + all reports + comparative analysis.
**Time**: 4-12 hours depending on configuration space

#### Option B: Quick Test
```bash
python experimental_study/run_experiments.py --max-experiments 5
```
Tests with fewer configurations for quick verification.
**Time**: 30 minutes - 2 hours

#### Option C: Generate Reports from Existing Results
```bash
python experimental_study/run_experiments.py --skip-grid-search
```
Regenerates all reports without re-running experiments.
**Time**: < 1 minute

### Custom Configuration
```bash
python experimental_study/run_experiments.py \
  --knn 10 15 \
  --m 50 100 \
  --layers 2 3 \
  --nodes 32 64 \
  --epochs 25 50 \
  --data sift
```

## Generated Reports

All reports are saved to: `experimental_study/results/`

### 1. `grid_search_results.json`
- Complete results for all configurations
- Format: JSON with all metrics and parameters
- Can be imported for custom analysis

### 2. `grid_search_summary.md`
- Top 10 configurations ranked by recall
- Quick reference for best performing setups

### 3. `BUILD_SUMMARY.md`
Key sections:
- Build time statistics (min, max, mean, median, std dev)
- Index size statistics
- Top 10 fastest builds
- Top 10 most compact indexes
- All configurations with detailed metrics

### 4. `SEARCH_SUMMARY.md`
Key sections:
- Recall statistics
- QPS statistics  
- Efficiency score statistics
- Best recall configurations
- Fastest query configurations
- Best efficiency configurations

### 5. `COMBINED_ANALYSIS.md`
Key sections:
- Pareto efficiency analysis
- Fast build with high recall configs
- Compact index with high recall configs
- Recommendations for:
  - Maximum recall scenario
  - Maximum query speed scenario
  - Best efficiency scenario

### 6. `COMPARISON_REPORT.md`
- Best Neural-LSH configuration details
- Performance comparison table (recall, AF, QPS, efficiency)
- Improvement percentages vs. each old method
- Separate analysis for SIFT and MNIST datasets

## Metrics Explained

### Build Phase
- **Build Time**: Total seconds to construct index
- **Index Size**: Size in MB of serialized index
- **Build Time per Point**: Microseconds per data point

### Search Phase
- **Recall**: Fraction of true top-k neighbors found (0-1)
- **Approximation Factor (AF)**: Ratio of approximate to exact distance (≥1)
- **QPS**: Queries per second (throughput)
- **Time per Query**: Milliseconds per query
- **Efficiency Score**: Recall × QPS (quality-speed trade-off)

### Comparison
- **Recall Improvement**: % change in recall vs. baseline
- **QPS Improvement**: % change in throughput vs. baseline
- **Efficiency Improvement**: % change in efficiency score vs. baseline

## Workflow

1. **Grid Search Phase**
   - Builds index for each hyperparameter combination
   - Measures: build time, index size
   - Runs queries on each index
   - Measures: recall, AF, QPS

2. **Analysis Phase**
   - Processes results from all configurations
   - Computes statistics and rankings
   - Identifies Pareto-optimal configurations
   - Generates recommendations

3. **Comparison Phase**
   - Loads baseline results from `old_results/REPORT_TESTS_SUMMARY.json`
   - Compares best Neural-LSH config against best of each old method
   - Calculates improvement percentages
   - Generates comparison report

4. **Report Generation**
   - Creates Markdown reports with tables and analysis
   - Includes recommendations for different scenarios
   - Provides actionable insights

## Key Features

✅ **Fully Automated**: Single command to run complete study
✅ **Modular Design**: Run individual phases as needed
✅ **Comprehensive Analysis**: Build, search, cost-benefit, and comparative
✅ **Reproducible**: Fixed random seeds for all experiments
✅ **Flexible**: Customize hyperparameter ranges
✅ **Progress Tracking**: Real-time status updates
✅ **Well Documented**: README, quickstart, and inline comments

## Implementation Details

### Grid Search Configuration
- **Fixed parameters**: batch_size=32, lr=0.001, imbalance=0.03, kahip_mode=2
- **Variable parameters**: knn, m, layers, nodes, epochs
- **Default space**: 3×3×3×3×2 = 162 configurations

### Result Storage
- Individual results: `grid_search_results.json`
- Per-config search output: `*_search.json`
- Generated reports: `*.md` files
- Index files: `nlsh_*.pkl` (can be deleted after experiments)

### Report Generation
- Markdown format for readability
- Tables for easy comparison
- Detailed breakdowns of top configurations
- Actionable recommendations

## Next Steps

1. **Run Experiments**
   ```bash
   python experimental_study/run_experiments.py
   ```

2. **Review Reports** (in order)
   - Start with `COMBINED_ANALYSIS.md` for overview
   - Check `SEARCH_SUMMARY.md` for best configurations
   - Review `COMPARISON_REPORT.md` to see competitive advantage
   - Look at `BUILD_SUMMARY.md` for deployment considerations

3. **Make Decisions**
   - Identify optimal configuration for your use case
   - Consider trade-offs between accuracy, speed, and resources
   - Choose configuration based on report recommendations

4. **Deploy**
   - Use recommended configuration for production
   - Reference final report for expected performance

## Files Created

```
experimental_study/
├── experiment_runner.py           (500+ lines) - Grid search orchestrator
├── summary_report_generator.py    (400+ lines) - Report generator
├── comparative_analysis.py        (350+ lines) - Comparative analysis
├── run_experiments.py             (300+ lines) - Master orchestrator
├── README.md                      (350+ lines) - Complete guide
├── QUICKSTART.md                  (200+ lines) - Quick reference
└── IMPLEMENTATION_SUMMARY.md      (this file)
```

Plus generated results in `results/` subdirectory.

## Usage Examples

### Run Complete Study
```bash
python experimental_study/run_experiments.py
```

### Run with Reduced Configuration Space
```bash
python experimental_study/run_experiments.py \
  --knn 10 15 --m 50 100 --epochs 25
```

### Run Individual Phases
```bash
# Grid search only
python experimental_study/experiment_runner.py --knn 10 15 20

# Build summary only
python experimental_study/summary_report_generator.py --build

# Search summary only
python experimental_study/summary_report_generator.py --search

# Combined analysis only
python experimental_study/summary_report_generator.py --combined

# Comparative analysis only
python experimental_study/comparative_analysis.py --dataset sift
```

### Regenerate Reports
```bash
python experimental_study/run_experiments.py --skip-grid-search
```

## Estimated Runtime

| Scenario | Configs | Est. Time |
|----------|---------|-----------|
| Quick test | 5 | 10-30 min |
| Small grid | 20 | 1-2 hours |
| Medium grid | 60 | 3-6 hours |
| Full grid (default) | 162 | 8-16 hours |

Factors affecting runtime:
- Dataset size (SIFT: 1M points, MNIST: 60K points)
- Configuration complexity (layers, nodes, epochs)
- System performance (CPU speed, available RAM)

## System Requirements

- Python 3.7+
- PyTorch (for MLP training)
- NumPy, scikit-learn
- KaHIP (for graph partitioning)
- ~10GB free disk space for indexes and results
- 8GB+ RAM recommended

## Troubleshooting

### Grid search takes too long
- Use `--max-experiments` to limit configurations
- Reduce hyperparameter ranges
- Use MNIST instead of SIFT for faster iteration

### Memory issues
- Increase system RAM or swap space
- Reduce MLP nodes parameter
- Use smaller batch sizes

### Missing old results
- Ensure `old_results/REPORT_TESTS_SUMMARY.json` exists
- Run comparative analysis with `--skip-grid-search` to regenerate reports

## Support

For questions or issues:
1. Check `README.md` and `QUICKSTART.md` for detailed documentation
2. Review generated reports for interpretation guidance
3. Check experiment logs for specific errors
4. Modify scripts as needed for custom analysis

---

**Framework Version**: 1.0
**Last Updated**: 2025-12-05
**Status**: Complete and Ready to Use
