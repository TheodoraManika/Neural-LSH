# Quick Start: Running Experiments

## 5-Minute Setup

### Option 1: Full Study (Recommended)
Runs complete hyperparameter optimization with comparative analysis:

```bash
cd /Users/theodora/Desktop/uni/project/Neural-LSH
python experimental_study/run_experiments.py
```

Expected output:
- `experimental_study/results/grid_search_results.json` - All results
- `experimental_study/results/BUILD_SUMMARY.md` - Build analysis
- `experimental_study/results/SEARCH_SUMMARY.md` - Search analysis
- `experimental_study/results/COMBINED_ANALYSIS.md` - Recommendations
- `experimental_study/results/COMPARISON_REPORT.md` - vs. old methods

### Option 2: Quick Test (5-10 minutes)
Tests with fewer configurations to verify everything works:

```bash
python experimental_study/run_experiments.py \
  --knn 10 15 \
  --m 50 100 \
  --layers 2 3 \
  --nodes 32 64 \
  --epochs 25 \
  --max-experiments 5
```

### Option 3: Custom Configuration
Run with your specific hyperparameter ranges:

```bash
python experimental_study/run_experiments.py \
  --knn 10 15 20 \
  --m 50 100 200 \
  --layers 2 3 4 \
  --nodes 32 64 128 \
  --epochs 25 50
```

### Option 4: Regenerate Reports Only
If you already have results and want to regenerate reports:

```bash
python experimental_study/run_experiments.py --skip-grid-search
```

## What Each Report Shows

| Report | Purpose | Key Info |
|--------|---------|----------|
| `grid_search_summary.md` | Top configurations | Best 10 configs by recall |
| `BUILD_SUMMARY.md` | Index construction | Build time, index size analysis |
| `SEARCH_SUMMARY.md` | Query performance | Recall, QPS, efficiency metrics |
| `COMBINED_ANALYSIS.md` | Trade-offs | Recommendations for different use cases |
| `COMPARISON_REPORT.md` | vs. old methods | How Neural-LSH compares to LSH, Hypercube, IVFFlat, IVFPQ |

## Expected Results Examples

### For SIFT dataset:
- **Best Recall**: ~0.9+ (depending on parameters)
- **QPS**: 2-8 queries/second
- **Build Time**: 30-300 seconds per configuration
- **Index Size**: 10-100 MB

### For MNIST dataset:
- **Best Recall**: ~0.95+ 
- **QPS**: 10-50 queries/second
- **Build Time**: 5-60 seconds per configuration
- **Index Size**: 1-50 MB

## Command Reference

### Run Grid Search Only
```bash
python experimental_study/experiment_runner.py \
  --knn 10 15 20 \
  --m 50 100 200
```

### Generate Build Report Only
```bash
python experimental_study/summary_report_generator.py --build
```

### Generate Search Report Only
```bash
python experimental_study/summary_report_generator.py --search
```

### Generate Combined Analysis Only
```bash
python experimental_study/summary_report_generator.py --combined
```

### Compare with SIFT Baselines
```bash
python experimental_study/comparative_analysis.py --dataset sift
```

### Compare with MNIST Baselines
```bash
python experimental_study/comparative_analysis.py --dataset mnist
```

## Monitoring Progress

The scripts output progress information:

```
================================================================================
Running Experiment: nlsh_sift_k10_m50_l2_n32_e25
================================================================================
Config: {...}

[BUILD PHASE]
✓ Build completed in 45.23s
  Index size: 15.34 MB

[SEARCH PHASE]
✓ Search completed in 2.15s
```

Each experiment reports:
- Configuration parameters
- Build time and index size
- Search time and metrics
- Overall progress (e.g., `[5/120]`)

## Stopping Gracefully

Press `Ctrl+C` to stop experiments. The already-completed experiments' results will be saved.

## Troubleshooting

### "Module not found" errors
Make sure you're in the project directory:
```bash
cd /Users/theodora/Desktop/uni/project/Neural-LSH
```

### Grid search takes forever
Limit experiments:
```bash
python experimental_study/run_experiments.py --max-experiments 3
```

Or use smaller hyperparameter ranges:
```bash
python experimental_study/run_experiments.py --knn 10 15 --m 50 100 --epochs 25
```

### Memory errors
The system may need more memory for SIFT dataset (1M points). Either:
1. Use MNIST instead: `--data mnist`
2. Reduce batch size (modify experiment_runner.py)
3. Increase system RAM or swap space

## Next Steps

1. **Run experiments** using one of the options above
2. **Check results** in `experimental_study/results/`
3. **Read reports** starting with `COMBINED_ANALYSIS.md` for recommendations
4. **Compare performance** in `COMPARISON_REPORT.md`
5. **Identify best config** from `SEARCH_SUMMARY.md`
6. **Deploy** with the recommended configuration

## Report Locations

All reports are generated in: `experimental_study/results/`

```
experimental_study/results/
├── grid_search_results.json      ← Raw results in JSON
├── grid_search_summary.md        ← Top 10 by recall
├── BUILD_SUMMARY.md              ← Build metrics analysis
├── SEARCH_SUMMARY.md             ← Search metrics analysis
├── COMBINED_ANALYSIS.md          ← Recommendations
└── COMPARISON_REPORT.md          ← vs. old methods
```

View any markdown file:
```bash
cat experimental_study/results/COMBINED_ANALYSIS.md
```

## Time Estimates

| Scenario | Configs | Time |
|----------|---------|------|
| Quick test | 5 | 10-30 min |
| Small grid | 20 | 1-2 hours |
| Medium grid | 60 | 3-6 hours |
| Full grid | 180+ | 8-16 hours |

Times depend on:
- Dataset size (SIFT vs MNIST)
- Configuration complexity (MLP layers/nodes)
- System performance (CPU/GPU availability)
