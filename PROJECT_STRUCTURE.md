# Project Structure - Neural-LSH

## Core Implementation

```
src/
├── nlsh_build.py          # Main index building script
└── nlsh_search.py         # Main search script
```

## Neural-LSH Library

```
nlsh/
├── __init__.py
├── graph/                 # k-NN graph operations
│   ├── __init__.py
│   ├── knn_graph.py      # k-NN graph construction
│   └── graph_build.py    # Graph building utilities
└── index/                # Index and model operations
    ├── __init__.py
    ├── index_builder.py  # Index construction
    └── mlp_model.py      # MLP model definition
```

## Supporting Modules

```
search/
├── search_algorithms.py   # LSH and exact search algorithms
├── search_io.py          # Data loading (SIFT, MNIST, fvecs, idx3-ubyte)
├── search_metrics.py     # Recall, AF, QPS calculations
├── search_output.py      # Result formatting
└── create_brute_force_cache.py  # Pre-compute ground truth distances

utils/
├── args.py               # Argument parsing
├── math_utils.py         # Mathematical utilities
└── output.py             # Output formatting

dataparser/
├── sift_parser.py        # SIFT .fvecs format handler
└── mnist_parser.py       # MNIST idx3-ubyte format handler
```

## Experimental Framework

```
experimental_study/
├── experiment_runner.py   # Grid search orchestrator
├── comparative_analysis.py  # Compare methods
├── summary_report_generator.py  # Report generation
└── run_experiment_scenario.sh   # Predefined scenarios
```

## Data & Results

```
dataset/              # Input data (downloaded separately)
├── sift_base.fvecs   # 1M SIFT vectors
├── sift_query.fvecs  # SIFT queries
├── train-images.idx3-ubyte  # 60k MNIST training
└── t10k-images.idx3-ubyte   # 10k MNIST test

cache/                # Build artifacts & caches
├── knn_graphs/       # k-NN graph cache (.npz files)
├── partition_caches/ # KaHIP partition results (.pkl files)
├── index_model/      # Built Neural-LSH indices (.pkl files)
└── brute_force/      # Ground truth distances (.npz files)

results/              # Experiment outputs
├── reports/          # JSON/Markdown reports
└── archive/          # Old experimental results

(Note: Index files are now stored in cache/index_model/)
```

## Configuration Files

```
requirements.txt      # Python dependencies
.gitignore           # Git ignore rules
scripts/
└── setup_mac.sh     # Setup script for macOS
```

## Key Files

- **README.md** - Project overview
- **Ergasia_2_2025.pdf** - Assignment specification
- **2.Neural_LSH.pdf** - Research paper reference

## Workflow

### 1. Building an Index
```bash
python3 src/nlsh_build.py \
  --data dataset/sift_base.fvecs \
  --type sift \
  --knn 15 \
  --m 100
```

### 2. Searching with Index
```bash
python3 src/nlsh_search.py \
  --index results/indices/nlsh_sift_k15_m100_l2_n64_e10.pkl \
  --query dataset/sift_query.fvecs \
  --type sift \
  -T 2 \
  -L 10
```

### 3. Running Experiments
```bash
# Interactive menu
./experimental_study/run_experiment_scenario.sh

# Direct command
python3 experimental_study/experiment_runner.py \
  --knn 15 20 \
  --m 100 200 \
  --layers 2 3 \
  --nodes 64 \
  --epochs 10 \
  --data sift
```

### 4. Pre-computing Ground Truth Cache
```bash
python3 search/create_brute_force_cache.py --sift 1000 --sift-only
python3 search/create_brute_force_cache.py --mnist 1000 --mnist-only
```

## Caching System

All caches are stored in `cache/` directory for easy management and portability:

- **k-NN Graphs**: `cache/knn_graphs/{type}_knn{k}.npz` - Cached k-NN graph structures (CSR format)
- **KaHIP Partitions**: `cache/partition_caches/{type}_kahip_m{m}.pkl` - Cached graph partitioning results
- **Index Models**: `cache/index_model/nlsh_{type}_k{k}_m{m}_l{layers}_n{nodes}_e{epochs}.pkl` - Built Neural-LSH index files
- **Brute Force Distances**: `cache/brute_force/brute_force_{type}_q{n_queries}_n{n_points}.npz` - Pre-computed ground truth (3.8GB for SIFT, 229MB for MNIST)

## Clean-up Notes

- Old experimental results moved to `results/archive/`
- Index PKL files organized in `results/indices/`
- KaHIP partitions in `cache/partition_caches/`
- k-NN graph caches in `cache/knn_graphs/`
- All generated files covered by .gitignore
