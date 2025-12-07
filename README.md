# Neural-LSH: Neural Learned Search Hierarchy

A Python implementation of Neural-LSH for approximate k-NN search using learned hash functions and neural networks.

## Quick Start

```bash
# 1. Setup (one-time)
bash scripts/setup_mac.sh
source venv/bin/activate

# 2. Pre-compute ground truth caches (optional but recommended)
python3 search/create_brute_force_cache.py --sift 1000 --mnist 1000

# 3. Build an index
python3 src/nlsh_build.py --data dataset/sift_base.fvecs --type sift --knn 15 --m 100

# 4. Run a quick experiment
python3 experimental_study/experiment_runner.py --knn 15 --m 100 --layers 2 --nodes 64 --epochs 10 --data sift --max-experiments 1

# 5. View results
cat results/reports/grid_search_summary.md
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed breakdown.

**Key Directories:**

| Directory | Purpose |
|-----------|---------|
| `src/` | Main entry points (nlsh_build.py, nlsh_search.py) |
| `nlsh/` | Core Neural-LSH library (graph, index, model) |
| `search/` | Search algorithms, I/O, metrics, caching |
| `experimental_study/` | Grid search & comparative analysis |
| `dataset/` | Input data (SIFT, MNIST) |
| `cache/` | Build artifacts (graphs, partitions, indices, brute force) |
| `results/` | Reports and experiment outputs |
| `utils/` | Common utilities (args, math, output) |
| `dataparser/` | Data format handlers (fvecs, idx3-ubyte) |

## Setup & Installation

### 1. Environment Setup (One-time)

```bash
# Automated setup for macOS
bash scripts/setup_mac.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 2. Activate Environment (Each Session)

```bash
source venv/bin/activate
```

### 3. Pre-compute Ground Truth Caches (Optional)

Recommended for faster search evaluation (saves 90-95% time on search phase):

```bash
python3 search/create_brute_force_cache.py --sift 1000 --mnist 1000
```

This creates:
- `cache/brute_force/brute_force_sift_q1000_n1000000.npz` (3.7GB)
- `cache/brute_force/brute_force_mnist_q1000_n60000.npz` (229MB)

## Usage

### Build an Index

```bash
python3 src/nlsh_build.py \
  --data dataset/sift_base.fvecs \
  --type sift \
  --knn 15 \
  --m 100 \
  --layers 2 \
  --nodes 64 \
  --epochs 10
```

**Options:**
- `--data`: Path to dataset file
- `--type`: Dataset type (`sift` or `mnist`)
- `--knn`: k for k-NN graph (default: 10)
- `--m`: Number of partitions (default: 50)
- `--layers`: MLP layers (default: 2)
- `--nodes`: Nodes per layer (default: 64)
- `--epochs`: Training epochs (default: 10)
- `--skip-graph`: Skip k-NN graph construction
- `--skip-kahip`: Skip KaHIP partitioning

### Search with Index

```bash
python3 src/nlsh_search.py \
  --index results/indices/nlsh_sift_k15_m100_l2_n64_e10.pkl \
  --query dataset/sift_query.fvecs \
  --type sift \
  -T 2 \
  -L 10
```

**Options:**
- `--index`: Path to index file
- `--query`: Path to query file
- `--type`: Dataset type (`sift` or `mnist`)
- `-T`: Top buckets to probe (default: 2)
- `-L`: k for k-NN results (default: 10)

### Run Experiments

#### Interactive Menu
```bash
./experimental_study/run_experiment_scenario.sh
```

Provides predefined scenarios:
- Quick Test (5-10 min)
- Balanced Study (3-6 hours)
- Full Study (8-16 hours)
- Focused Grid (1-2 hours)
- MNIST Study (2-4 hours)
- Architecture Search (2-4 hours)
- And more...

#### Direct Command
```bash
python3 experimental_study/experiment_runner.py \
  --knn 15 20 \
  --m 100 200 \
  --layers 2 3 \
  --nodes 64 \
  --epochs 10 \
  --data sift \
  --max-experiments 10
```

#### View Results
```bash
cat results/reports/grid_search_summary.md
cat results/reports/grid_search_results.json
```

## Architecture

Neural-LSH uses a 4-step search pipeline:

1. **Hash Lookup**: Query mapped through trained MLP to find candidate buckets
2. **Candidate Collection**: Retrieve points from top-T buckets
3. **Approximation**: Compute approximate distances using learned representations
4. **Exact Search**: Compute true distances for final ranking

The MLP model learns to map query vectors to hash bucket indices that minimize approximation error.

## Performance

With ground truth caching enabled:
- **Build**: ~25 seconds (index creation + MLP training)
- **Search**: ~100-500ms per query (with 1000 queries)
- **Index Size**: 30-200MB depending on configuration

## Dataset Support

| Format | Extension | Status |
|--------|-----------|--------|
| SIFT fvecs | `.fvecs` | ✓ Full support |
| MNIST idx3-ubyte | `.idx3-ubyte` | ✓ Full support |
| NumPy arrays | `.npy` | ✓ Supported |
| HDF5 | `.h5` | (Can be added) |

Current datasets:
- **SIFT**: 1M base vectors, 10k queries, 128-dim
- **MNIST**: 60k training, 10k test, 784-dim

## Directory Overview

| Directory | Contents |
|-----------|----------|
| `cache/` | All cached artifacts: k-NN graphs (.npz), partition results (.pkl), index files (.pkl), brute force distances (.npz) |
| `cache/knn_graphs/` | k-NN graph cache with naming: `{type}_knn{k}.npz` (e.g., `sift_knn10.npz`) |
| `cache/partition_caches/` | KaHIP partition results with naming: `{type}_kahip_m{m}.pkl` |
| `cache/index_model/` | Built Neural-LSH indices with naming: `nlsh_{type}_k{k}_m{m}_l{layers}_n{nodes}_e{epochs}.pkl` |
| `cache/brute_force/` | Pre-computed distances: `brute_force_{type}_q{n_queries}_n{n_points}.npz` |
| `results/` | Experiment reports and results (JSON, Markdown) |
| `results/archive/` | Historical experimental results |

See [cache/README.md](cache/README.md) and [results/README.md](results/README.md) for details.

## Troubleshooting

### KaHIP Installation Issues
```bash
pip install kahip-python --force-reinstall
```

### Memory Issues
- Reduce batch size: `--batch-size 32`
- Use `--skip-graph` to skip k-NN construction
- Reduce dataset size: Use MNIST instead of SIFT

### Slow Performance
- Reduce `-T` (top buckets to probe)
- Increase `--nodes` for better model quality
- Enable ground truth caching: `python3 search/create_brute_force_cache.py`

## Key Metrics Explained

- **Recall@k**: % of true k-NN found in approximate results
- **AF (Approximation Factor)**: avg(true_distance / approx_distance)
- **QPS**: Queries Per Second throughput
- **Build Time**: Index construction time
- **Index Size**: Final .pkl file size

## References

- Original Neural-LSH paper
- KaHIP: K-Way HyperGraph Partitioning
- SIFT Descriptors: Object Recognition from Local Scale-Invariant Features

## License

Academic research project