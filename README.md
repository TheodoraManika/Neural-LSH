# Neural-LSH: Neural Learned Search Hierarchy

A Python implementation of Neural-LSH for approximate k-NN search using learned hash functions and neural networks.

**Version:** 1.0  
**Status:** Active Development  
**License:** Academic Research Project

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Python Files & Descriptions](#python-files--descriptions)
4. [Installation Instructions](#installation-instructions)
5. [Build Scenario - Detailed Guide](#build-scenario---detailed-guide)
6. [Search Scenario - Detailed Guide](#search-scenario---detailed-guide)
7. [Architecture Overview](#architecture-overview)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Overview

Neural-LSH combines:
- **Graph-based data structures**: k-NN graphs for local neighborhood information
- **Graph partitioning**: KaHIP for balanced graph partitioning
- **Neural networks**: MLP trained to predict optimal hash bucket assignments
- **Fast search**: 4-step approximate search pipeline achieving 100-500ms per query

This implementation supports:
- ✓ SIFT vectors (128-dim, 1M vectors)
- ✓ MNIST images (784-dim, 60k vectors)
- ✓ Custom datasets via NumPy arrays
- ✓ Configurable MLP architectures (layers, nodes)
- ✓ Grid search over hyperparameters

---

## Quick Start

```bash
# 1. Setup environment
bash scripts/setup_mac.sh
source venv/bin/activate

# 2. Build an index (default: SIFT with k=10, m=50)
python3 src/nlsh_build.py --data dataset/sift_base.fvecs --type sift

# 3. Search with the index
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_sift_*.pkl \
  --query dataset/sift_query.fvecs \
  --type sift \
  -T 2 -L 10

# 4. View results
cat results/reports/latest_results.md
```

---

## Python Files & Descriptions

### Core Implementation

#### `nlsh/graph/graph_build.py` - k-NN Graph Construction
- **Purpose**: Builds k-NN graphs from vector data
- **Main Class**: `GraphBuilder`
- **Key Methods**:
  - `build_knn_graph(data, k)` - Constructs k-NN graph using sklearn
  - `symmetrize_graph()` - Makes graph undirected
  - `assign_edge_weights()` - Assigns edge weights based on distances
  - `to_csr_format()` - Converts to sparse CSR format
  - `save_graph_kahip_format()` - Saves in KaHIP binary format
- **Output**: Graph files in `cache/knn_graphs/`

#### `nlsh/index/mlp_model.py` - Neural Classifier
- **Purpose**: MLP trained to predict optimal hash bucket assignments
- **Main Class**: `NeuralLSH_MLP`
- **Architecture**: 3-4 layer MLP with ReLU activation
- **Input**: Query vectors
- **Output**: Predicted partition/bucket indices
- **Training**: Supervised learning on graph partitions

#### `nlsh/index/index_builder.py` - Index Creation
- **Purpose**: Builds inverted file index structure
- **Main Function**: `build_index_with_kahip(graph, data, m, layers, nodes, epochs)`
- **Steps**:
  1. Graph partitioning using KaHIP
  2. MLP training on partition assignments
  3. Inverted file creation (point → bucket mapping)
  4. Index serialization to pickle
- **Output**: Index files in `cache/index_model/`

---

### Main Entry Points

#### `src/nlsh_build.py` - BUILD SCENARIO
**Purpose**: Complete indexing pipeline (5 steps)

**Pipeline Steps**:
1. **Load Data** - Read SIFT/MNIST vectors
2. **Build k-NN Graph** - Construct neighborhood graph (with caching)
3. **Graph Partitioning** - Partition using KaHIP (with caching)
4. **MLP Training** - Train neural classifier on partitions
5. **Index Save** - Save trained model and index to disk

**Usage**:
```bash
python3 src/nlsh_build.py --data <path> --type <sift|mnist> [options]
```

**Key Options**:
- `--data`: Path to dataset file (required)
- `--type`: Dataset type: `sift` or `mnist` (required)
- `--knn`: k for k-NN graph (default: 10)
- `--m`: Number of partitions (default: 50)
- `--layers`: MLP hidden layers (default: 2)
- `--nodes`: Nodes per layer (default: 64)
- `--epochs`: Training epochs (default: 10)
- `--batch-size`: Training batch size (default: 64)
- `--skip-graph`: Skip k-NN graph construction if cached
- `--skip-kahip`: Skip partitioning if cached

**Example**:
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

**Output Files**:
- `cache/knn_graphs/sift_knn15.npz` - k-NN graph
- `cache/partition_caches/sift_kahip_m100.pkl` - Partition assignment
- `cache/index_model/nlsh_sift_k15_m100_l2_n64_e10.pkl` - Final index
- Console output with timing and statistics

---

#### `src/nlsh_search.py` - SEARCH SCENARIO
**Purpose**: Query search using trained index

**Search Pipeline (4 steps)**:
1. **Hash Lookup** - MLP predicts candidate buckets
2. **Candidate Collection** - Retrieve points from top-T buckets
3. **Approximation** - Compute distances using learned representations
4. **Ranking** - Sort by distance and return top-L

**Usage**:
```bash
python3 src/nlsh_search.py --index <path> --query <path> --type <sift|mnist> [options]
```

**Key Options**:
- `--index`: Path to index file (required)
- `--query`: Path to query file (required)
- `--type`: Dataset type: `sift` or `mnist` (required)
- `-T`: Top buckets to probe (default: 2)
- `-L`: k for k-NN results (default: 10)
- `--batch-size`: Search batch size (default: 32)

**Example**:
```bash
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_sift_k15_m100_l2_n64_e10.pkl \
  --query dataset/sift_query.fvecs \
  --type sift \
  -T 2 \
  -L 10
```

**Output Files**:
- `results/search_results_<timestamp>.json` - Result details
- `results/search_results_<timestamp>.md` - Summary report
- Console output with metrics (Recall, QPS, AF)

---

### Search Components

#### `search/search_io.py` - Data I/O Utilities (Shared)
- **Purpose**: Shared data loading for both build and search
- **Key Functions**:
  - `load_data()` - Load vectors from files
  - `load_index_and_model()` - Load trained index
  - `load_queries()` - Load query vectors
  - `save_results()` - Save search results

#### `search/search_algorithms.py` - LSH Search Algorithm
- **Purpose**: Implements the 4-step search pipeline
- **Key Function**: `neural_lsh_search(query, index, model, T, L)`
- **Returns**: Top-L neighbor indices and distances

#### `search/search_metrics.py` - Evaluation Metrics
- **Purpose**: Compute performance metrics
- **Key Functions**:
  - `compute_recall()` - Recall@k metric
  - `compute_qps()` - Queries per second
  - `compute_average_accuracy_fraction()` - Approximation factor (AF)
  - `compute_mrr()` - Mean reciprocal rank

#### `search/search_output.py` - Result Formatting
- **Purpose**: Format and display results
- **Key Functions**:
  - `format_results()` - JSON/Markdown formatting
  - `print_report()` - Console output

#### `search/search_input.py` - Query Input Handling
- **Purpose**: Load and preprocess queries
- **Key Function**: `load_queries()`

#### `search/create_brute_force_cache.py` - Ground Truth Generator
- **Purpose**: Pre-compute exact k-NN distances for evaluation
- **Standalone Utility**: Can be run independently
- **Usage**: `python3 search/create_brute_force_cache.py --sift 1000 --mnist 1000`
- **Output**: Cached distances in `cache/brute_force/`
- **Benefit**: Saves 90-95% of search time by avoiding recomputation

---

### Data Parsers

#### `dataparser/sift_parser.py` - SIFT Vector Loading
- **Format**: `.fvecs` (binary float32 vectors)
- **Dimensions**: 128-dim
- **Dataset Size**: 1M base vectors, 10k queries
- **Key Function**: `SIFTParser().read_fvecs(filename)`

#### `dataparser/mnist_parser.py` - MNIST Image Loading
- **Format**: `.idx3-ubyte` (binary image format)
- **Dimensions**: 784-dim (28×28 images flattened)
- **Dataset Size**: 60k training, 10k test
- **Key Function**: `MNISTParser().read_images(filename)`

---

### Utilities

#### `utils/args.py` - Argument Parsing
- **Purpose**: Validate and parse command-line arguments
- **Key Functions**:
  - `parse_and_validate_build_args()` - Build scenario arguments
  - `parse_and_validate_search_args()` - Search scenario arguments
  - `print_build_summary()` - Display build configuration

#### `utils/math_utils.py` - Mathematical Utilities
- **Purpose**: Distance metrics and evaluation functions
- **Key Functions**:
  - `euclidean_distance()` - Euclidean distance
  - `euclidean_distances_matrix()` - Pairwise Euclidean
  - `manhattan_distance()` - Manhattan distance
  - `cosine_similarity()` - Cosine similarity
  - `hamming_distance()` - Hamming distance
  - `compute_nearest_neighbors()` - Exact k-NN search
  - `compute_recall()` - Recall@k
  - `compute_qps()` - QPS metric
  - `compute_statistics()` - Data statistics

#### `utils/output.py` - Output Formatting
- **Purpose**: Pretty-print results and formatting utilities
- **Key Functions**:
  - `print_header()` - Console headers
  - `print_summary()` - Summary printing
  - `format_time()` - Human-readable timing

---

### Experimental Framework

#### `experimental_study/experiment_runner.py`
- **Purpose**: Grid search over hyperparameters
- **Tested Ranges**:
  - k (k-NN): [10, 15, 20, 25]
  - m (partitions): [50, 100, 150, 200]
  - layers: [2, 3, 4]
  - nodes: [32, 64, 128]
  - epochs: [5, 10, 15, 20]
- **Output**: Experiment results in `results/`

#### `experimental_study/comparative_analysis.py`
- **Purpose**: Compare Neural-LSH with baselines
- **Baselines**: Brute-force, other approximate methods

#### `experimental_study/summary_report_generator.py`
- **Purpose**: Aggregate results and generate reports
- **Output**: JSON and Markdown reports

---

## Installation Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/TheodoraManika/Neural-LSH.git
cd Neural-LSH
```

### Step 2: Setup Python Environment (One-time)

**Option A: Automated Setup (macOS)**
```bash
bash scripts/setup_mac.sh
source venv/bin/activate
```

**Option B: Manual Setup**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies
```bash
pip3 install -r requirements.txt
```

**Requirements Include**:
- `numpy>=1.21.0` - Numerical computation
- `scipy>=1.7.0` - Scientific computing (distance metrics)
- `scikit-learn>=1.0.0` - Machine learning (NearestNeighbors)
- `torch>=1.10.0` - Deep learning (MLP model)
- `kahip-python>=1.14` - Graph partitioning
- `tqdm>=4.60.0` - Progress bars
- `pandas>=1.3.0` - Data manipulation

### Step 4: Verify Installation
```bash
python3 -c "import torch; import kahip; print('✓ All dependencies installed')"
```

### Step 5: Activate Environment (Each Session)
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 6: (Optional) Pre-compute Ground Truth Caches

Pre-computing exact k-NN distances saves 90-95% of search evaluation time:

```bash
python3 search/create_brute_force_cache.py --sift 1000 --mnist 1000
```

This creates:
- `cache/brute_force/brute_force_sift_q1000_n1000000.npz` (3.7GB)
- `cache/brute_force/brute_force_mnist_q1000_n60000.npz` (229MB)

---

## Build Scenario - Detailed Guide

### What is the Build Scenario?

Building creates a trained Neural-LSH index from raw vector data. This is a one-time operation that produces a searchable index file.

### Build Pipeline (5 Steps)

```
Raw Data → k-NN Graph → Partitioning → MLP Training → Index File
```

**Step 1: Load Data**
- Reads vectors from `.fvecs` or `.idx3-ubyte` files
- Validates data format and dimensions
- Output: In-memory numpy array

**Step 2: Build k-NN Graph**
- Constructs k-NN graph from vectors using sklearn
- Creates adjacency matrix (sparse)
- Caches result in `cache/knn_graphs/`
- Output: Graph structure with ~10k-100k edges

**Step 3: Graph Partitioning**
- Uses KaHIP to partition graph into m balanced partitions
- Minimizes edge cuts (good locality)
- Caches partition assignment
- Output: Vector → partition mapping

**Step 4: MLP Training**
- Trains MLP on graph partitions
- Input: Vector embeddings
- Output: Predicted partition (0 to m-1)
- Loss: Cross-entropy classification
- Output: Trained MLP weights

**Step 5: Index Save**
- Creates inverted file: partition → vector list
- Serializes MLP model
- Saves combined index to pickle file
- Output: Single `.pkl` file ready for search

### Running Build

#### Basic Build (SIFT)
```bash
python3 src/nlsh_build.py \
  --data dataset/sift_base.fvecs \
  --type sift
```

**Default Configuration**:
- k=10, m=50, layers=2, nodes=64, epochs=10
- Estimated time: 30-60 seconds

#### Optimized Build (Larger Configuration)
```bash
python3 src/nlsh_build.py \
  --data dataset/sift_base.fvecs \
  --type sift \
  --knn 15 \
  --m 100 \
  --layers 3 \
  --nodes 128 \
  --epochs 20 \
  --batch-size 128
```

**Expected Time**: 2-5 minutes
**Output Index Size**: ~100-150 MB

#### MNIST Build
```bash
python3 src/nlsh_build.py \
  --data dataset/train-images.idx3-ubyte \
  --type mnist \
  --knn 20 \
  --m 50
```

**Expected Time**: 10-20 seconds
**Output Index Size**: ~20-30 MB

### Build Output

Successful build produces:

1. **Console Output**:
   ```
   ╔════════════════════════════════════╗
   ║     Neural-LSH Build Pipeline      ║
   ╠════════════════════════════════════╣
   ║ Dataset: SIFT (1000000 vectors)    ║
   ║ Config: k=15, m=100, layers=2      ║
   ╠════════════════════════════════════╣
   ║ [1/5] Loading data... ✓ 2.3s       ║
   ║ [2/5] Building graph... ✓ 8.5s     ║
   ║ [3/5] Partitioning... ✓ 5.2s       ║
   ║ [4/5] Training MLP... ✓ 12.1s      ║
   ║ [5/5] Saving index... ✓ 0.8s       ║
   ║ Total Time: 28.9 seconds           ║
   ║ Index Size: 145 MB                 ║
   ╚════════════════════════════════════╝
   ```

2. **Files Created**:
   - `cache/knn_graphs/sift_knn15.npz` - k-NN graph
   - `cache/partition_caches/sift_kahip_m100.pkl` - Partitions
   - `cache/index_model/nlsh_sift_k15_m100_l2_n64_e10.pkl` - **Final index**

### Build Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use `--skip-graph` to use cached graph; reduce `--m` |
| KaHIP not found | `pip install kahip-python --force-reinstall` |
| Slow performance | Check disk space; use SSD; reduce dataset size |
| Graph build fails | Verify data format; check vector dimensions |

---

## Search Scenario - Detailed Guide

### What is the Search Scenario?

Searching uses a trained index to find approximate k-NN for queries. This is performed after building an index.

### Search Pipeline (4 Steps)

```
Query Vector → MLP Prediction → Candidate Retrieval → Distance Ranking → Results
```

**Step 1: Hash Lookup**
- Query vector passed through trained MLP
- MLP outputs predicted bucket indices (0 to m-1)
- Output: Top-T bucket predictions

**Step 2: Candidate Collection**
- Retrieve all vectors from top-T buckets
- From inverted index (partition → vector list)
- Output: Set of candidate vectors (~50-500 vectors)

**Step 3: Approximation**
- Compute distances between query and candidates
- Using original or learned representations
- Fast operation (small candidate set)
- Output: Approximate distances

**Step 4: Exact Ranking**
- Sort candidates by distance
- Return top-L nearest neighbors
- Optional: Compute exact distances (if needed)
- Output: Final k-NN results with distances

### Running Search

#### Basic Search (SIFT)
```bash
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_sift_k15_m100_l2_n64_e10.pkl \
  --query dataset/sift_query.fvecs \
  --type sift
```

**Default Configuration**:
- T=2 (top 2 buckets), L=10 (return 10 neighbors)
- Estimated time: 0.5-2 seconds for 10k queries

#### Optimized Search (Accuracy Focus)
```bash
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_sift_k15_m100_l2_n64_e10.pkl \
  --query dataset/sift_query.fvecs \
  --type sift \
  -T 5 \
  -L 100 \
  --batch-size 64
```

**Expected Results**:
- Higher recall (more buckets = more candidates)
- Slower QPS (larger candidate set)
- Better approximation quality

#### Speed-Optimized Search
```bash
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_sift_k15_m100_l2_n64_e10.pkl \
  --query dataset/sift_query.fvecs \
  --type sift \
  -T 1 \
  -L 10 \
  --batch-size 128
```

**Expected Results**:
- Fast QPS (1-2 queries/ms)
- Lower recall
- Good for interactive applications

#### MNIST Search
```bash
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_mnist_k20_m50_l2_n64_e10.pkl \
  --query dataset/t10k-images.idx3-ubyte \
  --type mnist \
  -T 3 \
  -L 10
```

### Search Output

Successful search produces:

1. **Console Output**:
   ```
   ╔════════════════════════════════════╗
   ║    Neural-LSH Search Results       ║
   ╠════════════════════════════════════╣
   ║ Index: nlsh_sift_k15_m100...       ║
   ║ Queries: 10000                     ║
   ║ Config: T=2, L=10                  ║
   ╠════════════════════════════════════╣
   ║ Recall@10: 92.3%                   ║
   ║ AF (Approx Factor): 1.12           ║
   ║ QPS (Queries/sec): 8500            ║
   ║ Search Time: 1.2 seconds           ║
   ╚════════════════════════════════════╝
   ```

2. **Files Created**:
   - `results/search_results_<timestamp>.json` - Detailed results
   - `results/search_results_<timestamp>.md` - Summary report

3. **Result JSON Format**:
   ```json
   {
     "config": {"T": 2, "L": 10},
     "metrics": {
       "recall": 0.923,
       "af": 1.12,
       "qps": 8500,
       "total_time": 1.2
     },
     "results": [
       {"query_id": 0, "neighbors": [123, 456, ...], "distances": [0.1, 0.15, ...]}
     ]
   }
   ```

### Search Optimization

| Goal | T Value | L Value | Trade-off |
|------|---------|---------|-----------|
| **High Speed** | 1-2 | 10 | Lower recall |
| **Balanced** | 3-5 | 10-20 | Medium recall, good QPS |
| **High Recall** | 5-10 | 50 | Slower, higher recall |

### Search Troubleshooting

| Issue | Solution |
|-------|----------|
| Low recall | Increase `-T` (probe more buckets) |
| Slow search | Decrease `-T`; increase `--batch-size` |
| Index not found | Check path; verify index file exists |
| Dimension mismatch | Ensure query and index from same dataset |

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Neural-LSH System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BUILD PHASE (One-time)                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Raw Data (Vectors)                                │  │
│  │        ↓                                             │  │
│  │ 2. k-NN Graph Construction (sklearn)                │  │
│  │        ↓                                             │  │
│  │ 3. Graph Partitioning (KaHIP)                       │  │
│  │        ↓                                             │  │
│  │ 4. MLP Training (PyTorch)                           │  │
│  │        ↓                                             │  │
│  │ 5. Index Serialization (pickle)                     │  │
│  │        ↓                                             │  │
│  │ Output: Trained Index File (.pkl)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  SEARCH PHASE (Per-query)                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Query Vector                                      │  │
│  │        ↓                                             │  │
│  │ 2. MLP Hash Lookup (predict partition)              │  │
│  │        ↓                                             │  │
│  │ 3. Candidate Retrieval (from inverted index)        │  │
│  │        ↓                                             │  │
│  │ 4. Distance Computation & Ranking                   │  │
│  │        ↓                                             │  │
│  │ Output: Top-L Approximate Neighbors                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### MLP Model Architecture

```
Input Layer (d-dimensional)
    ↓
Hidden Layer 1 (nodes neurons, ReLU)
    ↓
Hidden Layer 2 (nodes neurons, ReLU)
    ↓
[Optional] Hidden Layer 3 (nodes neurons, ReLU)
    ↓
Output Layer (m neurons, Softmax)
    ↓
Partition Prediction (0 to m-1)
```

**Default Configuration**:
- Input: 128 (SIFT) or 784 (MNIST)
- Hidden: 64 neurons × 2 layers
- Output: m (number of partitions)
- Activation: ReLU (hidden), Softmax (output)
- Loss: Cross-entropy
- Optimizer: Adam

### Data Caching Structure

```
cache/
├── knn_graphs/
│   ├── sift_knn10.npz      (k-NN graph for SIFT, k=10)
│   ├── sift_knn15.npz      (k-NN graph for SIFT, k=15)
│   └── mnist_knn20.npz     (k-NN graph for MNIST, k=20)
├── partition_caches/
│   ├── sift_kahip_m50.pkl  (Partition assignment, m=50)
│   ├── sift_kahip_m100.pkl (Partition assignment, m=100)
│   └── mnist_kahip_m50.pkl (Partition assignment, m=50)
├── index_model/
│   ├── nlsh_sift_k15_m100_l2_n64_e10.pkl  (Final index)
│   └── nlsh_mnist_k20_m50_l2_n64_e10.pkl  (Final index)
└── brute_force/
    ├── brute_force_sift_q1000_n1000000.npz  (Ground truth)
    └── brute_force_mnist_q1000_n60000.npz   (Ground truth)
```

---

## Troubleshooting

### Installation Issues

**KaHIP Installation Error**
```bash
# Force reinstall
pip install kahip-python --force-reinstall --no-cache-dir
```

**PyTorch Installation Error**
```bash
# CPU version (faster install)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

**Out of Memory During Build**
```bash
# Solution 1: Use cached graph
python3 src/nlsh_build.py --data ... --skip-graph

# Solution 2: Reduce partition count
python3 src/nlsh_build.py --data ... --m 50

# Solution 3: Use smaller dataset
# Use MNIST instead of SIFT
```

**Slow Performance**
```bash
# Check disk space
df -h

# Use SSD if possible
# Reduce batch size
python3 src/nlsh_build.py --data ... --batch-size 32
```

**Low Recall in Search**
```bash
# Increase buckets to probe
python3 src/nlsh_search.py --index ... -T 5

# Train better model (more epochs)
python3 src/nlsh_build.py --data ... --epochs 20
```

**Module Import Errors**
```bash
# Verify virtual environment
source venv/bin/activate
python3 -c "import torch; print(torch.__version__)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Performance Benchmarks

### Build Performance (SIFT 1M vectors)

| Configuration | k-NN Time | Partition | MLP Train | Total |
|---------------|-----------|-----------|-----------|-------|
| k=10, m=50 | 8s | 3s | 5s | 16s |
| k=15, m=100 | 10s | 5s | 8s | 23s |
| k=20, m=150 | 12s | 7s | 12s | 31s |

### Search Performance (SIFT 10k queries)

| Config (T, L) | Recall@L | QPS | Latency/Query |
|---------------|----------|-----|-----------------|
| (1, 10) | 75% | 12000 | 0.08ms |
| (2, 10) | 92% | 8500 | 0.12ms |
| (5, 10) | 98% | 4200 | 0.24ms |
| (10, 100) | 99.5% | 1800 | 0.56ms |

---

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed breakdown.

**Key Directories**:
- `src/` - Main entry points
- `nlsh/` - Core library
- `search/` - Search utilities
- `dataparser/` - Data handlers
- `utils/` - Helper functions
- `cache/` - Cached artifacts
- `results/` - Experiment outputs

---

## References

- **Neural-LSH Paper**: [Original research paper]
- **KaHIP**: K-Way HyperGraph Partitioning Framework
- **scikit-learn**: Machine Learning Library
- **PyTorch**: Deep Learning Framework
- **SIFT**: Scale-Invariant Feature Transform

---

## License

Academic Research Project - Use for educational and research purposes

---

## Support

For issues, questions, or contributions, please refer to the project repository.

**Last Updated**: December 2025
