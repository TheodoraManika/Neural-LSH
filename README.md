# Neural-LSH: Neural Learned Search Hierarchy

Στοιχεία

Φοιτητές:

  Τσάκωνα Μυρτώ - [1115202100188] - [sdi2100188@di.uoa.gr]
  Μανίκα Θεοδώρα - [1115202100267] - [sdi2100267@di.uoa.gr]

Ακαδημαϊκό Έτος: 2025-2026

Μάθημα: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα

Ίδρυμα: ΕΚΠΑ

---

## Overview

Neural-LSH combines:
- **Graph-based data structures**: k-NN graphs for local neighborhood information
- **Graph partitioning**: KaHIP for balanced graph partitioning
- **Neural networks**: MLP trained to predict optimal hash bucket assignments
- **Fast search**: 4-step approximate search pipeline achieving 100-500ms per query

---

## Python Files & Descriptions

### Core Implementation

```
src/
├── nlsh_build.py          # Main index building script
└── nlsh_search.py         # Main search script
```

### Neural-LSH Library

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

### Supporting Modules

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

### Experimental Framework

```
experimental_study/
├── experiment_runner.py            # Grid search orchestrator
├── experiment_runner_extended.py   # Extended experiments with novel scenarios
├── comparative_analysis.py         # Compare methods
├── summary_report_generator.py     # Report generation
└── EXPERIMENTAL_RESULTS_NEURAL_LSH.md  # Comprehensive results report

scripts/
├── run_experiment_scenario.sh      # Predefined experimental scenarios
└── setup_mac.sh                    # macOS setup script
```

### Data & Results

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
└── [search/build results stored here]

build/                # Legacy build folder (deprecated)
├── graph/            # Legacy graph building
└── index_model/      # Legacy index building (superseded by nlsh/)
```
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
---

### Additional Utilities

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

**Default Configuration**:
- k=10, m=50, layers=2, nodes=64, epochs=10
- Estimated time: 30-60 seconds

#### Sift
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

#### MNIST Build
```bash
python3 src/nlsh_build.py \
  --data dataset/train-images.idx3-ubyte \
  --type mnist \
  --knn 20 \
  --m 50
```

## Search Pipeline (4 Steps)

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

#### MNIST Search
```bash
python3 src/nlsh_search.py \
  --index cache/index_model/nlsh_mnist_k20_m50_l2_n64_e10.pkl \
  --query dataset/t10k-images.idx3-ubyte \
  --type mnist \
  -T 3 \
  -L 10
```
---