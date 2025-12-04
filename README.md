# Neural-LSH: Neural Learned Search Hierarchy

A Python implementation of Neural-LSH for approximate k-NN search using learned hash functions.

## Project Structure

```
Neural-LSH/
├── src/
│   ├── nlsh_build.py          # Build index and train MLP model
│   └── nlsh_search.py         # Search using trained model
├── nlsh/
│   ├── graph/                 # k-NN graph construction
│   │   └── knn_graph.py
│   └── model_index_init/      # Model and index initialization
│       ├── mlp_model.py
│       └── index_builder.py
├── search/                    # Search pipeline modules
│   ├── search_algorithms.py   # 4-step search algorithm
│   ├── search_io.py          # Data loading and result saving
│   ├── search_output.py      # Statistics and logging
│   └── search_metrics.py     # Metrics computation
├── utils/                     # Utility functions
│   ├── args.py               # Argument parsing
│   ├── math_utils.py         # Math utilities
│   └── output.py             # Output utilities
├── dataparser/                # Data loading modules
│   ├── mnist_parser.py
│   └── sift_parser.py
├── dataset/                   # Data files (SIFT and MNIST)
├── requirements.txt           # Python dependencies
└── README.md
```

## Prerequisites

### 1. Install Dependencies (macOS)

**Option A: Automatic Setup (Recommended)**
```bash
bash setup_mac.sh
```

This script will:
- Check Python3 installation
- Create a virtual environment
- Install all dependencies

**Option B: Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip3 install --upgrade pip

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Prepare Data

The project includes two datasets:

#### **MNIST Data**
- `dataset/train-images.idx3-ubyte` - Training images
- `dataset/t10k-images.idx3-ubyte` - Test images

#### **SIFT Data**
- `dataset/sift_base.fvecs` - Base vectors (1M SIFT descriptors)
- `dataset/sift_learn.fvecs` - Learning set
- `dataset/sift_query.fvecs` - Query vectors
- `dataset/sift_groundtruth.ivecs` - Ground truth k-NN

## Usage

### Activate Virtual Environment (if used)
```bash
source venv/bin/activate
```

### Step 1: Build Index and Train Model

```bash
cd src/
python3 nlsh_build.py -d <dataset_file> -i <index_output> -type <sift|mnist> [options]
```

**Required arguments:**
- `-d, --dataset` - Path to dataset file (e.g., `../dataset/sift_base.fvecs`)
- `-i, --index` - Output path for index file (e.g., `../model_index/sift_index.bin`)
- `-type` - Dataset type: `sift` or `mnist`

**Optional arguments:**
- `-m, --parts` - Number of buckets (default: 100)
- `--knn` - k for k-NN graph (default: 10)
- `--imbalance` - KaHIP imbalance parameter (default: 0.03)
- `--kahip_mode` - KaHIP mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)
- `--layers` - MLP layers (default: 3)
- `--nodes` - Nodes per layer (default: 64)
- `--epochs` - Training epochs (default: 10)
- `--batch_size` - Batch size (default: 128)
- `--lr` - Learning rate (default: 0.001)
- `--seed` - Random seed (default: 1)

**Example:**
```bash
python3 src/nlsh_build.py -d dataset/sift_learn.fvecs -i model_index/sift_index.bin -type sift -m 100 --knn 10 --layers 3
```

### Step 2: Search with Trained Model

```bash
cd src/
python3 nlsh_search.py -d <dataset_file> -q <query_file> -i <index_file> -o <output_file> -type <sift|mnist> [options]
```

**Required arguments:**
- `-d, --data` - Full dataset file (for brute-force ground truth)
- `-q, --query` - Query vectors file
- `-i, --index` - Index file from build step
- `-o, --output` - Output results file
- `-type` - Dataset type: `sift` or `mnist`

**Optional arguments:**
- `-N` - k for k-NN search (default: 10)
- `-T` - Number of top buckets to probe (default: 5)
- `-R` - Radius for range search (default: 2000 for MNIST, 2800 for SIFT)
- `-range` - Enable range search (default: true)
- `--seed` - Random seed (default: 1)

**Example:**
```bash
python3 nlsh_search.py \
  -d ../dataset/sift_base.fvecs \
  -q ../dataset/sift_query.fvecs \
  -i ../model_index/sift_index.bin \
  -o ../results/sift_results.txt \
  -type sift \
  -N 10 -T 5 -R 2800
```

## Output Format

Results are saved in the specified output file with the following format:

```
<METHOD NAME> Neural LSH
Query: <query_id>
Nearest neighbor-1: <neighbor_id>
distanceApproximate: <distance>
distanceTrue: <distance>
...
Nearest neighbor-N: <neighbor_id>
distanceApproximate: <distance>
distanceTrue: <distance>
R-near neighbors:
<neighbor_id_1>
...
<neighbor_id_K>
Average AF: <metric>
Recall@N: <metric>
QPS: <metric>
tApproximateAverage: <time>
tTrueAverage: <time>
```

## Key Metrics

- **Average AF (Accuracy Fraction)**: Average ratio of true distance to approximate distance
- **Recall@N**: Fraction of true k-NN found in approximate k-NN results
- **QPS**: Queries Per Second (throughput)
- **tApproximateAverage**: Average time for approximate distance computation
- **tTrueAverage**: Average time for brute-force true distance computation

## Testing

Check the `old_results/` folder for example output formats and previous test results.

## Notes

1. The search requires the **full dataset** to compute true distances via brute-force for metric evaluation
2. First run of `nlsh_build.py` may be slow due to k-NN graph construction and KaHIP partitioning
3. MLP training epochs affect accuracy; increase `--epochs` for better results
4. Range search radius (`-R`) depends on dataset dimension and distribution

## Troubleshooting

**KaHIP Installation Issues:**
```bash
# If KaHIP installation fails, try:
pip install kahip-python
```

**Memory Issues:**
- Reduce `--batch_size` in build step
- Use smaller dataset for initial testing

**Slow Performance:**
- Reduce `-m` (number of buckets) for faster search
- Reduce `-T` (probing depth) for faster search
- Increase `--nodes` for better approximation quality