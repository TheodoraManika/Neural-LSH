"""
Graph Building Module for Neural LSH

Constructs k-NN similarity graphs for SIFT and MNIST datasets.
The graphs are prepared for balanced partitioning using KaHIP algorithm.

Graph Construction Steps:
1. Build k-NN graph G = (P, E) where P is the dataset and E are k-nearest edges
2. Symmetrize: {i, j} ∈ E ⇔ j ∈ NNk(i) or i ∈ NNk(j)
3. Assign weights:
   - wij = 2 if i ↔ j (mutual neighbors)
   - wij = 1 if one-sided neighbor
   - wij = 0 otherwise (no edge)

Output format: KaHIP binary graph format suitable for partitioning
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Set, Dict
import struct
from tqdm import tqdm

# Import dataset parsers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataparser.sift_parser import SIFTParser
from dataparser.mnist_parser import MNISTParser
from utils.math_utils import euclidean_distances_matrix


class GraphBuilder:
    """Base class for k-NN graph construction."""
    
    def __init__(self, data: np.ndarray, k: int = 10, metric: str = 'euclidean'):
        """
        Initialize graph builder.
        
        Args:
            data: Dataset array of shape (N, D)
            k: Number of nearest neighbors
            metric: Distance metric (currently only 'euclidean' supported)
        """
        self.data = data
        self.k = k
        self.metric = metric
        self.n_points = data.shape[0]
        
    def build_knn_graph(self) -> Tuple[Dict[int, Set[int]], np.ndarray]:
        """
        Build directed k-NN graph.
        
        Returns:
            Tuple of:
            - adjacency_dict: Dictionary mapping point index to set of k-NN indices
            - distances: All pairwise distances matrix
        """
        print(f"Computing pairwise distances for {self.n_points} points...")
        distances = euclidean_distances_matrix(self.data)
        
        # Build k-NN adjacency
        adjacency_dict = {}
        print(f"Building k-NN graph with k={self.k}...")
        
        for i in tqdm(range(self.n_points), desc="k-NN graph"):
            # Get distances from point i to all others
            dists = distances[i, :]
            
            # Find k nearest neighbors (excluding self, which has distance 0)
            # We get k+1 and skip the first (self)
            nearest_indices = np.argsort(dists)[:self.k + 1]
            
            # Remove self (distance 0)
            nearest_indices = nearest_indices[nearest_indices != i][:self.k]
            
            adjacency_dict[i] = set(nearest_indices)
        
        return adjacency_dict, distances
    
    def symmetrize_graph(self, adjacency_dict: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
        """
        Convert directed k-NN graph to undirected via symmetrization.
        
        Symmetrization rule: {i, j} ∈ E ⇔ j ∈ NNk(i) or i ∈ NNk(j)
        
        Args:
            adjacency_dict: Directed k-NN adjacency dictionary
            
        Returns:
            Symmetrized adjacency dictionary
        """
        print("Symmetrizing graph...")
        
        symmetric_adj = {i: set() for i in range(self.n_points)}
        
        for i in tqdm(range(self.n_points), desc="Symmetrization"):
            # Add neighbors of i
            symmetric_adj[i].update(adjacency_dict[i])
            
            # Add points that have i as neighbor
            for j in adjacency_dict[i]:
                symmetric_adj[j].add(i)
        
        return symmetric_adj
    
    def assign_edge_weights(self, 
                           knn_adj: Dict[int, Set[int]], 
                           sym_adj: Dict[int, Set[int]]) -> Dict[Tuple[int, int], int]:
        """
        Assign weights to edges based on relationship type.
        
        Weight assignment:
        - wij = 2 if i ↔ j (mutual neighbors in k-NN)
        - wij = 1 if one-sided (only in symmetrized graph)
        - wij = 0 otherwise
        
        Args:
            knn_adj: Original directed k-NN adjacency
            sym_adj: Symmetrized adjacency
            
        Returns:
            Dictionary mapping (i, j) tuples to weights
        """
        print("Assigning edge weights...")
        
        edge_weights = {}
        
        for i in tqdm(range(self.n_points), desc="Edge weighting"):
            for j in sym_adj[i]:
                # Ensure i < j to avoid duplicate edges
                if i < j:
                    # Check if mutual neighbors (both directions in k-NN)
                    is_mutual = (j in knn_adj[i]) and (i in knn_adj[j])
                    weight = 2 if is_mutual else 1
                    edge_weights[(i, j)] = weight
        
        return edge_weights
    
    def save_graph_kahip_format(self, 
                               sym_adj: Dict[int, Set[int]], 
                               edge_weights: Dict[Tuple[int, int], int],
                               output_path: str) -> None:
        """
        Save graph in KaHIP binary format.
        
        KaHIP format (binary, little-endian):
        - Header: 4 bytes int = number of vertices
        - Header: 4 bytes int = number of edges
        - For each vertex i:
            - List of neighbor indices (4 bytes each, int)
            - List of weights corresponding to neighbors (4 bytes each, int)
        
        Args:
            sym_adj: Symmetrized adjacency dictionary
            edge_weights: Edge weights dictionary
            output_path: Path to save the graph file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving graph to {output_path}...")
        
        # Count total edges (undirected, so each edge counts once)
        total_edges = len(edge_weights)
        
        # Build adjacency lists with weights for each vertex
        vertex_neighbors = {i: [] for i in range(self.n_points)}
        vertex_weights = {i: [] for i in range(self.n_points)}
        
        for (i, j), weight in edge_weights.items():
            vertex_neighbors[i].append(j)
            vertex_weights[i].append(weight)
            vertex_neighbors[j].append(i)
            vertex_weights[j].append(weight)
        
        # Sort neighbors for consistency
        for i in range(self.n_points):
            sorted_idx = np.argsort(vertex_neighbors[i])
            vertex_neighbors[i] = [vertex_neighbors[i][idx] for idx in sorted_idx]
            vertex_weights[i] = [vertex_weights[i][idx] for idx in sorted_idx]
        
        # Write KaHIP format
        with open(output_path, 'wb') as f:
            # Write header: number of vertices and edges
            n_vertices = self.n_points
            n_edges = 2 * total_edges  # Each edge counted twice (both directions)
            
            f.write(struct.pack('<I', n_vertices))
            f.write(struct.pack('<I', n_edges))
            
            # Write adjacency list with weights for each vertex
            for i in tqdm(range(self.n_points), desc="Writing KaHIP format"):
                neighbors = vertex_neighbors[i]
                weights = vertex_weights[i]
                
                # Write neighbors (1-indexed for KaHIP compatibility)
                for neighbor in neighbors:
                    f.write(struct.pack('<I', neighbor + 1))  # 1-indexed
                
                # Write corresponding weights
                for weight in weights:
                    f.write(struct.pack('<I', weight))
        
        # Print statistics
        n_vertices = self.n_points
        n_total_edges = 2 * total_edges
        avg_degree = n_total_edges / n_vertices if n_vertices > 0 else 0
        
        print(f"\n✓ Graph saved successfully!")
        print(f"  Vertices: {n_vertices}")
        print(f"  Edges (undirected): {total_edges}")
        print(f"  Total edge endpoints: {n_total_edges}")
        print(f"  Average degree: {avg_degree:.2f}")
        
        return


def build_sift_graph(dataset_path: str = "dataset/sift_base.fvecs",
                     output_path: str = "nlsh/graph/sift_graph.bin",
                     k: int = 10) -> None:
    """
    Build and save k-NN graph for SIFT dataset.
    
    Args:
        dataset_path: Path to SIFT base vectors file
        output_path: Path to save the graph file
        k: Number of nearest neighbors for graph construction
    """
    print("\n" + "="*70)
    print("Building k-NN Graph for SIFT Dataset")
    print("="*70)
    
    # Load SIFT dataset
    print(f"Loading SIFT dataset from {dataset_path}...")
    parser = SIFTParser()
    sift_data = parser.read_fvecs(dataset_path)
    print(f"✓ Loaded SIFT data: shape={sift_data.shape}, dtype={sift_data.dtype}")
    
    # Build graph
    graph_builder = GraphBuilder(sift_data, k=k, metric='euclidean')
    knn_adj, distances = graph_builder.build_knn_graph()
    sym_adj = graph_builder.symmetrize_graph(knn_adj)
    edge_weights = graph_builder.assign_edge_weights(knn_adj, sym_adj)
    graph_builder.save_graph_kahip_format(sym_adj, edge_weights, output_path)
    
    print("="*70 + "\n")


def build_mnist_graph(dataset_path: str = "dataset/train-images.idx3-ubyte",
                      output_path: str = "nlsh/graph/mnist_graph.bin",
                      k: int = 10) -> None:
    """
    Build and save k-NN graph for MNIST dataset.
    
    Args:
        dataset_path: Path to MNIST training images file
        output_path: Path to save the graph file
        k: Number of nearest neighbors for graph construction
    """
    print("\n" + "="*70)
    print("Building k-NN Graph for MNIST Dataset")
    print("="*70)
    
    # Load MNIST dataset
    print(f"Loading MNIST dataset from {dataset_path}...")
    parser = MNISTParser()
    mnist_images = parser.read_images(dataset_path)
    print(f"✓ Loaded MNIST data: shape={mnist_images.shape}")
    
    # Flatten and normalize images
    mnist_data = parser.flatten_images(mnist_images)
    mnist_data = parser.normalize_images(mnist_data)
    print(f"✓ Preprocessed MNIST data: shape={mnist_data.shape}, dtype={mnist_data.dtype}")
    
    # Build graph
    graph_builder = GraphBuilder(mnist_data, k=k, metric='euclidean')
    knn_adj, distances = graph_builder.build_knn_graph()
    sym_adj = graph_builder.symmetrize_graph(knn_adj)
    edge_weights = graph_builder.assign_edge_weights(knn_adj, sym_adj)
    graph_builder.save_graph_kahip_format(sym_adj, edge_weights, output_path)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build k-NN graphs for SIFT and MNIST datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graph_build.py --dataset sift --knn 15
  python graph_build.py --dataset mnist --knn 10
  python graph_build.py --dataset both --knn 12
        """
    )
    
    parser.add_argument('--dataset', type=str, choices=['sift', 'mnist', 'both'],
                       default='both',
                       help='Dataset to build graph for (default: both)')
    parser.add_argument('--knn', type=int, default=10,
                       help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--sift-input', type=str, default='dataset/sift_base.fvecs',
                       help='Path to SIFT base file (default: dataset/sift_base.fvecs)')
    parser.add_argument('--sift-output', type=str, default='nlsh/graph/sift_graph.bin',
                       help='Output path for SIFT graph (default: nlsh/graph/sift_graph.bin)')
    parser.add_argument('--mnist-input', type=str, default='dataset/train-images.idx3-ubyte',
                       help='Path to MNIST images file (default: dataset/train-images.idx3-ubyte)')
    parser.add_argument('--mnist-output', type=str, default='nlsh/graph/mnist_graph.bin',
                       help='Output path for MNIST graph (default: nlsh/graph/mnist_graph.bin)')
    
    args = parser.parse_args()
    
    if args.dataset in ['sift', 'both']:
        build_sift_graph(
            dataset_path=args.sift_input,
            output_path=args.sift_output,
            k=args.knn
        )
    
    if args.dataset in ['mnist', 'both']:
        build_mnist_graph(
            dataset_path=args.mnist_input,
            output_path=args.mnist_output,
            k=args.knn
        )
