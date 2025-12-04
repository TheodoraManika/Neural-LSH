"""
Neural-LSH Build Script
========================
Υπεύθυνο για την κατασκευή του ευρετηρίου (training and indexing phase).

Βήματα:
1. Φόρτωση και Προεπεξεργασία Δεδομένων
2. Κατασκευή Γράφου k-NN
3. Διαμέριση Γράφου με KaHIP
4. Εκπαίδευση MLP
5. Αποθήκευση Μοντέλου και Ευρετηρίου (ενοποιημένο αρχείο inverted file)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from typing import Dict, Set, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and index builder directly
import importlib.util
spec = importlib.util.spec_from_file_location("mlp_model", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nlsh/index/mlp_model.py"))
mlp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_module)
NeuralLSH_MLP = mlp_module.NeuralLSH_MLP

spec = importlib.util.spec_from_file_location("index_builder",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nlsh/index/index_builder.py"))
builder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder_module)
build_index_with_kahip = builder_module.build_index_with_kahip

# Import from utils
spec = importlib.util.spec_from_file_location("args",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils/args.py"))
args_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(args_module)
parse_and_validate_build_args = args_module.parse_and_validate_build_args
print_build_summary = args_module.print_build_summary

# Import data parsers
spec = importlib.util.spec_from_file_location("sift_parser",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataparser/sift_parser.py"))
sift_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sift_module)
SIFTParser = sift_module.SIFTParser

spec = importlib.util.spec_from_file_location("mnist_parser",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataparser/mnist_parser.py"))
mnist_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnist_module)
MNISTParser = mnist_module.MNISTParser

# Import math utilities for distance computation
spec = importlib.util.spec_from_file_location("math_utils",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils/math_utils.py"))
math_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_module)
euclidean_distances_matrix = math_module.euclidean_distances_matrix

try:
    import kahip
    KAHIP_AVAILABLE = True
except ImportError:
    KAHIP_AVAILABLE = False

# Use sklearn for efficient nearest neighbor search
from sklearn.neighbors import NearestNeighbors


def load_data(dataset_path, data_type):
    """
    Φόρτωση δεδομένων από αρχείο χρησιμοποιώντας τους κατάλληλους parsers.
    
    Args:
        dataset_path: Διαδρομή προς το αρχείο δεδομένων
        data_type: Τύπος δεδομένων ('sift' ή 'mnist')
        
    Returns:
        numpy array με τα δεδομένα
    """
    # Resolve relative paths based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, dataset_path) if not os.path.isabs(dataset_path) else dataset_path
    
    print(f"Φόρτωση δεδομένων από: {dataset_path} (type: {data_type})")
    
    if dataset_path.endswith('.npy'):
        # Load from numpy format
        data = np.load(dataset_path)
    elif data_type.lower() == 'sift':
        # Load SIFT data using SIFTParser
        parser = SIFTParser()
        data = parser.read_fvecs(dataset_path)
    elif data_type.lower() == 'mnist':
        # Load MNIST data using MNISTParser
        parser = MNISTParser()
        images = parser.read_images(dataset_path)
        # Flatten images from (N, 28, 28) to (N, 784)
        data = parser.flatten_images(images)
        # Normalize to [0, 1]
        data = parser.normalize_images(data)
    else:
        raise ValueError(f"Άγνωστος τύπος δεδομένων: {data_type}. "
                        "Χρησιμοποιήστε 'sift', 'mnist' ή .npy αρχεία.")
    
    print(f"Φορτώθηκαν {len(data)} σημεία με διάσταση {data.shape[1]}")
    return data.astype(np.float32)


def build_knn_graph(data: np.ndarray, k: int = 10) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Build k-NN graph from data using sklearn's efficient implementation.
    
    Args:
        data: Data points (N, D)
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (knn_adjacency, symmetric_adjacency)
    """
    print(f"\n[Κατασκευή k-NN Γράφου]")
    print(f"  - Κατασκευή k-NN γράφου με k={k} (sklearn NearestNeighbors)...")
    
    n_points = len(data)
    
    # Use sklearn's efficient KNN search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Build directed k-NN graph (skip self at index 0)
    knn_adjacency = {}
    for i in range(n_points):
        # indices[i, 0] is self, so take indices[i, 1:k+1]
        nearest_indices = indices[i, 1:k+1]
        knn_adjacency[i] = set(nearest_indices)
    
    # Symmetrize graph
    print(f"  - Συμμετροποίηση γράφου...")
    symmetric_adjacency = {i: set() for i in range(n_points)}
    
    for i in range(n_points):
        symmetric_adjacency[i].update(knn_adjacency[i])
        for j in knn_adjacency[i]:
            symmetric_adjacency[j].add(i)
    
    total_edges = sum(len(neighbors) for neighbors in symmetric_adjacency.values()) // 2
    print(f"  ✓ Γράφος κατασκευάστηκε: {total_edges} ακμές")
    
    return knn_adjacency, symmetric_adjacency


def convert_graph_to_csr(knn_adj: Dict[int, Set[int]], sym_adj: Dict[int, Set[int]], 
                        n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert symmetric graph to CSR format for KaHIP.
    
    Args:
        knn_adj: Directed k-NN adjacency
        sym_adj: Symmetric adjacency
        n_points: Total number of points
        
    Returns:
        Tuple of (xadj, adjncy, adjcwt) arrays
    """
    print(f"  - Μετατροπή σε CSR format για KaHIP...")
    
    # Build CSR arrays
    xadj = [0]  # First element is always 0
    adjncy = []
    adjcwt = []
    
    for i in range(n_points):
        neighbors = sorted(list(sym_adj[i]))
        
        for j in neighbors:
            adjncy.append(j)
            
            # Assign weights: 2 if mutual, 1 if one-sided
            is_mutual = (j in knn_adj[i]) and (i in knn_adj[j])
            weight = 2 if is_mutual else 1
            adjcwt.append(weight)
        
        xadj.append(len(adjncy))
    
    xadj = np.array(xadj, dtype=np.int32)
    adjncy = np.array(adjncy, dtype=np.int32)
    adjcwt = np.array(adjcwt, dtype=np.int32)
    
    print(f"  ✓ CSR format: xadj={len(xadj)}, adjncy={len(adjncy)}, adjcwt={len(adjcwt)}")
    
    return xadj, adjncy, adjcwt


def main():
    """Κύρια συνάρτηση για την κατασκευή του Neural-LSH ευρετηρίου."""
    # Parse and validate arguments using utils.args
    args = parse_and_validate_build_args()
    
    # Ορισμός seed για αναπαραγωγιμότητα
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Δημιουργία φακέλων αν δεν υπάρχουν
    index_dir = os.path.dirname(args.index)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    
    # Print configuration summary
    print_build_summary(args)
    
    # =========================================================================
    # ΒΗΜΑ 1: Φόρτωση και Προεπεξεργασία Δεδομένων
    # =========================================================================
    print("[ΒΗΜΑ 1] Φόρτωση Δεδομένων...")
    data = load_data(args.data, args.type)
    n_points, dim = data.shape
    
    # =========================================================================
    # ΒΗΜΑ 2: Κατασκευή k-NN Γράφου
    # =========================================================================
    print(f"\n[ΒΗΜΑ 2] Κατασκευή k-NN Γράφου και Διαμέριση με KaHIP...")
    
    # Build k-NN graph
    knn_adj, sym_adj = build_knn_graph(data, k=args.knn)
    
    # Convert to CSR format
    xadj, adjncy, adjcwt = convert_graph_to_csr(knn_adj, sym_adj, n_points)
    
    # =========================================================================
    # ΒΗΜΑ 3: Διαμέριση Γράφου με KaHIP
    # =========================================================================
    if not KAHIP_AVAILABLE:
        print("\n  ⚠ KaHIP δεν είναι διαθέσιμο - χρησιμοποιώντας τυχαία διαμέριση")
        partition_vector = np.random.randint(0, args.m, size=n_points)
    else:
        print(f"\n  [KaHIP Partitioning]")
        print(f"    • Nodes: {n_points}, Edges: {len(adjncy)//2}")
        print(f"    • Target buckets (m): {args.m}")
        print(f"    • Imbalance (ε): {args.imbalance}")
        print(f"    • Mode: {['FAST', 'ECO', 'STRONG'][args.kahip_mode]}")
        
        try:
            # Call KaHIP for partitioning
            edgecut, partition_vector = kahip.kaffpa(
                vwgt=np.ones(n_points, dtype=np.int32),
                xadj=xadj,
                adjcwt=adjcwt,
                adjncy=adjncy,
                nblocks=args.m,
                imbalance=args.imbalance,
                suppress_output=False,
                seed=args.seed,
                mode=args.kahip_mode
            )
            partition_vector = np.array(partition_vector, dtype=np.int32)
            print(f"  ✓ KaHIP completed: Edge-cut = {edgecut}")
        except Exception as e:
            print(f"  ✗ KaHIP error: {e} - falling back to random partition")
            partition_vector = np.random.randint(0, args.m, size=n_points)
    
    # =========================================================================
    # ΒΗΜΑ 4: Εκπαίδευση MLP
    # =========================================================================
    print(f"\n[ΒΗΜΑ 4] Εκπαίδευση MLP μοντέλου...")
    print(f"  - Αρχιτεκτονική: {args.layers} στρώματα, {args.nodes} κόμβοι")
    print(f"  - Παράμετροι: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Device: {device}")
    
    # Δημιουργία μοντέλου
    model = NeuralLSH_MLP(
        input_dim=dim,
        num_layers=args.layers,
        nodes_per_layer=args.nodes,
        output_dim=args.m
    ).to(device)
    
    # Ορισμός κριτηρίου απώλειας και βελτιστοποιητή
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Μετατροπή δεδομένων σε PyTorch tensors
    X_tensor = torch.FloatTensor(data).to(device)
    y_tensor = torch.LongTensor(partition_vector).to(device)
    
    # Εκπαίδευση
    print("\n  Εκπαίδευση:")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Mini-batch training
        permutation = np.random.permutation(n_points)
        epoch_loss = 0.0
        n_batches = (n_points + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, n_points)
            
            batch_indices = permutation[start_idx:end_idx]
            X_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}")
    
    print(f"  ✓ Εκπαίδευση ολοκληρώθηκε. Best Loss: {best_loss:.6f}")
    
    # =========================================================================
    # ΒΗΜΑ 5: Αποθήκευση Μοντέλου και Ευρετηρίου (Inverted File)
    # =========================================================================
    print(f"\n[ΒΗΜΑ 5] Αποθήκευση Μοντέλου και Ευρετηρίου...")
    
    # Δημιουργία index με KaHIP
    index_file = build_index_with_kahip(
        data=data,
        xadj=xadj,
        adjncy=adjncy,
        adjcwt=adjcwt,
        model_state_dict=model.state_dict(),
        model_config=model.get_config(),
        num_buckets=args.m,
        imbalance=args.imbalance,
        kahip_mode=args.kahip_mode
    )
    
    # Αποθήκευση
    index_file.save(args.index)
    
    print("\n" + "=" * 80)
    print("NEURAL-LSH BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nΕπιτυχής δημιουργία ευρετηρίου:")
    print(f"  - Δεδομένα: {n_points} σημεία × {dim} διάστασης")
    print(f"  - k-NN Graph: k={args.knn}")
    print(f"  - Διαμέριση: {args.m} buckets (με KaHIP)")
    print(f"  - MLP: {args.layers} στρώματα × {args.nodes} κόμβοι")
    print(f"  - Ευρετήριο: {args.index}")
    print()


if __name__ == '__main__':
    main()
