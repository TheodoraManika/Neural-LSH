"""
Neural-LSH Build Script
========================
Υπεύθυνο για την κατασκευή του ευρετηρίου (training and indexing phase).

Βήματα:
1. Φόρτωση και Προεπεξεργασία Δεδομένων
2. Κατασκευή Γράφου k-NN
3. Διαμέριση Γράφου με KaHIP
4. Εκπαίδευση MLP
5. Αποθήκευση Μοντέλου και Ευρετηρίου
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import kahip
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlsh.graph.knn_graph import build_knn_graph, convert_graph_to_csr
from nlsh.model_index_init.mlp_model import NeuralLSH_MLP
from nlsh.model_index_init.index_builder import build_index
from dataparser.sift_parser import SIFTParser
from dataparser.mnist_parser import MNISTParser


def parse_arguments():
    """Ανάλυση των ορισμάτων γραμμής εντολών."""
    parser = argparse.ArgumentParser(description='Neural-LSH Build: Κατασκευή ευρετηρίου και εκπαίδευση μοντέλου')
    
    # Υποχρεωτικά ορίσματα
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Το αρχείο του συνόλου δεδομένων (π.χ., input.dat)')
    parser.add_argument('-i', '--index', type=str, required=True,
                        help='Το path όπου θα αποθηκευτούν τα αρχεία του ευρετηρίου (π.χ., nlsh_index.bin)')
    parser.add_argument('-type', '--type', type=str, required=True, choices=['sift', 'mnist'],
                        help='Ο τύπος των δεδομένων (sift ή mnist)')
    
    # Παράμετροι k-NN γράφου
    parser.add_argument('--knn', type=int, default=10,
                        help='Ο αριθμός γειτόνων k για την κατασκευή του k-NN γράφου (Default: 10)')
    
    # Παράμετροι KaHIP
    parser.add_argument('-m', '--parts', type=int, default=100,
                        help='Ο αριθμός των μερών (blocks/parts) m για το KaHIP (Default: 100)')
    parser.add_argument('--imbalance', type=float, default=0.03,
                        help='Το ποσοστό ανισορροπίας ϵ για το KaHIP (Default: 0.03)')
    parser.add_argument('--kahip_mode', type=int, default=2, choices=[0, 1, 2],
                        help='Η ρύθμιση του KaHIP (0=FAST, 1=ECO, 2=STRONG) (Default: 2)')
    
    # Παράμετροι MLP
    parser.add_argument('--layers', type=int, default=3,
                        help='Αριθμός στρωμάτων MLP (Default: 3)')
    parser.add_argument('--nodes', type=int, default=64,
                        help='Αριθμός κόμβων στρώματος (Default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Περίοδοι εκπαίδευσης του MLP (Default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Μέγεθος δέσμης (Default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Ρυθμός εκμάθησης (Default: 0.001)')
    
    # Φύτρο ψευδοτυχαίων
    parser.add_argument('--seed', type=int, default=1,
                        help='Φύτρο ψευδοτυχαίων (Default: 1)')
    
    return parser.parse_args()


def load_data(dataset_path, data_type):
    """
    Φόρτωση δεδομένων από αρχείο.
    Το dataparser θα υλοποιηθεί από άλλον, οπότε εδώ περιμένουμε
    ότι τα δεδομένα θα φορτωθούν από το κατάλληλο module.
    
    Args:
        dataset_path: Διαδρομή προς το αρχείο δεδομένων
        data_type: Τύπος δεδομένων ('sift' ή 'mnist')
        
    Returns:
        numpy array με τα δεδομένα
    """
    # Placeholder - θα αντικατασταθεί με το dataparser
    # Προς το παρόν υποθέτουμε ότι θα υπάρχει η συνάρτηση:
    # from dataparser import load_dataset
    # data = load_dataset(dataset_path, data_type)
    
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
        data = images.reshape(images.shape[0], -1)
        # Normalize to [0, 1]
        data = data.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Άγνωστος τύπος δεδομένων: {data_type}. "
                        "Χρησιμοποιήστε 'sift', 'mnist' ή .npy αρχεία.")
    
    print(f"Φορτώθηκαν {len(data)} σημεία με διάσταση {data.shape[1]}")
    return data


def main():
    """Κύρια συνάρτηση για την κατασκευή του Neural-LSH ευρετηρίου."""
    args = parse_arguments()
    
    # Ορισμός seed για αναπαραγωγιμότητα
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Δημιουργία φακέλων αν δεν υπάρχουν
    index_dir = os.path.dirname(args.index)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    
    print("=" * 80)
    print("NEURAL-LSH BUILD: Κατασκευή Ευρετηρίου και Εκπαίδευση Μοντέλου")
    print("=" * 80)
    
    # =========================================================================
    # ΒΗΜΑ 1: Φόρτωση και Προεπεξεργασία Δεδομένων
    # =========================================================================
    print("\n[ΒΗΜΑ 1] Φόρτωση Δεδομένων...")
    data = load_data(args.dataset, args.type)
    n_points, dim = data.shape
    
    # =========================================================================
    # ΒΗΜΑ 2: Κατασκευή Γράφου k-NN
    # =========================================================================
    print(f"\n[ΒΗΜΑ 2] Κατασκευή k-NN γράφου (k={args.knn})...")
    print("  - Εύρεση k-πλησιέστερων γειτόνων...")
    print("  - Συμμετρικοποίηση γράφου...")
    print("  - Απόδοση βαρών (mutual NN: 2, one-way NN: 1)...")
    
    graph, edge_weights = build_knn_graph(data, k=args.knn)
    
    print(f"  Γράφος κατασκευάστηκε με {len(graph)} κόμβους")
    
    # =========================================================================
    # ΒΗΜΑ 3: Διαμέριση Γράφου με KaHIP
    # =========================================================================
    print(f"\n[ΒΗΜΑ 3] Διαμέριση γράφου σε m={args.parts} buckets με KaHIP...")
    print(f"  - Παράμετροι: imbalance={args.imbalance}, mode={args.kahip_mode}")
    print("  - Μετατροπή γράφου σε CSR μορφή...")
    
    vwgt, xadj, adjncy, adjcwgt = convert_graph_to_csr(graph, edge_weights, n_points)
    
    print("  - Εκτέλεση KaHIP για balanced partitioning...")
    # Κλήση kahip.kaffpa για διαμέριση
    edgecut, partition_map = kahip.kaffpa(
        vwgt=vwgt,
        xadj=xadj,
        adjncy=adjncy,
        adjcwgt=adjcwgt,
        nparts=args.parts,
        imbalance=args.imbalance,
        seed=args.seed,
        mode=args.kahip_mode
    )
    
    partition_map = np.array(partition_map)
    print(f"  Διαμέριση ολοκληρώθηκε (edgecut={edgecut})")
    print(f"  Κατανομή σημείων ανά bucket:")
    for i in range(args.parts):
        count = np.sum(partition_map == i)
        print(f"    Bucket {i}: {count} σημεία")
    
    # =========================================================================
    # ΒΗΜΑ 4: Εκπαίδευση MLP
    # =========================================================================
    print(f"\n[ΒΗΜΑ 4] Εκπαίδευση MLP ({args.epochs} epochs)...")
    print(f"  Αρχιτεκτονική: {args.layers} layers, {args.nodes} nodes per layer")
    print(f"  Δομή: {dim} -> " + " -> ".join([f"{args.nodes}"] * (args.layers - 1)) + f" -> {args.parts}")
    
    # Δημιουργία μοντέλου
    model = NeuralLSH_MLP(
        input_dim=dim,
        num_layers=args.layers,
        nodes_per_layer=args.nodes,
        output_dim=args.parts
    )
    
    # Προετοιμασία για εκπαίδευση
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Χρήση συσκευής: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Μετατροπή δεδομένων σε tensors
    X_train = torch.FloatTensor(data).to(device)
    y_train = torch.LongTensor(partition_map).to(device)
    
    # Δημιουργία DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Εκπαίδευση
    print("  Ξεκινάει η εκπαίδευση...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("  Εκπαίδευση ολοκληρώθηκε!")
    
    # =========================================================================
    # ΒΗΜΑ 5: Αποθήκευση Μοντέλου και Ευρετηρίου
    # =========================================================================
    print(f"\n[ΒΗΜΑ 5] Αποθήκευση μοντέλου και ευρετηρίου...")
    
    # Δημιουργία ενός unified index file που περιέχει και το μοντέλο και τα δεδομένα
    index_data = {
        # Model information
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': dim,
            'num_layers': args.layers,
            'nodes_per_layer': args.nodes,
            'output_dim': args.parts,
        },
        # Index information
        'index': build_index(data, partition_map, args.parts),
        # Build parameters
        'build_params': {
            'knn': args.knn,
            'parts': args.parts,
            'imbalance': args.imbalance,
            'kahip_mode': args.kahip_mode,
            'layers': args.layers,
            'nodes': args.nodes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'data_type': args.type
        }
    }
    
    with open(args.index, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"  Ευρετήριο και μοντέλο αποθηκεύτηκαν: {args.index}")
    
    print("\n" + "=" * 80)
    print("ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
    print("=" * 80)
    print(f"Index file: {args.index}")
    print(f"Data type: {args.type}")
    print(f"Buckets (m): {args.parts}")
    print(f"k-NN: {args.knn}")
    print(f"KaHIP mode: {args.kahip_mode}")
    print(f"MLP layers: {args.layers}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")


if __name__ == '__main__':
    main()
