"""
Search I/O Module
=================
Functions for loading data and saving results.
"""

import os
import pickle
import numpy as np
from typing import Dict, Tuple


def load_index_and_model(index_path: str) -> Tuple[Dict, 'NeuralLSH_MLP', 'torch.device']:
    """
    Φόρτωση του ευρετηρίου και του εκπαιδευμένου μοντέλου.
    
    Args:
        index_path: Διαδρομή προς το αρχείο του ευρετηρίου
        
    Returns:
        Tuple[index_data, model, device]
    """
    import torch
    from nlsh.model_index_init.mlp_model import NeuralLSH_MLP
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Το αρχείο ευρετηρίου δεν βρέθηκε: {index_path}")
    
    print(f"Φόρτωση ευρετηρίου από: {index_path}")
    
    with open(index_path, 'rb') as f:
        index_data = pickle.load(f)
    
    # Αναδημιουργία του μοντέλου
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = index_data['model_config']
    model = NeuralLSH_MLP(
        input_dim=model_config['input_dim'],
        num_layers=model_config['num_layers'],
        nodes_per_layer=model_config['nodes_per_layer'],
        output_dim=model_config['output_dim']
    )
    
    # Φόρτωση των βαρών του μοντέλου
    model.load_state_dict(index_data['model_state_dict'])
    model = model.to(device)
    model.eval()  # Θέσιμο του μοντέλου σε κατάσταση αξιολόγησης
    
    print(f"Μοντέλο φορτώθηκε (device: {device})")
    print(f"  Input dim: {model_config['input_dim']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Buckets: {model_config['output_dim']}")
    
    return index_data, model, device


def load_data(data_path: str) -> np.ndarray:
    """
    Φόρτωση του συνόλου δεδομένων (απαραίτητο για υπολογισμό πραγματικών αποστάσεων).
    
    Args:
        data_path: Διαδρομή προς το αρχείο δεδομένων
        
    Returns:
        numpy array με τα δεδομένα (n_samples, dim)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Το αρχείο δεδομένων δεν βρέθηκε: {data_path}")
    
    print(f"Φόρτωση δεδομένων από: {data_path}")
    
    if data_path.endswith('.npy'):
        data = np.load(data_path)
    else:
        raise NotImplementedError(f"Μορφή αρχείου {data_path} δεν υποστηρίζεται. "
                                  "Χρησιμοποιήστε .npy αρχεία.")
    
    print(f"Φορτώθηκαν {len(data)} δεδομένα με διάσταση {data.shape[1]}")
    return data


def load_queries(queries_path: str) -> np.ndarray:
    """
    Φόρτωση των ερωτημάτων (queries).
    
    Args:
        queries_path: Διαδρομή προς το αρχείο με τα queries
        
    Returns:
        numpy array με τα queries (n_queries, dim)
    """
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Το αρχείο queries δεν βρέθηκε: {queries_path}")
    
    print(f"Φόρτωση queries από: {queries_path}")
    
    if queries_path.endswith('.npy'):
        queries = np.load(queries_path)
    else:
        raise NotImplementedError(f"Μορφή αρχείου {queries_path} δεν υποστηρίζεται. "
                                  "Χρησιμοποιήστε .npy αρχεία.")
    
    print(f"Φορτώθηκαν {len(queries)} queries με διάσταση {queries.shape[1]}")
    return queries


def save_results(results: list, output_path: str, method_name: str = 'Neural LSH') -> None:
    """
    Αποθήκευση αποτελεσμάτων σε αρχείο με τη μορφή που απαιτεί η εργασία.
    
    Μορφή εξόδου:
    <METHOD NAME> Neural LSH
    Query: image_id_in_query_set
    Nearest neighbor-1: image_id_in_data_set
    distanceApproximate: <double>
    distanceTrue: <double>
    ...
    R-near neighbors:
    image_id_A
    ...
    Average AF: <double>
    Recall@N: <double>
    QPS: <double>
    tApproximateAverage: <double>
    tTrueAverage: <double>
    
    Args:
        results: List με τα αποτελέσματα
        output_path: Διαδρομή αποθήκευσης
        method_name: Όνομα της μεθόδου (default: 'Neural LSH')
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Αποθήκευση σε μορφή εργασίας
    with open(output_path, 'w') as f:
        f.write(f"<METHOD NAME> {method_name}\n")
        
        for result in results:
            query_id = result['query_id']
            f.write(f"Query: {query_id}\n")
            
            # Γραφή k-NN γειτόνων
            for rank in range(len(result.get('neighbor_indices', []))):
                neighbor_id = result['neighbor_indices'][rank]
                approx_dist = result['neighbor_distances'][rank]
                true_dist = result['true_distances'][rank]
                
                f.write(f"Nearest neighbor-{rank + 1}: {neighbor_id}\n")
                f.write(f"distanceApproximate: {approx_dist:.6f}\n")
                f.write(f"distanceTrue: {true_dist:.6f}\n")
            
            # Γραφή range neighbors
            f.write("R-near neighbors:\n")
            for range_neighbor_id in result.get('range_neighbors', []):
                f.write(f"{range_neighbor_id}\n")
            
            # Γραφή μετρικών
            f.write(f"Average AF: {result.get('avg_af', 0.0):.6f}\n")
            f.write(f"Recall@N: {result.get('recall', 0.0):.6f}\n")
            f.write(f"QPS: {result.get('qps', 0.0):.6f}\n")
            f.write(f"tApproximateAverage: {result.get('t_approx', 0.0):.6f}\n")
            f.write(f"tTrueAverage: {result.get('t_true', 0.0):.6f}\n")
            f.write("\n")
    
    # Αποθήκευση ως pickle για ευκολία φόρτωσης στο Python
    with open(output_path + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nΑποτελέσματα αποθηκεύτηκαν:")
    print(f"  - Μορφή εργασίας: {output_path}")
    print(f"  - Pickle format: {output_path}.pkl")
