"""
Index Builder Module
=====================
Κατασκευή ευρετηρίου οργανωμένου ανά bucket.
"""

import numpy as np
from collections import defaultdict


def build_index(data, partition_map, K):
    """
    Κατασκευή ευρετηρίου οργανωμένου ανά bucket.
    
    Args:
        data: numpy array με τα δεδομένα (n_samples, n_features)
        partition_map: numpy array με το bucket για κάθε σημείο
        K: αριθμός buckets
        
    Returns:
        index_data: dictionary με τη δομή του ευρετηρίου
    """
    # Οργάνωση σημείων ανά bucket
    buckets = defaultdict(list)
    
    for idx, bucket_id in enumerate(partition_map):
        buckets[bucket_id].append(idx)
    
    # Μετατροπή σε dictionary με numpy arrays
    index_data = {
        'buckets': {},
        'data': data,
        'partition_map': partition_map,
        'K': K,
        'n_points': len(data),
        'dim': data.shape[1]
    }
    
    for bucket_id in range(K):
        indices = buckets[bucket_id]
        index_data['buckets'][bucket_id] = {
            'indices': np.array(indices),
            'vectors': data[indices] if len(indices) > 0 else np.array([])
        }
    
    return index_data
