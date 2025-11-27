"""
k-NN Graph Construction Module
================================
Κατασκευή γράφου k-πλησιέστερων γειτόνων με συμμετρικοποίηση και βάρη ακμών.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


def build_knn_graph(data, k=10):
    """
    Κατασκευή k-NN γράφου με συμμετρικοποίηση και βάρη ακμών.
    
    Args:
        data: numpy array με τα δεδομένα (n_samples, n_features)
        k: αριθμός πλησιέστερων γειτόνων
        
    Returns:
        graph: dictionary {node: set(neighbors)}
        edge_weights: dictionary {(node1, node2): weight}
    """
    n_samples = len(data)
    
    # Εύρεση k-NN για κάθε σημείο
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nbrs.fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Κατασκευή γράφου (χωρίς τον εαυτό του)
    knn_neighbors = {}
    for i in range(n_samples):
        # Αγνοούμε τον πρώτο γείτονα (είναι ο ίδιος ο κόμβος)
        knn_neighbors[i] = set(indices[i][1:])
    
    # Συμμετρικοποίηση και απόδοση βαρών
    graph = defaultdict(set)
    edge_weights = {}
    
    for node in range(n_samples):
        neighbors = knn_neighbors[node]
        
        for neighbor in neighbors:
            # Προσθήκη ακμής (μη κατευθυνόμενος γράφος)
            graph[node].add(neighbor)
            graph[neighbor].add(node)
            
            # Υπολογισμός βάρους
            # Αμοιβαίοι γείτονες (mutual NN): βάρος 2
            # Μονόπλευροι γείτονες: βάρος 1
            edge = tuple(sorted([node, neighbor]))
            
            if edge not in edge_weights:
                # Έλεγχος αν είναι mutual NN
                is_mutual = (neighbor in knn_neighbors[node]) and (node in knn_neighbors[neighbor])
                edge_weights[edge] = 2 if is_mutual else 1
    
    return dict(graph), edge_weights


def convert_graph_to_csr(graph, edge_weights, n_points):
    """
    Μετατροπή γράφου σε CSR (Compressed Sparse Row) μορφή για το KaHIP.
    
    Args:
        graph: dictionary {node: set(neighbors)}
        edge_weights: dictionary {(node1, node2): weight}
        n_points: αριθμός κόμβων
        
    Returns:
        vwgt: λίστα με βάρη κόμβων (συνήθως 1 για όλους)
        xadj: δείκτες αρχής γειτόνων για κάθε κόμβο
        adjncy: λίστα γειτόνων
        adjcwgt: λίστα βαρών ακμών
    """
    # Βάρη κόμβων (όλοι έχουν βάρος 1)
    vwgt = [1] * n_points
    
    # Κατασκευή CSR
    xadj = [0]
    adjncy = []
    adjcwgt = []
    
    for node in range(n_points):
        neighbors = sorted(graph.get(node, []))
        
        for neighbor in neighbors:
            adjncy.append(neighbor)
            
            # Βρες το βάρος της ακμής
            edge = tuple(sorted([node, neighbor]))
            weight = edge_weights.get(edge, 1)
            adjcwgt.append(weight)
        
        xadj.append(len(adjncy))
    
    return vwgt, xadj, adjncy, adjcwgt
