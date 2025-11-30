"""
Neural LSH Utility Functions

Input/output, mathematical operations, argument validation, and helper functions 
for the Neural LSH implementation.
"""

# Argument validation
from .args import (
    validate_build_args,
    validate_search_args,
    create_build_parser,
    create_search_parser,
    parse_and_validate_build_args,
    parse_and_validate_search_args,
    print_build_summary,
    print_search_summary,
)

# Math utilities
from .math_utils import (
    euclidean_distance,
    euclidean_distances_matrix,
    cosine_similarity,
    hamming_distance,
    manhattan_distance,
    compute_nearest_neighbors,
    compute_range_neighbors,
    compute_recall,
    compute_average_accuracy_fraction,
    compute_qps,
    compute_mrr,
    compute_statistics,
    compute_distance_distribution,
    normalize_vectors_l2,
    standardize_vectors,
)

__all__ = [
    # Argument validation
    'validate_build_args',
    'validate_search_args',
    'create_build_parser',
    'create_search_parser',
    'parse_and_validate_build_args',
    'parse_and_validate_search_args',
    'print_build_summary',
    'print_search_summary',
    
    # Math utilities - distances
    'euclidean_distance',
    'euclidean_distances_matrix',
    'cosine_similarity',
    'hamming_distance',
    'manhattan_distance',
    
    # Search utilities
    'compute_nearest_neighbors',
    'compute_range_neighbors',
    
    # Metrics
    'compute_recall',
    'compute_average_accuracy_fraction',
    'compute_qps',
    'compute_mrr',
    
    # Statistics
    'compute_statistics',
    'compute_distance_distribution',
    
    # Normalization
    'normalize_vectors_l2',
    'standardize_vectors',
]


