"""
Neural-LSH Search Script
=========================
Υπεύθυνο για την αναζήτηση k-NN χρησιμοποιώντας το εκπαιδευμένο μοντέλο και το ευρετήριο.

Modular Architecture:
  - search_algorithms.py: Core search operations (4-step algorithm)
  - search_io.py: Data loading and result saving
  - search_output.py: Statistics and logging
  - nlsh_search.py: Main entry point and orchestration
"""

import sys
import time
import numpy as np
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_and_validate_search_args, print_search_summary
from search.search_io import load_index_and_model, load_queries, load_data, save_results
from search.search_algorithms import search_all_queries
from search.search_output import print_summary, log_search_experiment


def main():
    """Κύρια συνάρτηση για την αναζήτηση."""
    # Χρήση utility function για parsing και validation των arguments
    args = parse_and_validate_search_args()
    
    # Ορισμός seed για αναπαραγωγιμότητα
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 80)
    print("NEURAL-LSH SEARCH: Αναζήτηση k-NN με το Εκπαιδευμένο Μοντέλο")
    print("=" * 80)
    
    # Εκτύπωση parameters summary με utility function
    print_search_summary(args)
    
    try:
        # =========================================================================
        # Φόρτωση Ευρετηρίου και Μοντέλου
        # =========================================================================
        print("\n[ΒΗΜΑ 1] Φόρτωση Ευρετηρίου και Μοντέλου...")
        start_time = time.time()
        index_data, model, device = load_index_and_model(args.index)
        load_time = time.time() - start_time
        print(f"  Χρόνος φόρτωσης ευρετηρίου: {load_time:.3f}s")
        
        # =========================================================================
        # Φόρτωση Δεδομένων (απαραίτητο για πραγματικές αποστάσεις)
        # =========================================================================
        print("\n[ΒΗΜΑ 2] Φόρτωση Συνόλου Δεδομένων...")
        data_start = time.time()
        dataset = load_data(args.data)
        data_load_time = time.time() - data_start
        print(f"  Χρόνος φόρτωσης δεδομένων: {data_load_time:.3f}s")
        
        # =========================================================================
        # Φόρτωση Queries
        # =========================================================================
        print("\n[ΒΗΜΑ 3] Φόρτωση Ερωτημάτων...")
        queries_start = time.time()
        queries = load_queries(args.query)
        queries_load_time = time.time() - queries_start
        print(f"  Χρόνος φόρτωσης queries: {queries_load_time:.3f}s")
        
        # =========================================================================
        # Αναζήτηση για όλα τα Queries
        # =========================================================================
        print("\n[ΒΗΜΑ 4] Εκτέλεση Αναζήτησης...")
        search_start = time.time()
        results = search_all_queries(
            model, queries, index_data, dataset, device,
            k=args.N,
            T=args.T,
            radius=args.R
        )
        search_time = time.time() - search_start
        print(f"  Συνολικός χρόνος αναζήτησης: {search_time:.3f}s")
        
        # =========================================================================
        # Αποθήκευση Αποτελεσμάτων
        # =========================================================================
        print("\n[ΒΗΜΑ 5] Αποθήκευση Αποτελεσμάτων...")
        save_results(results, args.output, method_name='Neural LSH')
        
        # =========================================================================
        # Εκτύπωση Σύνοψης με χρήση utility functions
        # =========================================================================
        print_summary(results, index_data['build_params'], search_time, len(queries))
        
        # =========================================================================
        # Logging με utility function
        # =========================================================================
        log_search_experiment(args, results, search_time)
        
        print("\nΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
        
    except Exception as e:
        print(f"\n✗ Σφάλμα κατά την εκτέλεση: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
