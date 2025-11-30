"""
SIFT Dataset Parser

Parses SIFT (.fvecs and .ivecs) format files commonly used in approximate 
nearest neighbor search benchmarks. The SIFT dataset contains feature vectors 
extracted from natural images.

File Formats:
- .fvecs: Float vectors (4-byte dimension + 4-byte floats per value)
- .ivecs: Integer vectors (4-byte dimension + 4-byte integers per value)
"""

import numpy as np
import struct
from pathlib import Path
from typing import Tuple, Optional


class SIFTParser:
    """Parser for SIFT dataset files (.fvecs and .ivecs formats)."""
    
    SIFT_DIMENSION = 128  # Standard SIFT feature dimension
    
    @staticmethod
    def read_fvecs(filepath: str) -> np.ndarray:
        """
        Read a .fvecs file (float vectors).
        
        Format:
        - Each vector starts with a 4-byte integer indicating dimension D
        - Followed by D 4-byte floats
        
        Args:
            filepath: Path to the .fvecs file
            
        Returns:
            numpy array of shape (N, D) where N is number of vectors
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        vectors = []
        
        with open(filepath, 'rb') as f:
            while True:
                # Read dimension (4 bytes)
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                
                dim = struct.unpack('I', dim_bytes)[0]
                
                # Read vector values (4 bytes each)
                vector_data = f.read(dim * 4)
                if len(vector_data) < dim * 4:
                    break
                
                vector = struct.unpack(f'{dim}f', vector_data)
                vectors.append(vector)
        
        return np.array(vectors, dtype=np.float32)
    
    @staticmethod
    def read_ivecs(filepath: str) -> np.ndarray:
        """
        Read a .ivecs file (integer vectors, typically indices).
        
        Format:
        - Each vector starts with a 4-byte integer indicating dimension D
        - Followed by D 4-byte integers
        
        Args:
            filepath: Path to the .ivecs file
            
        Returns:
            numpy array of shape (N, D) where N is number of vectors
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        vectors = []
        
        with open(filepath, 'rb') as f:
            while True:
                # Read dimension (4 bytes)
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                
                dim = struct.unpack('I', dim_bytes)[0]
                
                # Read vector values (4 bytes each as integers)
                vector_data = f.read(dim * 4)
                if len(vector_data) < dim * 4:
                    break
                
                vector = struct.unpack(f'{dim}I', vector_data)
                vectors.append(vector)
        
        return np.array(vectors, dtype=np.int32)

def load_sift_dataset(base_dir: str = "dataset") -> dict:
    """
    Load complete SIFT dataset (base, query, groundtruth, learn sets).
    
    Args:
        base_dir: Directory containing SIFT files
        
    Returns:
        Dictionary with keys: 'base', 'query', 'learn', 'groundtruth'
    """
    base_dir = Path(base_dir)
    parser = SIFTParser()
    
    sift_data = {}
    
    # Load base vectors
    base_path = base_dir / "sift_base.fvecs"
    if base_path.exists():
        sift_data['base'] = parser.read_fvecs(str(base_path))
        print(f"✓ Loaded base vectors: {sift_data['base'].shape}")
    
    # Load query vectors
    query_path = base_dir / "sift_query.fvecs"
    if query_path.exists():
        sift_data['query'] = parser.read_fvecs(str(query_path))
        print(f"✓ Loaded query vectors: {sift_data['query'].shape}")
    
    # Load learn vectors
    learn_path = base_dir / "sift_learn.fvecs"
    if learn_path.exists():
        sift_data['learn'] = parser.read_fvecs(str(learn_path))
        print(f"✓ Loaded learn vectors: {sift_data['learn'].shape}")
    
    # Load groundtruth
    gt_path = base_dir / "sift_groundtruth.ivecs"
    if gt_path.exists():
        sift_data['groundtruth'] = parser.read_ivecs(str(gt_path))
        print(f"✓ Loaded groundtruth: {sift_data['groundtruth'].shape}")
    
    return sift_data


if __name__ == "__main__":
    # Example usage
    sift_data = load_sift_dataset()
    
    if sift_data:
        print("\n=== SIFT Dataset Summary ===")
        for key, data in sift_data.items():
            print(f"{key:12s}: shape={data.shape}, dtype={data.dtype}")
