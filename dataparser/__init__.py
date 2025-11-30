"""
Unified Dataset Parser

High-level interface for loading SIFT and MNIST datasets.
Provides convenient functions to load, normalize, and manipulate both dataset types.
"""

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import numpy as np

from .sift_parser import SIFTParser, load_sift_dataset
from .mnist_parser import MNISTParser, load_mnist_dataset


class DatasetLoader:
    """Unified loader for SIFT and MNIST datasets."""
    
    @staticmethod
    def load_sift(base_dir: str = "dataset") -> Dict[str, np.ndarray]:
        """
        Load SIFT dataset.
        
        Args:
            base_dir: Directory containing SIFT files
            
        Returns:
            Dictionary with 'base', 'query', 'learn', 'groundtruth' arrays
        """
        return load_sift_dataset(base_dir)
    
    @staticmethod
    def load_mnist(base_dir: str = "dataset") -> Dict[str, np.ndarray]:
        """
        Load MNIST dataset.
        
        Args:
            base_dir: Directory containing MNIST files
            
        Returns:
            Dictionary with 'train_images', 'train_labels', 'test_images', 'test_labels' arrays
        """
        return load_mnist_dataset(
            f"{base_dir}/train-images.idx3-ubyte",
            f"{base_dir}/train-labels.idx1-ubyte",
            f"{base_dir}/t10k-images.idx3-ubyte",
            f"{base_dir}/t10k-labels.idx1-ubyte"
        )
    
    @staticmethod
    def load_dataset(dataset_type: str, base_dir: str = "dataset") -> Dict[str, np.ndarray]:
        """
        Load either SIFT or MNIST dataset.
        
        Args:
            dataset_type: 'sift' or 'mnist'
            base_dir: Directory containing dataset files
            
        Returns:
            Dictionary with loaded dataset
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type == 'sift':
            return DatasetLoader.load_sift(base_dir)
        elif dataset_type == 'mnist':
            return DatasetLoader.load_mnist(base_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'sift' or 'mnist'.")


class DataPreprocessor:
    """Utilities for preprocessing dataset vectors."""
    
    @staticmethod
    def normalize_vectors(vectors: np.ndarray, method: str = 'l2') -> np.ndarray:
        """
        Normalize vectors using different methods.
        
        Args:
            vectors: numpy array of shape (N, D)
            method: 'l2' (default), 'l1', or 'minmax'
            
        Returns:
            Normalized vectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        
        if method == 'l2':
            # L2 normalization (unit norm)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / (norms + 1e-10)
        
        elif method == 'l1':
            # L1 normalization
            norms = np.linalg.norm(vectors, axis=1, ord=1, keepdims=True)
            return vectors / (norms + 1e-10)
        
        elif method == 'minmax':
            # Min-Max scaling to [0, 1]
            min_vals = vectors.min(axis=0)
            max_vals = vectors.max(axis=0)
            return (vectors - min_vals) / (max_vals - min_vals + 1e-10)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def standardize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Standardize vectors (zero mean, unit variance).
        
        Args:
            vectors: numpy array of shape (N, D)
            
        Returns:
            Standardized vectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        mean = vectors.mean(axis=0)
        std = vectors.std(axis=0)
        return (vectors - mean) / (std + 1e-10)
    
    @staticmethod
    def flatten_mnist(images: np.ndarray) -> np.ndarray:
        """
        Flatten MNIST images to 1D vectors.
        
        Args:
            images: numpy array of shape (N, 28, 28)
            
        Returns:
            Flattened array of shape (N, 784)
        """
        return MNISTParser.flatten_images(images)
    
    @staticmethod
    def normalize_mnist(images: np.ndarray) -> np.ndarray:
        """
        Normalize MNIST image pixel values to [0, 1].
        
        Args:
            images: numpy array with values 0-255
            
        Returns:
            Normalized array with values 0-1
        """
        return MNISTParser.normalize_images(images)



if __name__ == "__main__":
    # Example usage
    import sys
    
    print("Dataset Parser Examples")
    print("=" * 50)
    
    # Example 1: Load SIFT
    print("\n1. Loading SIFT dataset...")
    try:
        sift_data = DatasetLoader.load_sift()
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Load MNIST
    print("\n2. Loading MNIST dataset...")
    try:
        mnist_data = DatasetLoader.load_mnist()
    except Exception as e:
        print(f"Error: {e}")

