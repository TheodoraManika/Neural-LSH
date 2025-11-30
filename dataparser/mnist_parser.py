"""
MNIST Dataset Parser

Parses MNIST (.idx3-ubyte and .idx1-ubyte) format files commonly used in 
machine learning benchmarks. The MNIST dataset contains 28x28 pixel images 
of handwritten digits.

File Formats:
- .idx3-ubyte: Image files (4 bytes magic + 4x4 bytes metadata + pixel data)
- .idx1-ubyte: Label files (4 bytes magic + 4 bytes count + labels)

Magic numbers:
- Images: 2051 (0x00000803)
- Labels: 2049 (0x00000801)
"""

import numpy as np
import struct
from pathlib import Path
from typing import Tuple, Optional


class MNISTParser:
    """Parser for MNIST dataset files (.idx3-ubyte and .idx1-ubyte formats)."""
    
    MAGIC_IMAGE = 2051  # 0x00000803
    MAGIC_LABEL = 2049  # 0x00000801
    
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    
    @staticmethod
    def read_images(filepath: str) -> np.ndarray:
        """
        Read image file in .idx3-ubyte format.
        
        Format:
        - 4 bytes: magic number (2051)
        - 4 bytes: number of images
        - 4 bytes: number of rows (28)
        - 4 bytes: number of columns (28)
        - Variable: image pixel data (0-255 uint8)
        
        Args:
            filepath: Path to the image file
            
        Returns:
            numpy array of shape (N, 28, 28) with uint8 pixel values
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            # Read magic number
            magic_bytes = f.read(4)
            magic = struct.unpack('>I', magic_bytes)[0]
            if magic != MNISTParser.MAGIC_IMAGE:
                raise ValueError(f"Invalid magic number {magic}, expected {MNISTParser.MAGIC_IMAGE}")
            
            # Read dimensions
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            
            # Read all pixel data
            pixel_data = f.read(num_images * num_rows * num_cols)
            
            # Convert to numpy array and reshape
            images = np.frombuffer(pixel_data, dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
        
        return images
    
    @staticmethod
    def flatten_images(images: np.ndarray) -> np.ndarray:
        """
        Flatten images to 1D vectors for machine learning.
        
        Args:
            images: numpy array of shape (N, 28, 28)
            
        Returns:
            numpy array of shape (N, 784) with flattened pixels
        """
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:
            return images.reshape(images.shape[0], -1)
        return images
    
    @staticmethod
    def normalize_images(images: np.ndarray) -> np.ndarray:
        """
        Normalize images to [0, 1] range.
        
        Args:
            images: numpy array with values 0-255
            
        Returns:
            numpy array with values normalized to [0, 1]
        """
        images = np.asarray(images, dtype=np.float32)
        return images / 255.0


def load_mnist_dataset(train_images_path: str = "dataset/train-images.idx3-ubyte",
                       train_labels_path: str = "dataset/train-labels.idx1-ubyte",
                       test_images_path: str = "dataset/t10k-images.idx3-ubyte",
                       test_labels_path: str = "dataset/t10k-labels.idx1-ubyte") -> dict:
    """
    Load complete MNIST dataset (training and test sets).
    
    Note: Label files are optional. If missing, they will be set to None.
    
    Args:
        train_images_path: Path to training images file
        train_labels_path: Path to training labels file (optional)
        test_images_path: Path to test images file
        test_labels_path: Path to test labels file (optional)
        
    Returns:
        Dictionary with keys: 'train_images', 'train_labels', 'test_images', 'test_labels'
        (Label values will be None if files don't exist)
    """
    parser = MNISTParser()
    mnist_data = {}
    
    # Load training data
    train_images_p = Path(train_images_path)
    if train_images_p.exists():
        try:
            mnist_data['train_images'] = parser.read_images(str(train_images_p))
            print(f"✓ Loaded training images: {mnist_data['train_images'].shape}")
        except Exception as e:
            print(f"✗ Failed to load training images: {e}")
    else:
        print(f"✗ Training images file not found: {train_images_p}")
    
    
    # Load test data
    test_images_p = Path(test_images_path)
    if test_images_p.exists():
        try:
            mnist_data['test_images'] = parser.read_images(str(test_images_p))
            print(f"✓ Loaded test images: {mnist_data['test_images'].shape}")
        except Exception as e:
            print(f"✗ Failed to load test images: {e}")
    else:
        print(f"✗ Test images file not found: {test_images_p}")
    
    return mnist_data


if __name__ == "__main__":
    # Example usage
    mnist_data = load_mnist_dataset()
    
    if mnist_data:
        print("\n=== MNIST Dataset Summary ===")
        for key, data in mnist_data.items():
            print(f"{key:20s}: shape={data.shape}, dtype={data.dtype}")
