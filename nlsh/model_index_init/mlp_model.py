"""
Neural-LSH MLP Model
=====================
3-στρωμάτων MLP για την πρόβλεψη των buckets.
"""

import torch
import torch.nn as nn


class NeuralLSH_MLP(nn.Module):
    """
    Multilayer Perceptron για Neural-LSH.
    
    Δυναμική αρχιτεκτονική με configurable αριθμό layers.
    Για 3 layers: Input -> L1 -> ReLU -> L2 -> ReLU -> L3 -> Output
    """
    
    def __init__(self, input_dim, num_layers=3, nodes_per_layer=64, output_dim=100):
        """
        Args:
            input_dim: διάσταση εισόδου (διάσταση διανυσμάτων)
            num_layers: αριθμός γραμμικών στρωμάτων (default: 3)
            nodes_per_layer: αριθμός νευρώνων ανά κρυφό στρώμα (default: 64)
            output_dim: αριθμός buckets (K)
        """
        super(NeuralLSH_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.output_dim = output_dim
        
        # Δημιουργία layers δυναμικά
        layers = []
        
        # Πρώτο layer: input -> hidden
        layers.append(nn.Linear(input_dim, nodes_per_layer))
        layers.append(nn.ReLU())
        
        # Ενδιάμεσα hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            layers.append(nn.ReLU())
        
        # Τελευταίο layer: hidden -> output (χωρίς ReLU)
        if num_layers > 1:
            layers.append(nn.Linear(nodes_per_layer, output_dim))
        else:
            # Αν έχουμε μόνο 1 layer
            layers = [nn.Linear(input_dim, output_dim)]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass του δικτύου.
        
        Args:
            x: input tensor (batch_size, input_dim)
            
        Returns:
            output: logits για κάθε bucket (batch_size, output_dim)
        """
        return self.model(x)
    
    def predict_bucket(self, x):
        """
        Πρόβλεψη του bucket για ένα σύνολο διανυσμάτων.
        
        Args:
            x: input tensor (batch_size, input_dim)
            
        Returns:
            predicted buckets (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            _, predicted = torch.max(logits, 1)
        return predicted
