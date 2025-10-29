import torch
import torch.nn as nn

class EnhancedDiabetesModel(nn.Module):
    """
    Enhanced neural network for diabetes detection with Batch Normalization.
    This model is designed for binary classification and is configurable for hyperparameter tuning.
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.4):
        super(EnhancedDiabetesModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer for binary classification (outputs raw logits)
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class CleanDiabetesModel(nn.Module):
    """
    Original multi-class model architecture, kept for reference or future use.
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3, num_classes=4):
        super(CleanDiabetesModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        # Output layer for multi-class
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
