# mine_network.py
"""
Define MINE network structures, supporting both Float and Index inputs for X and Y
"""
import torch
import torch.nn as nn

def _build_mlp(input_dim, hidden_dims, output_dim=1, activation="relu", batch_norm=True, dropout=0.1):
    """Helper function to build MLP layers"""
    layers = []
    prev_dim = input_dim
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim))  # Input is [N, C]

        # Activation function
        if activation.lower() == 'relu':
            layers.append(nn.ReLU())
        elif activation.lower() == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation.lower() == 'elu':
            layers.append(nn.ELU())
        else:
            layers.append(nn.ReLU())  # Default to ReLU

        if dropout > 0 and i < len(hidden_dims) - 1:  # Usually no dropout after the last hidden layer
            layers.append(nn.Dropout(dropout))

        prev_dim = dim

    # Output layer
    network = nn.Sequential(*layers)
    output_layer = nn.Linear(prev_dim, output_dim)

    # Initialize output layer weights for stable early training
    nn.init.normal_(output_layer.weight, std=0.01)
    nn.init.constant_(output_layer.bias, 0)

    return network, output_layer


class MINENetworkBase(nn.Module):
    """Base MINE Network with common functionality"""
    
    def __init__(self):
        super().__init__()
    
    def _reshape_inputs(self, x, y):
        """
        Reshape inputs to 2D for network processing
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, x_dim] or [batch_size, seq_len]
               (or [batch_size, x_dim] or [batch_size] for non-sequential data)
            y: Input tensor of shape [batch_size, seq_len, y_dim] or [batch_size, seq_len]
               (or [batch_size, y_dim] or [batch_size] for non-sequential data)
               
        Returns:
            Flattened tensors suitable for network processing
        """
        # Handle x
        if x.dim() == 3:  # [B, S, Dx]
            B, S, _ = x.shape
            N = B * S
            x_flat = x.reshape(N, -1)  # [N, Dx]
        elif x.dim() == 2 and not hasattr(self, 'x_embedding'):  # [B, Dx] for float features
            N = x.shape[0]
            x_flat = x  # Already [N, Dx]
        elif x.dim() == 2 and hasattr(self, 'x_embedding'):  # [B, S] for index features
            B, S = x.shape
            N = B * S
            x_flat = x.reshape(N)  # [N]
        elif x.dim() == 1:  # [B] for index features
            N = x.shape[0]
            x_flat = x  # Already [N]
        else:
            raise ValueError(f"Unsupported x dimension: {x.dim()}")
            
        # Handle y
        if y.dim() == 3:  # [B, S, Dy]
            if x.dim() == 3:
                assert y.shape[0] == B and y.shape[1] == S, "Batch and sequence dimensions mismatch between x and y"
            y_flat = y.reshape(N, -1)  # [N, Dy]
        elif y.dim() == 2 and not hasattr(self, 'y_embedding'):  # [B, Dy] for float features
            assert y.shape[0] == N, "Batch dimension mismatch between x and y"
            y_flat = y  # Already [N, Dy]
        elif y.dim() == 2 and hasattr(self, 'y_embedding'):  # [B, S] for index features
            if x.dim() == 2:
                assert y.shape[0] == x.shape[0] and y.shape[1] == x.shape[1], "Dimensions mismatch between x and y"
            y_flat = y.reshape(N)  # [N]
        elif y.dim() == 1:  # [B] for index features
            assert y.shape[0] == N, "Batch dimension mismatch between x and y"
            y_flat = y  # Already [N]
        else:
            raise ValueError(f"Unsupported y dimension: {y.dim()}")
            
        return x_flat, y_flat, N


class MINENetworkFloatFloat(MINENetworkBase):
    """MINE network - processes float type inputs for both x and y"""

    def __init__(self, x_dim, y_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1):
        """
        Initialize Float-Float MINE network
        
        Args:
            x_dim: x feature dimension
            y_dim: y feature dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        mlp_input_dim = x_dim + y_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )

    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x: Float input, shape [batch_size, seq_len, x_dim] or [batch_size, x_dim]
            y: Float input, shape [batch_size, seq_len, y_dim] or [batch_size, y_dim]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Concatenate features
        combined = torch.cat([x_flat, y_flat], dim=1)  # [N, x_dim + y_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output


class MINENetworkIndexFloat(MINENetworkBase):
    """MINE network - processes index type x and float type y inputs"""

    def __init__(self, vocab_size, embedding_dim, y_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1):
        """
        Initialize Index-Float MINE network
        
        Args:
            vocab_size: Size of vocabulary for x (for Embedding)
            embedding_dim: Embedding dimension for x
            y_dim: y feature dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.y_dim = y_dim

        # Embedding layer for x
        self.x_embedding = nn.Embedding(vocab_size, embedding_dim)

        # MLP input dimension is embedding output dimension + y dimension
        mlp_input_dim = embedding_dim + y_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )

    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x: Long input (indices), shape [batch_size, seq_len] or [batch_size]
            y: Float input, shape [batch_size, seq_len, y_dim] or [batch_size, y_dim]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Pass x indices through Embedding layer
        x_embedded = self.x_embedding(x_flat)  # [N, embedding_dim]

        # Concatenate embedded x and y
        combined = torch.cat([x_embedded, y_flat], dim=1)  # [N, embedding_dim + y_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output


class MINENetworkFloatIndex(MINENetworkBase):
    """MINE network - processes float type x and index type y inputs"""

    def __init__(self, x_dim, vocab_size, embedding_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1):
        """
        Initialize Float-Index MINE network
        
        Args:
            x_dim: x feature dimension
            vocab_size: Size of vocabulary for y (for Embedding)
            embedding_dim: Embedding dimension for y
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.x_dim = x_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embedding layer for y
        self.y_embedding = nn.Embedding(vocab_size, embedding_dim)

        # MLP input dimension is x dimension + embedding output dimension
        mlp_input_dim = x_dim + embedding_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )

    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x: Float input, shape [batch_size, seq_len, x_dim] or [batch_size, x_dim]
            y: Long input (indices), shape [batch_size, seq_len] or [batch_size]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Pass y indices through Embedding layer
        y_embedded = self.y_embedding(y_flat)  # [N, embedding_dim]

        # Concatenate x and embedded y
        combined = torch.cat([x_flat, y_embedded], dim=1)  # [N, x_dim + embedding_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output


class MINENetworkIndexIndex(MINENetworkBase):
    """MINE network - processes index type inputs for both x and y"""

    def __init__(self, x_vocab_size, x_embedding_dim, y_vocab_size, y_embedding_dim, 
                 hidden_dims=[512, 512, 256], activation="relu", batch_norm=True, dropout=0.1):
        """
        Initialize Index-Index MINE network
        
        Args:
            x_vocab_size: Size of vocabulary for x (for Embedding)
            x_embedding_dim: Embedding dimension for x
            y_vocab_size: Size of vocabulary for y (for Embedding)
            y_embedding_dim: Embedding dimension for y
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.x_vocab_size = x_vocab_size
        self.x_embedding_dim = x_embedding_dim
        self.y_vocab_size = y_vocab_size
        self.y_embedding_dim = y_embedding_dim

        # Embedding layers for x and y
        self.x_embedding = nn.Embedding(x_vocab_size, x_embedding_dim)
        self.y_embedding = nn.Embedding(y_vocab_size, y_embedding_dim)

        # MLP input dimension is sum of embedding output dimensions
        mlp_input_dim = x_embedding_dim + y_embedding_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )

    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x: Long input (indices), shape [batch_size, seq_len] or [batch_size]
            y: Long input (indices), shape [batch_size, seq_len] or [batch_size]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Pass indices through Embedding layers
        x_embedded = self.x_embedding(x_flat)  # [N, x_embedding_dim]
        y_embedded = self.y_embedding(y_flat)  # [N, y_embedding_dim]

        # Concatenate embedded features
        combined = torch.cat([x_embedded, y_embedded], dim=1)  # [N, x_embedding_dim + y_embedding_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output