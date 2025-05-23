# mine_network.py
"""
Define MINE network structures with projection layers, supporting both Float and Index inputs for X and Y with context windowing
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

def _build_projection_layer(input_dim, proj_dim, activation="relu", batch_norm=True, dropout=0.1):
    """
    Helper function to build projection layer for dimensionality reduction
    
    Args:
        input_dim: Input feature dimension
        proj_dim: Output projection dimension
        activation: Activation function
        batch_norm: Whether to use batch normalization
        dropout: Dropout rate
        
    Returns:
        Projection layer (nn.Sequential)
    """
    layers = []
    layers.append(nn.Linear(input_dim, proj_dim))
    
    if batch_norm:
        layers.append(nn.BatchNorm1d(proj_dim))
    
    # Activation function
    if activation.lower() == 'relu':
        layers.append(nn.ReLU())
    elif activation.lower() == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation.lower() == 'elu':
        layers.append(nn.ELU())
    else:
        layers.append(nn.ReLU())  # Default to ReLU
    
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    
    # Initialize weights for stable training
    projection = nn.Sequential(*layers)
    nn.init.xavier_uniform_(projection[0].weight)  # First layer is Linear
    nn.init.constant_(projection[0].bias, 0)
    
    return projection


class MINENetworkBase(nn.Module):
    """Base MINE Network with common functionality"""
    
    def __init__(self):
        super().__init__()
    
    def _reshape_inputs(self, x, y):
        """
        Reshape inputs to 2D for network processing, supporting context windowing
        
        Args:
            x: Input tensor of various shapes:
               - Float features: [batch_size, seq_len, x_dim] or [batch_size, x_dim * context_frames]
               - Index features: [batch_size, seq_len] or [batch_size, seq_len, context_frames] or [batch_size, context_frames]
            y: Input tensor of various shapes (similar to x)
               
        Returns:
            Flattened tensors suitable for network processing
        """
        # Handle x
        if x.dim() == 3:
            B, S, D = x.shape
            N = B * S
            if hasattr(self, 'x_embedding'):
                # Index features with context: [B, S, context_frames] -> [N, context_frames]
                x_flat = x.reshape(N, D)  # [N, context_frames]
            else:
                # Float features: [B, S, Dx] -> [N, Dx] (Dx may include context expansion)
                x_flat = x.reshape(N, D)  # [N, Dx]
        elif x.dim() == 2:
            if hasattr(self, 'x_embedding'):
                # Index features: could be [B, S] (no context) or [B, context_frames] (with context)
                B, D = x.shape
                N = B
                if D == 1:
                    # Single index per sample: [B, 1] -> [B]
                    x_flat = x.squeeze(1)  # [B]
                else:
                    # Multiple indices per sample (context): [B, context_frames] -> [B, context_frames]
                    x_flat = x  # [B, context_frames]
            else:
                # Float features: [B, Dx] (Dx may include context expansion)
                N = x.shape[0]
                x_flat = x  # Already [N, Dx]
        elif x.dim() == 1:
            # Single index per sample: [B]
            N = x.shape[0]
            x_flat = x  # Already [N]
        else:
            raise ValueError(f"Unsupported x dimension: {x.dim()}")
            
        # Handle y (similar logic as x)
        if y.dim() == 3:
            B_y, S_y, D_y = y.shape
            N_y = B_y * S_y
            if hasattr(self, 'y_embedding'):
                # Index features with context: [B, S, context_frames] -> [N, context_frames]
                y_flat = y.reshape(N_y, D_y)  # [N, context_frames]
            else:
                # Float features: [B, S, Dy] -> [N, Dy] (Dy may include context expansion)
                y_flat = y.reshape(N_y, D_y)  # [N, Dy]
        elif y.dim() == 2:
            if hasattr(self, 'y_embedding'):
                # Index features: could be [B, S] (no context) or [B, context_frames] (with context)
                B_y, D_y = y.shape
                N_y = B_y
                if D_y == 1:
                    # Single index per sample: [B, 1] -> [B]
                    y_flat = y.squeeze(1)  # [B]
                else:
                    # Multiple indices per sample (context): [B, context_frames] -> [B, context_frames]
                    y_flat = y  # [B, context_frames]
            else:
                # Float features: [B, Dy] (Dy may include context expansion)
                N_y = y.shape[0]
                y_flat = y  # Already [N, Dy]
        elif y.dim() == 1:
            # Single index per sample: [B]
            N_y = y.shape[0]
            y_flat = y  # Already [N]
        else:
            raise ValueError(f"Unsupported y dimension: {y.dim()}")
        
        # Consistency checks
        if x.dim() == 3 and y.dim() == 3:
            # Both are 3D, check batch and sequence dimensions match
            assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1], \
                "Batch and sequence dimensions mismatch between x and y"
            N = N_y  # Use common N
        elif x.dim() <= 2 and y.dim() <= 2:
            # Both are 2D or less, check batch dimensions match
            if hasattr(self, 'x_embedding') and x_flat.dim() == 2:
                batch_x = x_flat.shape[0]
            elif hasattr(self, 'x_embedding') and x_flat.dim() == 1:
                batch_x = x_flat.shape[0]
            else:
                batch_x = x_flat.shape[0]
                
            if hasattr(self, 'y_embedding') and y_flat.dim() == 2:
                batch_y = y_flat.shape[0]
            elif hasattr(self, 'y_embedding') and y_flat.dim() == 1:
                batch_y = y_flat.shape[0]
            else:
                batch_y = y_flat.shape[0]
                
            assert batch_x == batch_y, f"Batch dimension mismatch between x ({batch_x}) and y ({batch_y})"
            N = batch_x
        else:
            # Mixed dimensions, use the larger N
            N = max(N if 'N' in locals() else x_flat.shape[0], N_y if 'N_y' in locals() else y_flat.shape[0])
            
        return x_flat, y_flat, N


class MINENetworkFloatFloat(MINENetworkBase):
    """MINE network - processes float type inputs for both x and y with optional projection layers"""

    def __init__(self, x_dim, y_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1,
                 x_proj_dim=None, y_proj_dim=None):
        """
        Initialize Float-Float MINE network with optional projection layers
        
        Args:
            x_dim: x feature dimension (may be expanded by context windowing)
            y_dim: y feature dimension (may be expanded by context windowing)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            x_proj_dim: Projection dimension for x (None to disable projection)
            y_proj_dim: Projection dimension for y (None to disable projection)
        """
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_proj_dim = x_proj_dim
        self.y_proj_dim = y_proj_dim

        # Create projection layers if specified
        if x_proj_dim is not None:
            self.x_projection = _build_projection_layer(x_dim, x_proj_dim, activation, batch_norm, dropout)
            final_x_dim = x_proj_dim
            print(f"  Added X projection layer: {x_dim} -> {x_proj_dim}")
        else:
            self.x_projection = None
            final_x_dim = x_dim
            
        if y_proj_dim is not None:
            self.y_projection = _build_projection_layer(y_dim, y_proj_dim, activation, batch_norm, dropout)
            final_y_dim = y_proj_dim
            print(f"  Added Y projection layer: {y_dim} -> {y_proj_dim}")
        else:
            self.y_projection = None
            final_y_dim = y_dim

        # MLP operates on projected dimensions
        mlp_input_dim = final_x_dim + final_y_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )
        
        print(f"  MLP input dimension: {mlp_input_dim} (x: {final_x_dim} + y: {final_y_dim})")

    def forward(self, x, y):
        """
        Forward pass with optional projection
        
        Args:
            x: Float input, shape [batch_size, seq_len, x_dim] or [batch_size, x_dim * context_frames]
            y: Float input, shape [batch_size, seq_len, y_dim] or [batch_size, y_dim * context_frames]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Apply projections if available
        if self.x_projection is not None:
            x_projected = self.x_projection(x_flat)  # [N, x_proj_dim]
        else:
            x_projected = x_flat  # [N, x_dim]
            
        if self.y_projection is not None:
            y_projected = self.y_projection(y_flat)  # [N, y_proj_dim]
        else:
            y_projected = y_flat  # [N, y_dim]

        # Concatenate projected features
        combined = torch.cat([x_projected, y_projected], dim=1)  # [N, final_x_dim + final_y_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output


class MINENetworkIndexFloat(MINENetworkBase):
    """MINE network - processes index type x and float type y inputs with optional y projection"""

    def __init__(self, vocab_size, embedding_dim, y_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1, y_proj_dim=None, context_frame_numbers=1):
        """
        Initialize Index-Float MINE network with optional y projection
        
        Args:
            vocab_size: Size of vocabulary for x (for Embedding)
            embedding_dim: Embedding dimension for x (single frame)
            y_dim: y feature dimension (may be expanded by context windowing)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            y_proj_dim: Projection dimension for y (None to disable projection)
            context_frame_numbers: Number of context frames (affects final embedding dimension)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.y_dim = y_dim
        self.y_proj_dim = y_proj_dim
        self.context_frame_numbers = context_frame_numbers

        # Embedding layer for x (single frame)
        self.x_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Create projection layer for y if specified
        if y_proj_dim is not None:
            self.y_projection = _build_projection_layer(y_dim, y_proj_dim, activation, batch_norm, dropout)
            final_y_dim = y_proj_dim
            print(f"  Added Y projection layer: {y_dim} -> {y_proj_dim}")
        else:
            self.y_projection = None
            final_y_dim = y_dim

        # Calculate effective x embedding dimension considering context
        effective_x_embed_dim = embedding_dim * context_frame_numbers
        
        # MLP input dimension includes context-expanded embedding dimension
        mlp_input_dim = effective_x_embed_dim + final_y_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )
        
        print(f"  X embedding: {embedding_dim} per frame × {context_frame_numbers} frames = {effective_x_embed_dim}")
        print(f"  MLP input dimension: {mlp_input_dim} (x_embed: {effective_x_embed_dim} + y: {final_y_dim})")

    def forward(self, x, y):
        """
        Forward pass with optional y projection
        
        Args:
            x: Long input (indices), shape [batch_size, seq_len] or [batch_size] or [batch_size, context_frames]
            y: Float input, shape [batch_size, seq_len, y_dim] or [batch_size, y_dim * context_frames]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Pass x indices through Embedding layer
        if x_flat.dim() == 1:
            # Single index per sample: [N] -> [N, embedding_dim]
            x_embedded = self.x_embedding(x_flat)  # [N, embedding_dim]
            # If context_frame_numbers > 1 but we only have single indices, 
            # this means context windowing was not applied (shouldn't happen in normal cases)
            if self.context_frame_numbers > 1:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got single indices")
        elif x_flat.dim() == 2:
            # Multiple indices per sample (context): [N, context_frames] -> [N, embedding_dim * context_frames]
            N, context_frames = x_flat.shape
            if context_frames != self.context_frame_numbers:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got {context_frames}")
            
            # Embed each frame and concatenate
            x_embedded_frames = []
            for i in range(context_frames):
                frame_embedded = self.x_embedding(x_flat[:, i])  # [N, embedding_dim]
                x_embedded_frames.append(frame_embedded)
            x_embedded = torch.cat(x_embedded_frames, dim=1)  # [N, embedding_dim * context_frames]
        else:
            raise ValueError(f"Unexpected x_flat dimension: {x_flat.dim()}")

        # Apply projection to y if available
        if self.y_projection is not None:
            y_projected = self.y_projection(y_flat)  # [N, y_proj_dim]
        else:
            y_projected = y_flat  # [N, y_dim]

        # Concatenate embedded x and projected y
        combined = torch.cat([x_embedded, y_projected], dim=1)  # [N, final_x_dim + final_y_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output


class MINENetworkFloatIndex(MINENetworkBase):
    """MINE network - processes float type x and index type y inputs with optional x projection"""

    def __init__(self, x_dim, vocab_size, embedding_dim, hidden_dims=[512, 512, 256],
                 activation="relu", batch_norm=True, dropout=0.1, x_proj_dim=None, context_frame_numbers=1):
        """
        Initialize Float-Index MINE network with optional x projection
        
        Args:
            x_dim: x feature dimension (may be expanded by context windowing)
            vocab_size: Size of vocabulary for y (for Embedding)
            embedding_dim: Embedding dimension for y (single frame)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            x_proj_dim: Projection dimension for x (None to disable projection)
            context_frame_numbers: Number of context frames (affects final embedding dimension)
        """
        super().__init__()
        self.x_dim = x_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_proj_dim = x_proj_dim
        self.context_frame_numbers = context_frame_numbers

        # Create projection layer for x if specified
        if x_proj_dim is not None:
            self.x_projection = _build_projection_layer(x_dim, x_proj_dim, activation, batch_norm, dropout)
            final_x_dim = x_proj_dim
            print(f"  Added X projection layer: {x_dim} -> {x_proj_dim}")
        else:
            self.x_projection = None
            final_x_dim = x_dim

        # Embedding layer for y (single frame)
        self.y_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Calculate effective y embedding dimension considering context
        effective_y_embed_dim = embedding_dim * context_frame_numbers

        # MLP input dimension includes context-expanded embedding dimension
        mlp_input_dim = final_x_dim + effective_y_embed_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )
        
        print(f"  Y embedding: {embedding_dim} per frame × {context_frame_numbers} frames = {effective_y_embed_dim}")
        print(f"  MLP input dimension: {mlp_input_dim} (x: {final_x_dim} + y_embed: {effective_y_embed_dim})")

    def forward(self, x, y):
        """
        Forward pass with optional x projection
        
        Args:
            x: Float input, shape [batch_size, seq_len, x_dim] or [batch_size, x_dim * context_frames]
            y: Long input (indices), shape [batch_size, seq_len] or [batch_size] or [batch_size, context_frames]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Apply projection to x if available
        if self.x_projection is not None:
            x_projected = self.x_projection(x_flat)  # [N, x_proj_dim]
        else:
            x_projected = x_flat  # [N, x_dim]

        # Pass y indices through Embedding layer
        if y_flat.dim() == 1:
            # Single index per sample: [N] -> [N, embedding_dim]
            y_embedded = self.y_embedding(y_flat)  # [N, embedding_dim]
            # If context_frame_numbers > 1 but we only have single indices, 
            # this means context windowing was not applied (shouldn't happen in normal cases)
            if self.context_frame_numbers > 1:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got single indices")
        elif y_flat.dim() == 2:
            # Multiple indices per sample (context): [N, context_frames] -> [N, embedding_dim * context_frames]
            N, context_frames = y_flat.shape
            if context_frames != self.context_frame_numbers:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got {context_frames}")
            
            # Embed each frame and concatenate
            y_embedded_frames = []
            for i in range(context_frames):
                frame_embedded = self.y_embedding(y_flat[:, i])  # [N, embedding_dim]
                y_embedded_frames.append(frame_embedded)
            y_embedded = torch.cat(y_embedded_frames, dim=1)  # [N, embedding_dim * context_frames]
        else:
            raise ValueError(f"Unexpected y_flat dimension: {y_flat.dim()}")

        # Concatenate projected x and embedded y
        combined = torch.cat([x_projected, y_embedded], dim=1)  # [N, final_x_dim + final_y_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output


class MINENetworkIndexIndex(MINENetworkBase):
    """MINE network - processes index type inputs for both x and y with context windowing support"""

    def __init__(self, x_vocab_size, x_embedding_dim, y_vocab_size, y_embedding_dim, 
                 hidden_dims=[512, 512, 256], activation="relu", batch_norm=True, dropout=0.1, context_frame_numbers=1):
        """
        Initialize Index-Index MINE network
        
        Args:
            x_vocab_size: Size of vocabulary for x (for Embedding)
            x_embedding_dim: Embedding dimension for x (single frame)
            y_vocab_size: Size of vocabulary for y (for Embedding)
            y_embedding_dim: Embedding dimension for y (single frame)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            context_frame_numbers: Number of context frames (affects final embedding dimensions)
            
        Note: No projection layers needed as embeddings already provide dimensionality control
        """
        super().__init__()
        self.x_vocab_size = x_vocab_size
        self.x_embedding_dim = x_embedding_dim
        self.y_vocab_size = y_vocab_size
        self.y_embedding_dim = y_embedding_dim
        self.context_frame_numbers = context_frame_numbers

        # Embedding layers for x and y (single frame)
        self.x_embedding = nn.Embedding(x_vocab_size, x_embedding_dim)
        self.y_embedding = nn.Embedding(y_vocab_size, y_embedding_dim)

        # Calculate effective embedding dimensions considering context
        effective_x_embed_dim = x_embedding_dim * context_frame_numbers
        effective_y_embed_dim = y_embedding_dim * context_frame_numbers

        # MLP input dimension is sum of context-expanded embedding output dimensions
        mlp_input_dim = effective_x_embed_dim + effective_y_embed_dim
        self.network, self.output_layer = _build_mlp(
            mlp_input_dim, hidden_dims, 1, activation, batch_norm, dropout
        )
        
        print(f"  X embedding: {x_embedding_dim} per frame × {context_frame_numbers} frames = {effective_x_embed_dim}")
        print(f"  Y embedding: {y_embedding_dim} per frame × {context_frame_numbers} frames = {effective_y_embed_dim}")
        print(f"  MLP input dimension: {mlp_input_dim} (x_embed: {effective_x_embed_dim} + y_embed: {effective_y_embed_dim})")

    def forward(self, x, y):
        """
        Forward pass
        
        Args:
            x: Long input (indices), shape [batch_size, seq_len] or [batch_size] or [batch_size, context_frames]
            y: Long input (indices), shape [batch_size, seq_len] or [batch_size] or [batch_size, context_frames]
            
        Returns:
            Network output T(x, y), shape [batch_size * seq_len, 1] or [batch_size, 1]
        """
        # Reshape inputs
        x_flat, y_flat, _ = self._reshape_inputs(x, y)

        # Pass x indices through Embedding layer
        if x_flat.dim() == 1:
            # Single index per sample: [N] -> [N, x_embedding_dim]
            x_embedded = self.x_embedding(x_flat)  # [N, x_embedding_dim]
            # If context_frame_numbers > 1 but we only have single indices, 
            # this means context windowing was not applied (shouldn't happen in normal cases)
            if self.context_frame_numbers > 1:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got single indices for x")
        elif x_flat.dim() == 2:
            # Multiple indices per sample (context): [N, context_frames] -> [N, x_embedding_dim * context_frames]
            N, context_frames = x_flat.shape
            if context_frames != self.context_frame_numbers:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got {context_frames} for x")
            
            # Embed each frame and concatenate
            x_embedded_frames = []
            for i in range(context_frames):
                frame_embedded = self.x_embedding(x_flat[:, i])  # [N, x_embedding_dim]
                x_embedded_frames.append(frame_embedded)
            x_embedded = torch.cat(x_embedded_frames, dim=1)  # [N, x_embedding_dim * context_frames]
        else:
            raise ValueError(f"Unexpected x_flat dimension: {x_flat.dim()}")

        # Pass y indices through Embedding layer
        if y_flat.dim() == 1:
            # Single index per sample: [N] -> [N, y_embedding_dim]
            y_embedded = self.y_embedding(y_flat)  # [N, y_embedding_dim]
            # If context_frame_numbers > 1 but we only have single indices, 
            # this means context windowing was not applied (shouldn't happen in normal cases)
            if self.context_frame_numbers > 1:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got single indices for y")
        elif y_flat.dim() == 2:
            # Multiple indices per sample (context): [N, context_frames] -> [N, y_embedding_dim * context_frames]
            N, context_frames = y_flat.shape
            if context_frames != self.context_frame_numbers:
                print(f"Warning: Expected {self.context_frame_numbers} context frames but got {context_frames} for y")
            
            # Embed each frame and concatenate
            y_embedded_frames = []
            for i in range(context_frames):
                frame_embedded = self.y_embedding(y_flat[:, i])  # [N, y_embedding_dim]
                y_embedded_frames.append(frame_embedded)
            y_embedded = torch.cat(y_embedded_frames, dim=1)  # [N, y_embedding_dim * context_frames]
        else:
            raise ValueError(f"Unexpected y_flat dimension: {y_flat.dim()}")

        # Concatenate embedded features
        combined = torch.cat([x_embedded, y_embedded], dim=1)  # [N, final_x_dim + final_y_dim]

        # Forward through MLP network
        features = self.network(combined)  # [N, hidden_dims[-1]]

        # Forward through output layer
        output = self.output_layer(features)  # [N, 1]
        return output