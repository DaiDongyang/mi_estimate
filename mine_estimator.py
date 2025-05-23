# mine_estimator.py
"""
MINE mutual information estimator with projection layer support
"""
import torch
import torch.optim as optim
import numpy as np
# Import network classes
from mine_network import (
    MINENetworkFloatFloat, 
    MINENetworkIndexFloat, 
    MINENetworkFloatIndex,
    MINENetworkIndexIndex
)

class MINE:
    """Mutual Information Neural Estimator for I(X;Y) with projection layer support"""

    def __init__(self, x_type='float', y_type='float',
                 x_dim=None, y_dim=None,
                 x_vocab_size=None, x_embedding_dim=None,
                 y_vocab_size=None, y_embedding_dim=None,
                 x_proj_dim=None, y_proj_dim=None,
                 context_frame_numbers=1,
                 hidden_dims=[128, 64], activation="relu", 
                 batch_norm=True, dropout=0.1,
                 lr=1e-4, device="cuda"):
        """
        Initialize MINE estimator with optional projection layers and context window support
        
        Args:
            x_type: Type of X feature ('float' or 'index')
            y_type: Type of Y feature ('float' or 'index')
            x_dim: X feature dimension (required when x_type='float', may include context expansion)
            y_dim: Y feature dimension (required when y_type='float', may include context expansion)
            x_vocab_size: X vocabulary size (required when x_type='index')
            x_embedding_dim: X embedding dimension (required when x_type='index', single frame)
            y_vocab_size: Y vocabulary size (required when y_type='index')
            y_embedding_dim: Y embedding dimension (required when y_type='index', single frame)
            x_proj_dim: X projection dimension (only used when x_type='float')
            y_proj_dim: Y projection dimension (only used when y_type='float')
            context_frame_numbers: Number of context frames (affects index embedding dimensions)
            hidden_dims: List of hidden layer dimensions for MINE network
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            lr: Learning rate
            device: Device ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"MINE Estimator using device: {self.device}")
        self.x_type = x_type
        self.y_type = y_type
        self.context_frame_numbers = context_frame_numbers

        # --- Create appropriate MINE network based on feature types ---
        if x_type == 'float' and y_type == 'float':
            if x_dim is None or y_dim is None:
                raise ValueError("x_dim and y_dim must be provided for x_type='float' and y_type='float'")
            
            # Print projection info
            proj_info = []
            if x_proj_dim is not None:
                proj_info.append(f"x_proj_dim={x_proj_dim}")
            if y_proj_dim is not None:
                proj_info.append(f"y_proj_dim={y_proj_dim}")
            proj_str = f" with projections ({', '.join(proj_info)})" if proj_info else ""
            
            print(f"Initializing MINENetworkFloatFloat (x_dim={x_dim}, y_dim={y_dim}){proj_str}")
            self.mine_net = MINENetworkFloatFloat(
                x_dim, y_dim, hidden_dims, activation, batch_norm, dropout,
                x_proj_dim=x_proj_dim, y_proj_dim=y_proj_dim
            ).to(self.device)
            
        elif x_type == 'index' and y_type == 'float':
            if x_vocab_size is None or x_embedding_dim is None or y_dim is None:
                raise ValueError("x_vocab_size, x_embedding_dim, and y_dim must be provided for x_type='index' and y_type='float'")
            
            proj_str = f" with y_proj_dim={y_proj_dim}" if y_proj_dim is not None else ""
            context_str = f" with context_frames={context_frame_numbers}" if context_frame_numbers > 1 else ""
            print(f"Initializing MINENetworkIndexFloat (x_vocab={x_vocab_size}, x_embed_dim={x_embedding_dim}, y_dim={y_dim}){proj_str}{context_str}")
            self.mine_net = MINENetworkIndexFloat(
                x_vocab_size, x_embedding_dim, y_dim, hidden_dims, activation, batch_norm, dropout,
                y_proj_dim=y_proj_dim, context_frame_numbers=context_frame_numbers
            ).to(self.device)
            
        elif x_type == 'float' and y_type == 'index':
            if x_dim is None or y_vocab_size is None or y_embedding_dim is None:
                raise ValueError("x_dim, y_vocab_size, and y_embedding_dim must be provided for x_type='float' and y_type='index'")
            
            proj_str = f" with x_proj_dim={x_proj_dim}" if x_proj_dim is not None else ""
            context_str = f" with context_frames={context_frame_numbers}" if context_frame_numbers > 1 else ""
            print(f"Initializing MINENetworkFloatIndex (x_dim={x_dim}, y_vocab={y_vocab_size}, y_embed_dim={y_embedding_dim}){proj_str}{context_str}")
            self.mine_net = MINENetworkFloatIndex(
                x_dim, y_vocab_size, y_embedding_dim, hidden_dims, activation, batch_norm, dropout,
                x_proj_dim=x_proj_dim, context_frame_numbers=context_frame_numbers
            ).to(self.device)
            
        elif x_type == 'index' and y_type == 'index':
            if x_vocab_size is None or x_embedding_dim is None or y_vocab_size is None or y_embedding_dim is None:
                raise ValueError("x_vocab_size, x_embedding_dim, y_vocab_size, and y_embedding_dim must be provided for x_type='index' and y_type='index'")
            
            context_str = f" with context_frames={context_frame_numbers}" if context_frame_numbers > 1 else ""
            print(f"Initializing MINENetworkIndexIndex (x_vocab={x_vocab_size}, x_embed_dim={x_embedding_dim}, y_vocab={y_vocab_size}, y_embed_dim={y_embedding_dim}){context_str}")
            print("  Note: Index features use embeddings for dimensionality control, no additional projection needed")
            self.mine_net = MINENetworkIndexIndex(
                x_vocab_size, x_embedding_dim, y_vocab_size, y_embedding_dim, hidden_dims, activation, batch_norm, dropout,
                context_frame_numbers=context_frame_numbers
            ).to(self.device)
            
        else:
            raise ValueError(f"Unsupported feature types: x_type={x_type}, y_type={y_type}. Choose 'float' or 'index'.")

        # Print parameter count
        total_params = sum(p.numel() for p in self.mine_net.parameters())
        trainable_params = sum(p.numel() for p in self.mine_net.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.mine_net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )

        # Learning rate scheduler (optional, but recommended)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr / 10
        )

        # Training history
        self.train_mi_history = []
        self.val_mi_history = []
        self.best_val_mi = float('-inf')

        # Diagnostic information
        self.recent_t_values = []
        self.gradient_norms = []
        self.ema_expectation = None
        self.ema_alpha = 0.01

    def compute_mutual_info(self, x_joint, y_joint):
        """
        Compute mutual information estimate (Donsker-Varadhan lower bound)
        
        Args:
            x_joint: Joint distribution X samples, shape [B, S, Dx] (float) or [B, S] (long) or [B, Dx] (context-expanded float) or [B, context_frames] (context-expanded index)
            y_joint: Joint distribution Y samples, shape [B, S, Dy] (float) or [B, S] (long) or [B, Dy] (context-expanded float) or [B, context_frames] (context-expanded index)
            
        Returns:
            Mutual information estimate (scalar Tensor) and intermediate values for bias correction
        """
        # --- 1. Generate y_marginal ---
        # Create marginal samples y' by shuffling y_joint along the batch dimension
        # The shuffling strategy needs to handle different tensor shapes from context windowing
        
        if y_joint.dim() == 3:  # [B, S, Dy] - sequential float features
            B, S, Dy = y_joint.shape
            N = B * S
            y_joint_flat = y_joint.reshape(N, Dy)
            shuffle_idx = torch.randperm(N).to(self.device)
            y_marginal_flat = y_joint_flat[shuffle_idx]
            # Reshape y_marginal back to original shape
            y_marginal = y_marginal_flat.reshape(B, S, Dy)
        elif y_joint.dim() == 2:
            if self.y_type == 'float':
                # [B, Dy] - non-sequential float features (possibly context-expanded)
                B, Dy = y_joint.shape
                shuffle_idx = torch.randperm(B).to(self.device)
                y_marginal = y_joint[shuffle_idx]
            else:
                # [B, S] - sequential index features or [B, context_frames] - context-expanded index features
                B, S = y_joint.shape
                shuffle_idx = torch.randperm(B).to(self.device)
                y_marginal = y_joint[shuffle_idx]
        elif y_joint.dim() == 1:  # [B] - non-sequential index features
            B = y_joint.shape[0]
            shuffle_idx = torch.randperm(B).to(self.device)
            y_marginal = y_joint[shuffle_idx]
        else:
            raise ValueError(f"Unsupported y_joint dimension: {y_joint.dim()}")

        # Verify x and y have compatible shapes
        if x_joint.shape[0] != y_joint.shape[0]:
            raise ValueError(f"Batch size mismatch: x_joint batch size {x_joint.shape[0]} != y_joint batch size {y_joint.shape[0]}")

        # --- 2. Compute network outputs ---
        # The network forward method handles reshaping internally, returns [N, 1]
        t_joint = self.mine_net(x_joint, y_joint)
        t_marginal = self.mine_net(x_joint, y_marginal)  # Use original x with shuffled y

        # --- Diagnostic information ---
        self.recent_t_values.append({
            'joint_mean': t_joint.mean().item(),
            'joint_std': t_joint.std().item(),
            'marginal_mean': t_marginal.mean().item(),
            'marginal_std': t_marginal.std().item()
        })
        if len(self.recent_t_values) > 50:
            self.recent_t_values.pop(0)
        # --- End diagnostic information ---

        # --- 3. Compute MI lower bound ---
        # E_P[T] - log(E_Q[e^T]), where P is joint distribution, Q is product of marginals
        e_joint = torch.mean(t_joint)

        # Use logsumexp trick for numerical stability in computing log(E_Q[e^T])
        max_t = torch.max(t_marginal).detach()
        stabilized_denom = torch.mean(torch.exp(t_marginal - max_t))
        
        # Calculate full expectation E_Q[e^T] (for EMA updates)
        # Using log-sum-exp trick to avoid overflow
        full_expectation = stabilized_denom * torch.exp(max_t.to(torch.float32))
        
        # Update EMA of the expectation
        if self.ema_expectation is None:
            self.ema_expectation = full_expectation.detach()
        else:
            self.ema_expectation = (1 - self.ema_alpha) * self.ema_expectation + self.ema_alpha * full_expectation.detach()
        
        # Standard MI computation (without bias correction)
        log_e_marginal = max_t + torch.log(stabilized_denom + 1e-8)
        mi_estimate = e_joint - log_e_marginal
        
        # Return MI estimate and intermediate values needed for bias correction
        return mi_estimate, t_joint, t_marginal, full_expectation

    def train_batch(self, x_joint_orig, y_joint_orig, bias_correction_method='none'):
        """
        Train MINE network on a single batch
        
        Args:
            x_joint_orig: Original joint X samples
            y_joint_orig: Original joint Y samples
            bias_correction_method: Method to use for bias correction ('default', or 'none')
                    
        Returns:
            MI estimate for this batch (float), loss value (float)
        """
        self.mine_net.train()

        # Move data to device and ensure correct types
        if self.x_type == 'index':
            x_joint = x_joint_orig.long().to(self.device)  # Ensure long tensor
        else:
            x_joint = x_joint_orig.float().to(self.device)  # Ensure float tensor
            
        if self.y_type == 'index':
            y_joint = y_joint_orig.long().to(self.device)  # Ensure long tensor
        else:
            y_joint = y_joint_orig.float().to(self.device)  # Ensure float tensor

        # Compute MI and get intermediate values for bias correction
        mi_estimate, t_joint, t_marginal, full_expectation = self.compute_mutual_info(x_joint, y_joint)

        # Compute loss with or without bias correction
        if bias_correction_method == 'none':
            # No bias correction (original implementation)
            loss = -mi_estimate
        else:
            # reference from: https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
            e_joint = torch.mean(t_joint)
            loss = -(e_joint - (1.0 / self.ema_expectation).detach() * full_expectation)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent gradient explosion)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.mine_net.parameters(), max_norm=1.0
        )
        self.gradient_norms.append(grad_norm.item())
        if len(self.gradient_norms) > 100:  # Only keep recent gradient norms
            self.gradient_norms.pop(0)

        self.optimizer.step()

        # Check for NaN or Inf (helpful for debugging)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss value is NaN or Inf!")

        return mi_estimate.item(), loss.item()

    def train_epoch(self, train_loader, bias_correction_method='none'):
        """Train for one epoch"""
        self.mine_net.train()
        epoch_mi = []
        total_loss = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            mi_value, loss_value = self.train_batch(batch_x, batch_y, bias_correction_method)

            if not np.isnan(mi_value) and not np.isinf(mi_value):
                epoch_mi.append(mi_value)
                total_loss += loss_value
            else:
                print(f"Warning: Batch {batch_idx} produced invalid MI/Loss values, skipped.")

            if abs(mi_value) > 100 and not np.isnan(mi_value):
                print(f"Warning: Batch {batch_idx} has abnormal MI value: {mi_value:.4f}")

        # Update learning rate
        self.scheduler.step()

        avg_mi = np.mean(epoch_mi) if epoch_mi else 0.0
        avg_loss = total_loss / len(epoch_mi) if epoch_mi else 0.0
        self.train_mi_history.append(avg_mi)

        return avg_mi, avg_loss

    def evaluate(self, data_loader):
        """Evaluate model on validation or test set"""
        self.mine_net.eval()
        eval_mi = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                # Move data to device and ensure correct types
                if self.x_type == 'index':
                    x_joint = batch_x.long().to(self.device)
                else:
                    x_joint = batch_x.float().to(self.device)
                    
                if self.y_type == 'index':
                    y_joint = batch_y.long().to(self.device)
                else:
                    y_joint = batch_y.float().to(self.device)

                # Compute MI (without bias correction for evaluation)
                mi_estimate, _, _, _ = self.compute_mutual_info(x_joint, y_joint)

                if not np.isnan(mi_estimate.item()) and not np.isinf(mi_estimate.item()):
                    eval_mi.append(mi_estimate.item())

        avg_mi = np.mean(eval_mi) if eval_mi else 0.0
        return avg_mi

    def validate(self, val_loader):
        """Perform validation step and update best MI"""
        avg_val_mi = self.evaluate(val_loader)
        self.val_mi_history.append(avg_val_mi)

        if avg_val_mi > self.best_val_mi:
            self.best_val_mi = avg_val_mi
            print(f"  * New best validation MI: {self.best_val_mi:.4f}")
            # Save model
            # torch.save(self.mine_net.state_dict(), 'best_mine_model.pth')

        return avg_val_mi

    def get_network_stats(self):
        """Return diagnostic information about network state"""
        if not self.recent_t_values:
            return "No statistics recorded yet."

        n = len(self.recent_t_values)
        joint_means = [v['joint_mean'] for v in self.recent_t_values]
        marginal_means = [v['marginal_mean'] for v in self.recent_t_values]

        stats = {
            'Recent Batches': n,
            'Avg Joint T(x,y)': np.mean(joint_means),
            'Std Joint T(x,y)': np.std(joint_means),
            'Avg Marginal T(x,y\')': np.mean(marginal_means),
            'Std Marginal T(x,y\')': np.std(marginal_means),
            'Avg Gradient Norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'Current LR': self.optimizer.param_groups[0]['lr']
        }
        return "\n".join([f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in stats.items()])