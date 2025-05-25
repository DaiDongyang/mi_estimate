"""
Main script for training MINE model with checkpoint saving functionality, projection layer and context window support
Now with periodic validation during training for large datasets
"""
import torch
import yaml
import os
import time
import argparse
import numpy as np
from tqdm import tqdm  # For progress bar

# Import modules from same directory or Python path
from mine_estimator import MINE
from dataset import create_data_loaders

def save_checkpoint(mine_model, epoch, step, best_val_mi, checkpoint_dir, filename):
    """
    Save model checkpoint
    
    Args:
        mine_model: MINE model instance
        epoch: Current epoch
        step: Current step (total batches processed)
        best_val_mi: Best validation MI value
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': mine_model.mine_net.state_dict(),
        'optimizer_state_dict': mine_model.optimizer.state_dict(),
        'scheduler_state_dict': mine_model.scheduler.state_dict(),
        'best_val_mi': best_val_mi,
        'train_mi_history': mine_model.train_mi_history,
        'val_mi_history': mine_model.val_mi_history,
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(mine_model, checkpoint_path):
    """
    Load model checkpoint
    
    Args:
        mine_model: MINE model instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        epoch: Last saved epoch
        step: Last saved step
        best_val_mi: Best validation MI from checkpoint
    """
    # Handle PyTorch 2.6+ weights_only security feature
    try:
        # First try with weights_only=True (safer)
        checkpoint = torch.load(checkpoint_path, map_location=mine_model.device, weights_only=True)
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            print(f"Note: Loading checkpoint with weights_only=False due to numpy objects in checkpoint.")
            print("This is safe for checkpoints you created yourself.")
            # Fallback to weights_only=False for compatibility
            checkpoint = torch.load(checkpoint_path, map_location=mine_model.device, weights_only=False)
        else:
            raise e
    
    mine_model.mine_net.load_state_dict(checkpoint['model_state_dict'])
    mine_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mine_model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    mine_model.train_mi_history = checkpoint['train_mi_history']
    mine_model.val_mi_history = checkpoint['val_mi_history']
    mine_model.best_val_mi = checkpoint['best_val_mi']
    
    return checkpoint['epoch'], checkpoint.get('step', 0), checkpoint['best_val_mi']

def main(config_path):
    """Main training function, takes configuration file path as parameter"""
    # --- 1. Load configuration ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration file: {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse configuration file {config_path} - {e}")
        return
    except Exception as e:
        print(f"Unknown error occurred while loading or parsing configuration file: {e}")
        return

    model_config = config.get('model', {})
    train_config = config.get('training', {})
    log_config = config.get('logging', {})
    
    # Checkpoint configuration
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_dir = checkpoint_config.get('dir', './checkpoints')
    checkpoint_interval = checkpoint_config.get('save_interval', 10)
    resume_training = checkpoint_config.get('resume_from', None)
    
    # Validation configuration
    validation_config = config.get('validation', {})
    validate_every_n_steps = validation_config.get('validate_every_n_steps', None)
    early_stopping_patience = validation_config.get('early_stopping_patience', None)
    
    # If validate_every_n_steps is not set, validate once per epoch
    validate_per_epoch = validate_every_n_steps is None

    # --- 2. Create data loaders and get dimensions ---
    try:
        print("Creating data loaders...")
        data_info = create_data_loaders(config_path)
        train_loader = data_info['train_loader']
        valid_loader = data_info['valid_loader']
        test_loader = data_info['test_loader']
        x_dim = data_info['x_dim']  # Might be None for index type
        y_dim = data_info['y_dim']  # Might be None for index type
        print("Data loaders created successfully.")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"Error: Failed to create data loaders - {e}")
        print("Please check the data paths and parameters in the configuration file.")
        return

    # --- 3. Initialize MINE model ---
    # Get feature types and dimensions from configuration
    x_type = model_config.get('x_type', 'float')
    y_type = model_config.get('y_type', 'float')
    bias_correction_method = model_config.get('bias_correction_method', 'none')
    
    # Get context frame numbers from data config
    data_config = config.get('data', {})
    context_frame_numbers = data_config.get('context_frame_numbers', 1)
    
    # Vocabulary sizes and embedding dimensions (for index features)
    x_vocab_size = model_config.get('x_vocab_size')
    x_embedding_dim = model_config.get('x_embedding_dim')
    y_vocab_size = model_config.get('y_vocab_size')
    y_embedding_dim = model_config.get('y_embedding_dim')
    
    # Projection dimensions (new feature)
    x_proj_dim = model_config.get('x_proj_dim')
    y_proj_dim = model_config.get('y_proj_dim')
    
    # Feature dimensions
    config_x_dim = model_config.get('x_dim')
    config_y_dim = model_config.get('y_dim')
    
    # Use configured dimensions if provided, otherwise use detected ones
    # Note: If context windowing is enabled, we need to multiply configured dimensions
    if config_x_dim is not None and x_type == 'float' and context_frame_numbers > 1:
        final_x_dim = config_x_dim * context_frame_numbers
        print(f"Note: Expanding configured x_dim {config_x_dim} by context_frame_numbers {context_frame_numbers} = {final_x_dim}")
    else:
        final_x_dim = config_x_dim if config_x_dim is not None else x_dim
        
    if config_y_dim is not None and y_type == 'float' and context_frame_numbers > 1:
        final_y_dim = config_y_dim * context_frame_numbers
        print(f"Note: Expanding configured y_dim {config_y_dim} by context_frame_numbers {context_frame_numbers} = {final_y_dim}")
    else:
        final_y_dim = config_y_dim if config_y_dim is not None else y_dim

    print(f"Preparing to initialize MINE model (x_type: {x_type}, y_type: {y_type})...")
    print(f"Feature dimensions: x_dim={final_x_dim}, y_dim={final_y_dim}")
    print(f"Context frame numbers: {context_frame_numbers}")
    
    # Print projection info if available
    if x_proj_dim is not None and x_type == 'float':
        print(f"X projection: {final_x_dim} -> {x_proj_dim}")
    if y_proj_dim is not None and y_type == 'float':
        print(f"Y projection: {final_y_dim} -> {y_proj_dim}")
    
    # Check required parameters based on feature types
    if x_type == 'index' and (x_vocab_size is None or x_embedding_dim is None):
        print("Error: When x_type='index', you must specify model.x_vocab_size and model.x_embedding_dim in config.yaml")
        return
        
    if y_type == 'index' and (y_vocab_size is None or y_embedding_dim is None):
        print("Error: When y_type='index', you must specify model.y_vocab_size and model.y_embedding_dim in config.yaml")
        return
        
    if x_type == 'float' and final_x_dim is None:
        print("Error: Could not determine x_dim for float features.")
        return
        
    if y_type == 'float' and final_y_dim is None:
        print("Error: Could not determine y_dim for float features.")
        return

    try:
        mine_model = MINE(
            x_type=x_type,
            y_type=y_type,
            x_dim=final_x_dim,
            y_dim=final_y_dim,
            x_vocab_size=x_vocab_size,
            x_embedding_dim=x_embedding_dim,
            y_vocab_size=y_vocab_size,
            y_embedding_dim=y_embedding_dim,
            x_proj_dim=x_proj_dim,
            y_proj_dim=y_proj_dim,
            context_frame_numbers=context_frame_numbers,
            hidden_dims=model_config.get('hidden_dims', [512, 512, 256]),
            activation=model_config.get('activation', 'relu'),
            batch_norm=model_config.get('batch_norm', True),
            dropout=model_config.get('dropout', 0.1),
            lr=train_config.get('lr', 1e-4),
            device=train_config.get('device', 'cuda')
        )
        print("MINE model initialized successfully.")
    except Exception as e:
        print(f"Error: Failed to initialize MINE model - {e}")
        return

    # --- 4. Resume from checkpoint if specified ---
    start_epoch = 1
    global_step = 0
    best_epoch = 0  # Initialize to 0 instead of -1
    if resume_training:
        try:
            print(f"Resuming training from checkpoint: {resume_training}")
            loaded_epoch, loaded_step, loaded_best_val_mi = load_checkpoint(mine_model, resume_training)
            start_epoch = loaded_epoch
            global_step = loaded_step
            print(f"Resumed from epoch {loaded_epoch}, step {loaded_step} with best validation MI: {loaded_best_val_mi:.6f}")
            best_epoch = loaded_epoch if loaded_best_val_mi == mine_model.best_val_mi else best_epoch
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch instead.")
    
    # --- 5. Training loop ---
    epochs = train_config.get('epochs', 100)
    log_interval = log_config.get('log_interval', 10)

    print("\n" + "="*30)
    print(f"Starting MINE model training (x_type: {x_type}, y_type: {y_type})")
    print(f"Total Epochs: {epochs}")
    print(f"Using device: {mine_model.device}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Checkpoint save interval: {checkpoint_interval} epochs")
    if validate_every_n_steps:
        print(f"Validation interval: every {validate_every_n_steps} steps")
    else:
        print("Validation interval: once per epoch")
    if early_stopping_patience:
        print(f"Early stopping patience: {early_stopping_patience} validations")
    print("="*30 + "\n")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    
    # Track best validation performance
    best_checkpoint_path = None
    patience_counter = 0
    
    # For tracking within-epoch training statistics
    epoch_mi_values = []
    epoch_loss_values = []
    steps_in_current_epoch = 0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        
        # Reset epoch statistics
        epoch_mi_values = []
        epoch_loss_values = []
        steps_in_current_epoch = 0
        
        # Set model to training mode
        mine_model.mine_net.train()
        
        # Create progress bar for the epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, ncols=100)
        
        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            try:
                # Train on batch
                mi_value, loss_value = mine_model.train_batch(batch_x, batch_y, bias_correction_method=bias_correction_method)
                
                if not np.isnan(mi_value) and not np.isinf(mi_value):
                    epoch_mi_values.append(mi_value)
                    epoch_loss_values.append(loss_value)
                else:
                    print(f"\nWarning: Batch {batch_idx} produced invalid MI/Loss values, skipped.")
                
                # Update global step counter
                global_step += 1
                steps_in_current_epoch += 1
                
                # Update progress bar with current MI
                if epoch_mi_values:
                    current_avg_mi = np.mean(epoch_mi_values[-100:])  # Average of last 100 batches
                    pbar.set_postfix({'MI': f'{current_avg_mi:.4f}'})
                
                # Periodic validation based on steps
                if validate_every_n_steps and global_step % validate_every_n_steps == 0:
                    print(f"\n[Step {global_step}] Running validation...")
                    avg_val_mi = mine_model.validate(valid_loader)
                    
                    # Check if new best validation MI
                    if avg_val_mi > mine_model.best_val_mi:
                        best_epoch = epoch
                        patience_counter = 0
                        # Save best model checkpoint
                        best_checkpoint_path = save_checkpoint(
                            mine_model, epoch, global_step, mine_model.best_val_mi, 
                            checkpoint_dir, f"best_model.pth"
                        )
                        print(f"  * New best model saved! Val MI: {avg_val_mi:.6f}")
                    else:
                        patience_counter += 1
                        if early_stopping_patience and patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered after {patience_counter} validations without improvement.")
                            pbar.close()
                            break
                    
                    # Save periodic checkpoint based on steps
                    if global_step % (validate_every_n_steps * checkpoint_interval) == 0:
                        save_checkpoint(
                            mine_model, epoch, global_step, mine_model.best_val_mi,
                            checkpoint_dir, f"checkpoint_step_{global_step}.pth"
                        )
                    
                    # Print current statistics
                    avg_train_mi = np.mean(epoch_mi_values) if epoch_mi_values else 0.0
                    avg_train_loss = np.mean(epoch_loss_values) if epoch_loss_values else 0.0
                    print(f"  Step {global_step} | Train MI: {avg_train_mi:.6f} | Train Loss: {avg_train_loss:.6f} | "
                          f"Val MI: {avg_val_mi:.6f} | Best Val MI: {mine_model.best_val_mi:.6f}")
                    
                    # Return to training mode
                    mine_model.mine_net.train()
                    
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        pbar.close()
        
        # Check for early stopping trigger
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print("Stopping training due to early stopping.")
            break
        
        # Update learning rate scheduler (once per epoch)
        mine_model.scheduler.step()
        
        # Compute epoch statistics
        avg_train_mi = np.mean(epoch_mi_values) if epoch_mi_values else 0.0
        avg_train_loss = np.mean(epoch_loss_values) if epoch_loss_values else 0.0
        mine_model.train_mi_history.append(avg_train_mi)
        
        # End-of-epoch validation (if not doing step-based validation)
        if validate_per_epoch:
            print(f"\n[Epoch {epoch}] Running validation...")
            avg_val_mi = mine_model.validate(valid_loader)
            
            # Check if new best validation MI
            if avg_val_mi > mine_model.best_val_mi:
                best_epoch = epoch
                patience_counter = 0
                # Save best model checkpoint
                best_checkpoint_path = save_checkpoint(
                    mine_model, epoch, global_step, mine_model.best_val_mi, 
                    checkpoint_dir, f"best_model.pth"
                )
                print(f"  * New best model saved at epoch {epoch}!")
            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement.")
                    break
        else:
            # Get the latest validation MI for logging
            avg_val_mi = mine_model.val_mi_history[-1] if mine_model.val_mi_history else 0.0
        
        # Save periodic checkpoint based on epochs
        if epoch % checkpoint_interval == 0:
            save_checkpoint(
                mine_model, epoch, global_step, mine_model.best_val_mi,
                checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs} | Time: {epoch_time:.2f}s | Steps: {steps_in_current_epoch} | "
              f"Train MI: {avg_train_mi:.6f} | Train Loss: {avg_train_loss:.6f} | "
              f"Val MI: {avg_val_mi:.6f} | Best Val MI: {mine_model.best_val_mi:.6f} (Epoch {best_epoch if best_epoch > 0 else 'N/A'})")
        
        # Periodically print more detailed network statistics
        if epoch % log_interval == 0 or epoch == epochs:
            print("-" * 20 + f" Epoch {epoch} Stats " + "-" * 20)
            stats_info = mine_model.get_network_stats()
            print(stats_info if isinstance(stats_info, str) else "\n".join([f"  {k}: {v}" for k, v in stats_info.items()]))
            print("-" * (40 + len(f" Epoch {epoch} Stats ")))

    # Save final checkpoint
    save_checkpoint(
        mine_model, epochs, global_step, mine_model.best_val_mi,
        checkpoint_dir, f"final_model_epoch_{epochs}.pth"
    )

    total_training_time = time.time() - start_time
    print("\n" + "="*30)
    print("Training complete")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Best validation MI: {mine_model.best_val_mi:.6f} (achieved at Epoch {best_epoch})")
    print(f"Best model checkpoint saved at: {best_checkpoint_path}")
    print("="*30 + "\n")

    # --- 6. Testing ---
    # Load the best model for evaluation if available
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        print(f"Loading best model from {best_checkpoint_path} for testing...")
        try:
            _, _, _ = load_checkpoint(mine_model, best_checkpoint_path)
            print("Best model loaded successfully.")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using final model for testing instead.")

    print("Evaluating on test set...")
    try:
        test_mi = mine_model.evaluate(test_loader)
        print(f"Test set MI: {test_mi:.6f}")
    except Exception as e:
        print(f"Error: Error occurred while evaluating on test set - {e}")

    print("="*30)


if __name__ == "__main__":
    # --- Setup ArgumentParser ---
    parser = argparse.ArgumentParser(description='Train MINE model using specified configuration file.')

    # Add --config parameter
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,  # Make config file parameter required
        help='Path to YAML configuration file (required)'
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Get configuration file path
    config_file_path = args.config

    # --- Check if configuration file exists ---
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found '{config_file_path}'")
        print("Please provide a valid configuration file path.")
        # Print example configuration to help user create one
        print("\nYou can create a config.yaml file similar to the following:")
        example_config = """
# config.yaml - MINE Training Configuration (Example with Context Window and Projection Layers)
data:
  features_dir: './features'  # <<<--- Change to your feature .npy directory
  train_csv: './train_list.csv' # <<<--- Change to your train list CSV
  valid_csv: './valid_list.csv' # <<<--- Change to your validation list CSV
  test_csv: './test_list.csv'   # <<<--- Change to your test list CSV
  segment_length: 100       # Cropped sequence length
  normalization: 'standard' # Float feature normalization: none, minmax, standard, robust
  context_frame_numbers: 3  # NEW: Number of consecutive frames for context (1 = disable)
  x_repeat: 1               # Repeat factor for x features (upsampling)
  y_repeat: 1               # Repeat factor for y features (upsampling)
  max_length_diff: 5        # Max difference between x and y lengths after repeat

model:
  # --- Feature types ---
  x_type: 'float'           # 'float' or 'index'
  y_type: 'float'           # 'float' or 'index'
  
  # --- Feature dimensions (for float type) ---
  x_dim: 128                # Base dimension of x features when x_type='float'
  y_dim: 256                # Base dimension of y features when y_type='float'
  
  # --- NEW: Projection dimensions (for dimensionality reduction) ---
  x_proj_dim: 64            # Project effective x_dim to this dimension (only for float features)
  y_proj_dim: 64            # Project effective y_dim to this dimension (only for float features)
  
  # --- Required for x_type='index' ---
  # x_vocab_size: 5000        # Size of vocabulary for x
  # x_embedding_dim: 256      # Embedding dimension for x
  
  # --- Required for y_type='index' ---
  # y_vocab_size: 5000        # Size of vocabulary for y
  # y_embedding_dim: 256      # Embedding dimension for y

  # --- Common MLP parameters ---
  hidden_dims: [128, 64]    # Network hidden layer dimensions (after projection/embedding)
  activation: 'relu'        # Activation function: relu, leaky_relu, elu
  batch_norm: True          # Whether to use BatchNorm
  dropout: 0.1              # Dropout rate (0 means none)
  bias_correction_method: none

training:
  epochs: 50                # Number of training epochs
  batch_size: 128           # Batch size
  lr: 0.0002                # Learning rate
  device: 'cuda'            # Use 'cuda' or 'cpu'
  num_workers: 4            # Number of data loading processes (adjust for your machine)
  seed: 42                  # Random seed

# NEW: Validation configuration for large datasets
validation:
  validate_every_n_steps: 1000  # Validate every N training steps (null = once per epoch)
  early_stopping_patience: 10   # Stop if no improvement for N validations (null = no early stopping)

checkpoint:
  dir: './checkpoints'      # Directory to save checkpoints
  save_interval: 10         # Save checkpoint every N epochs
  resume_from: null         # Path to checkpoint to resume from (null to start from scratch)

logging:
  log_interval: 10          # Print detailed statistics every this many epochs
"""
        print(example_config)
        exit(1)  # Exit program

    # --- If file exists, call main function to start training ---
    main(config_file_path)