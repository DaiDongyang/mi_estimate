# train.py
"""
Main script for training MINE model
"""
import torch
import yaml
import os
import time
import argparse
from tqdm import tqdm  # For progress bar

# Import modules from same directory or Python path
from mine_estimator import MINE
from dataset import create_data_loaders

def main(config_path):
    """Main training function, takes configuration file path as parameter"""
    # --- 1. Load configuration ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration file: {config_path}")
    except FileNotFoundError:
        # argparse should handle file not found case before calling main, but keep this as a fallback
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
    
    # Vocabulary sizes and embedding dimensions (for index features)
    x_vocab_size = model_config.get('x_vocab_size')
    x_embedding_dim = model_config.get('x_embedding_dim')
    y_vocab_size = model_config.get('y_vocab_size')
    y_embedding_dim = model_config.get('y_embedding_dim')
    
    # Feature dimensions - x_dim and y_dim might already be provided by the data_info
    # but model_config can override them if specified
    config_x_dim = model_config.get('x_dim')
    config_y_dim = model_config.get('y_dim')
    
    # Use configured dimensions if provided, otherwise use detected ones
    final_x_dim = config_x_dim if config_x_dim is not None else x_dim
    final_y_dim = config_y_dim if config_y_dim is not None else y_dim

    print(f"Preparing to initialize MINE model (x_type: {x_type}, y_type: {y_type})...")
    print(f"Feature dimensions: x_dim={final_x_dim}, y_dim={final_y_dim}")
    
    # Check required parameters based on feature types
    if x_type == 'index' and (x_vocab_size is None or x_embedding_dim is None):
        print("Error: When x_type='index', you must specify model.x_vocab_size and model.x_embedding_dim in config.yaml")
        return
        
    if y_type == 'index' and (y_vocab_size is None or y_embedding_dim is None):
        print("Error: When y_type='index', you must specify model.y_vocab_size and model.y_embedding_dim in config.yaml")
        return
        
    if x_type == 'float' and final_x_dim is None:
        print("Error: Could not determine x_dim for float features.")
        print("Please ensure data loading is working correctly and CSV files point to valid float features,")
        print("or specify x_dim explicitly in the model section of config.yaml.")
        return
        
    if y_type == 'float' and final_y_dim is None:
        print("Error: Could not determine y_dim for float features.")
        print("Please ensure data loading is working correctly and CSV files point to valid float features,")
        print("or specify y_dim explicitly in the model section of config.yaml.")
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

    # --- 4. Training loop ---
    epochs = train_config.get('epochs', 100)
    log_interval = log_config.get('log_interval', 10)  # Adjust print interval

    print("\n" + "="*30)
    print(f"Starting MINE model training (x_type: {x_type}, y_type: {y_type})")
    print(f"Total Epochs: {epochs}")
    print(f"Using device: {mine_model.device}")
    print("="*30 + "\n")

    start_time = time.time()
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # --- Training ---
        try:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False, ncols=100)
            avg_train_mi, avg_train_loss = mine_model.train_epoch(pbar)
            pbar.close()
            if len(pbar) == 0:  # Check if train_loader is empty
                print(f"Warning: Train loader for Epoch {epoch} is empty, cannot train.")
                continue  # Skip remainder of this epoch
        except Exception as e:
            print(f"\nError: Error occurred during training Epoch {epoch} - {e}")
            # Can choose to continue to next epoch or stop training
            print("Skipping this Epoch's training.")
            continue  # Go to next epoch

        # --- Validation ---
        try:
            avg_val_mi = mine_model.validate(valid_loader)
            # Check if new best validation MI
            if mine_model.best_val_mi == avg_val_mi and avg_val_mi != float('-inf'):
                best_epoch = epoch
        except Exception as e:
            print(f"\nError: Error occurred during validation Epoch {epoch} - {e}")
            print("Will skip validation and logging for this Epoch.")
            continue  # Go to next epoch

        epoch_time = time.time() - epoch_start_time

        # --- Print logs ---
        print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s | "
              f"Train MI: {avg_train_mi:.6f} | Train Loss: {avg_train_loss:.6f} | "
              f"Val MI: {avg_val_mi:.6f} | Best Val MI: {mine_model.best_val_mi:.6f} (Epoch {best_epoch if best_epoch > 0 else 'N/A'})")

        # Periodically print more detailed network statistics
        if epoch % log_interval == 0 or epoch == epochs:
            print("-" * 20 + f" Epoch {epoch} Stats " + "-" * 20)
            stats_info = mine_model.get_network_stats()
            print(stats_info if isinstance(stats_info, str) else "\n".join([f"  {k}: {v}" for k, v in stats_info.items()]))
            print("-" * (40 + len(f" Epoch {epoch} Stats ")))

    total_training_time = time.time() - start_time
    print("\n" + "="*30)
    print("Training complete")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Best validation MI: {mine_model.best_val_mi:.6f} (achieved at Epoch {best_epoch})")
    print("="*30 + "\n")

    # --- 5. Testing ---
    # Typically use best validation model for testing, if saved would need to load
    # Here we just evaluate the final model
    print("Evaluating final model on test set...")
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
# config.yaml - MINE Training Configuration (Example)
data:
  features_dir: './features'  # <<<--- Change to your feature .npy directory
  train_csv: './train_list.csv' # <<<--- Change to your train list CSV
  valid_csv: './valid_list.csv' # <<<--- Change to your validation list CSV
  test_csv: './test_list.csv'   # <<<--- Change to your test list CSV
  segment_length: 100       # Cropped sequence length
  normalization: 'standard' # Float feature normalization: none, minmax, standard, robust
  x_repeat: 1               # Repeat factor for x features (upsampling)
  y_repeat: 1               # Repeat factor for y features (upsampling)
  max_length_diff: 5        # Max difference between x and y lengths after repeat

model:
  # --- Feature types ---
  x_type: 'float'           # 'float' or 'index'
  y_type: 'float'           # 'float' or 'index'
  
  # --- Feature dimensions (for float type) ---
  x_dim: 20                 # Dimension of x features when x_type='float'
  y_dim: 40                 # Dimension of y features when y_type='float'
  
  # --- Required for x_type='index' ---
  # x_vocab_size: 5000        # Size of vocabulary for x
  # x_embedding_dim: 256      # Embedding dimension for x
  
  # --- Required for y_type='index' ---
  # y_vocab_size: 5000        # Size of vocabulary for y
  # y_embedding_dim: 256      # Embedding dimension for y

  # --- Common MLP parameters ---
  hidden_dims: [256, 256]   # Network hidden layer dimensions
  activation: 'relu'        # Activation function: relu, leaky_relu, elu
  batch_norm: True          # Whether to use BatchNorm
  dropout: 0.1              # Dropout rate (0 means none)

training:
  epochs: 50                # Number of training epochs
  batch_size: 128           # Batch size
  lr: 0.0002                # Learning rate
  device: 'cuda'            # Use 'cuda' or 'cpu'
  num_workers: 4            # Number of data loading processes (adjust for your machine)
  seed: 42                  # Random seed

logging:
  log_interval: 10          # Print detailed statistics every this many epochs
"""
        print(example_config)
        exit(1)  # Exit program

    # --- If file exists, call main function to start training ---
    main(config_file_path)