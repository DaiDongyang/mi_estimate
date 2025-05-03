# dataset.py
"""
Dataset module - Load preprocessed feature files, align, crop and normalize
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import yaml  # For loading config

class MIDataset(Dataset):
    """Mutual Information dataset, loads preprocessed feature files"""

    def __init__(self, csv_path, features_dir, segment_length=100,
                 normalization='standard', seed=42, 
                 x_type='float', y_type='float',
                 x_repeat=1, y_repeat=1,
                 max_length_diff=5,
                 x_dim=None, y_dim=None):  # Added configurable dimensions
        """
        Initialize dataset
        
        Args:
            csv_path: CSV file path (with id, x_feature, y_feature columns)
            features_dir: Feature files directory
            segment_length: Length of randomly cropped segments
            normalization: Feature normalization method ('none', 'minmax', 'standard', 'robust')
            seed: Random seed
            x_type: Type of x features ('float' or 'index')
            y_type: Type of y features ('float' or 'index') 
            x_repeat: Repeat factor for x features
            y_repeat: Repeat factor for y features
            max_length_diff: Maximum allowed difference between x and y lengths
            x_dim: Feature dimension for x (when x_type='float')
            y_dim: Feature dimension for y (when y_type='float')
        """
        self.features_dir = features_dir
        self.segment_length = segment_length
        self.normalization = normalization
        self.x_type = x_type
        self.y_type = y_type
        self.x_repeat = x_repeat
        self.y_repeat = y_repeat
        self.max_length_diff = max_length_diff
        
        # Set appropriate dtypes based on feature types
        self.x_dtype = torch.long if x_type == 'index' else torch.float32
        self.y_dtype = torch.long if y_type == 'index' else torch.float32

        random.seed(seed)
        np.random.seed(seed)

        self.data = pd.read_csv(csv_path)
        # Check for old column names and rename if needed
        if 'source_feature' in self.data.columns and 'target_feature' in self.data.columns:
            self.data = self.data.rename(columns={'source_feature': 'x_feature', 'target_feature': 'y_feature'})
            print("Renamed columns 'source_feature' and 'target_feature' to 'x_feature' and 'y_feature'")
            
        required_cols = ['id', 'x_feature', 'y_feature']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns {required_cols}")

        # Initialize dimensions (may be overridden by detected values if None)
        self._x_dim = x_dim if x_type == 'float' else None
        self._y_dim = y_dim if y_type == 'float' else None
        self._time_steps = None
        
        # Track filtered samples
        self.filtered_samples = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get data sample, returns aligned and cropped tensors.
        The dtype of x_tensor is controlled by self.x_dtype.
        The dtype of y_tensor is controlled by self.y_dtype.
        """
        overall_start_time = time.time()
        row = self.data.iloc[idx]

        x_path = row['x_feature']
        if not os.path.isabs(x_path):
            x_path = os.path.join(self.features_dir, x_path)

        y_path = row['y_feature']
        if not os.path.isabs(y_path):
            y_path = os.path.join(self.features_dir, y_path)

        io_start_time = time.time()
        try:
            # Load x features based on type
            if self.x_type == 'index':
                x_feature = np.load(x_path).astype(np.int64)
            else:
                x_feature = np.load(x_path).astype(np.float32)
                
            # Load y features based on type
            if self.y_type == 'index':
                y_feature = np.load(y_path).astype(np.int64)
            else:
                y_feature = np.load(y_path).astype(np.float32)

        except FileNotFoundError:
            print(f"Error: File not found - x: {x_path} or y: {y_path}")
            # Return placeholder data
            dummy_dim_x = self._x_dim if self._x_dim else 10
            dummy_dim_y = self._y_dim if self._y_dim else 10
            
            # Create appropriate shaped placeholders based on feature types
            if self.x_type == 'index':
                x_feature = np.random.randint(0, 100, size=(self.segment_length,)).astype(np.int64)
            else:
                x_feature = np.random.randn(self.segment_length, dummy_dim_x).astype(np.float32)
                
            if self.y_type == 'index':
                y_feature = np.random.randint(0, 100, size=(self.segment_length,)).astype(np.int64)
            else:
                y_feature = np.random.randn(self.segment_length, dummy_dim_y).astype(np.float32)
                
        except Exception as e:
            print(f"Error loading feature files (idx: {idx}): {e}")
            print(f"  X: {x_path}")
            print(f"  Y: {y_path}")
            
            # Return placeholder data
            dummy_dim_x = self._x_dim if self._x_dim else 10
            dummy_dim_y = self._y_dim if self._y_dim else 10
            
            # Create appropriate shaped placeholders based on feature types
            if self.x_type == 'index':
                x_feature = np.random.randint(0, 100, size=(self.segment_length,)).astype(np.int64)
            else:
                x_feature = np.random.randn(self.segment_length, dummy_dim_x).astype(np.float32)
                
            if self.y_type == 'index':
                y_feature = np.random.randint(0, 100, size=(self.segment_length,)).astype(np.int64)
            else:
                y_feature = np.random.randn(self.segment_length, dummy_dim_y).astype(np.float32)
                
        io_end_time = time.time()

        # --- Ensure correct shapes ---
        # For index features, expect [time] or [time, 1], crop to [time]
        # For float features, expect [time, features]
        if self.x_type == 'index':
            if x_feature.ndim == 2 and x_feature.shape[1] == 1:
                x_feature = x_feature.flatten()  # [time, 1] -> [time]
            elif x_feature.ndim != 1:
                raise ValueError(f"Index feature (x) should be 1D or 2D with dim 1, but got shape {x_feature.shape}")
        elif x_feature.ndim == 1:  # Float feature but 1D
            x_feature = x_feature.reshape(-1, 1)

        if self.y_type == 'index':
            if y_feature.ndim == 2 and y_feature.shape[1] == 1:
                y_feature = y_feature.flatten()  # [time, 1] -> [time]
            elif y_feature.ndim != 1:
                raise ValueError(f"Index feature (y) should be 1D or 2D with dim 1, but got shape {y_feature.shape}")
        elif y_feature.ndim == 1:  # Float feature but 1D
            y_feature = y_feature.reshape(-1, 1)

        # Store original lengths before repeat
        x_len_original = x_feature.shape[0]
        y_len_original = y_feature.shape[0]

        # --- Apply repetition (upsampling) ---
        if self.x_repeat > 1:
            if self.x_type == 'index':
                # For index features, repeat along time dimension
                x_feature = np.repeat(x_feature, self.x_repeat)
            else:
                # For float features, repeat along time dimension
                x_feature = np.repeat(x_feature, self.x_repeat, axis=0)
                
        if self.y_repeat > 1:
            if self.y_type == 'index':
                # For index features, repeat along time dimension
                y_feature = np.repeat(y_feature, self.y_repeat)
            else:
                # For float features, repeat along time dimension
                y_feature = np.repeat(y_feature, self.y_repeat, axis=0)
        
        # Get lengths after repeat
        x_len_after_repeat = x_feature.shape[0]
        y_len_after_repeat = y_feature.shape[0]
        
        # Log if lengths differ after repeat
        if x_len_after_repeat != y_len_after_repeat:
            print(f"Warning: After repeat, x length ({x_len_after_repeat}) != y length ({y_len_after_repeat}) for sample {idx}.")
            print(f"  Original lengths: x={x_len_original}, y={y_len_original}")
            print(f"  Repeat factors: x={self.x_repeat}, y={self.y_repeat}")
            
            # Check if length difference exceeds max_length_diff
            length_diff = abs(x_len_after_repeat - y_len_after_repeat)
            if length_diff > self.max_length_diff:
                print(f"Warning: Length difference ({length_diff}) exceeds maximum allowed ({self.max_length_diff}) for sample {idx}. Sample will be filtered.")
                self.filtered_samples.append(idx)
                
                # Return placeholder data (could be refined to skip this sample in training)
                dummy_dim_x = self._x_dim if self._x_dim else 10
                dummy_dim_y = self._y_dim if self._y_dim else 10
                
                # Create appropriate placeholder data
                if self.x_type == 'index':
                    x_feature = np.random.randint(0, 100, size=(self.segment_length,)).astype(np.int64)
                else:
                    x_feature = np.random.randn(self.segment_length, dummy_dim_x).astype(np.float32)
                    
                if self.y_type == 'index':
                    y_feature = np.random.randint(0, 100, size=(self.segment_length,)).astype(np.int64)
                else:
                    y_feature = np.random.randn(self.segment_length, dummy_dim_y).astype(np.float32)

        # --- Record dimensions (only if not already set and first time only) ---
        if self._x_dim is None and self.x_type == 'float':
            self._x_dim = x_feature.shape[1]
        if self._y_dim is None and self.y_type == 'float':
            self._y_dim = y_feature.shape[1]

        # --- Aligned crop ---
        crop_start_time = time.time()
        x_cropped, y_cropped = self._aligned_crop(
            x_feature, y_feature, self.segment_length
        )
        crop_end_time = time.time()
        
        # Record cropped time steps
        if self._time_steps is None:
            self._time_steps = x_cropped.shape[0]

        # --- Normalize (only for float features) ---
        norm_start_time = time.time()
        if self.x_type == 'float':
            x_normalized = self._normalize_feature(x_cropped)
        else:
            x_normalized = x_cropped  # No normalization for index features
            
        if self.y_type == 'float':
            y_normalized = self._normalize_feature(y_cropped)
        else:
            y_normalized = y_cropped  # No normalization for index features
            
        norm_end_time = time.time()

        # --- Convert to tensors ---
        x_tensor = torch.tensor(x_normalized, dtype=self.x_dtype)
        y_tensor = torch.tensor(y_normalized, dtype=self.y_dtype)

        overall_end_time = time.time()
        io_time_ms = (io_end_time - io_start_time) * 1000
        crop_time_ms = (crop_end_time - crop_start_time) * 1000
        norm_time_ms = (norm_end_time - norm_start_time) * 1000
        overall_time_ms = (overall_end_time - overall_start_time) * 1000
        
        # Print timing breakdown (can set a threshold, e.g., only print if total time > 50ms)
        # if overall_time_ms > 50:
        #    print(f"Sample {idx} | Total: {overall_time_ms:.1f}ms | IO: {io_time_ms:.1f}ms | Crop: {crop_time_ms:.1f}ms | Norm: {norm_time_ms:.1f}ms")

        return x_tensor, y_tensor

    def _normalize_feature(self, feature):
        """
        Vectorized feature normalization (only applicable for float features)
        
        Args:
            feature: Feature array, shape [time, feature_dim]
            
        Returns:
            Normalized feature
        """
        if self.normalization == 'none' or feature.shape[1] == 0 or feature.dtype != np.float32:
            return feature

        # Small epsilon to prevent division by zero
        eps = 1e-6
        normalized = np.zeros_like(feature, dtype=np.float32)

        if self.normalization == 'minmax':
            # Compute min and max for each column (axis=0 means compute along time axis)
            f_min = np.min(feature, axis=0, keepdims=True)  # [1, feature_dim]
            f_max = np.max(feature, axis=0, keepdims=True)  # [1, feature_dim]
            denominator = f_max - f_min
            # Handle zero denominator
            valid_mask = denominator > eps
            # Normalize valid columns
            normalized[:, valid_mask[0]] = (feature[:, valid_mask[0]] - f_min[:, valid_mask[0]]) / denominator[:, valid_mask[0]]
            # Set default value (0.5) for columns with zero denominator
            normalized[:, ~valid_mask[0]] = 0.5

        elif self.normalization == 'standard':
            f_mean = np.mean(feature, axis=0, keepdims=True)  # [1, feature_dim]
            f_std = np.std(feature, axis=0, keepdims=True)    # [1, feature_dim]
            # Handle zero standard deviation
            valid_mask = f_std > eps
            # Normalize valid columns
            normalized[:, valid_mask[0]] = (feature[:, valid_mask[0]] - f_mean[:, valid_mask[0]]) / f_std[:, valid_mask[0]]
            # Set zero for columns with zero standard deviation
            normalized[:, ~valid_mask[0]] = 0.0

        elif self.normalization == 'robust':
            # percentile can also specify axis
            q25, q50, q75 = np.percentile(feature, [25, 50, 75], axis=0, keepdims=True)  # [3, 1, feature_dim] -> squeeze
            q25 = q25.squeeze(axis=0)  # [1, feature_dim]
            q50 = q50.squeeze(axis=0)  # [1, feature_dim]
            q75 = q75.squeeze(axis=0)  # [1, feature_dim]

            iqr = q75 - q25  # [1, feature_dim]
            # Handle zero IQR
            valid_mask = iqr > eps
            # Normalize valid columns
            normalized[:, valid_mask[0]] = (feature[:, valid_mask[0]] - q50[:, valid_mask[0]]) / iqr[:, valid_mask[0]]
            # Set zero for columns with zero IQR
            normalized[:, ~valid_mask[0]] = 0.0

        else:  # 'none' or unexpected
            return feature  # Return original feature

        # Final check for NaN values
        if np.isnan(normalized).any():
            print(f"Warning: NaN values after vectorized normalization.")
            normalized = np.nan_to_num(normalized, nan=0.0)  # Replace NaN with 0.0

        return normalized

    def _aligned_crop(self, x, y, length):
        """
        Aligned random crop or padding to specified length.
        When lengths are different, we prefer truncating from the beginning.
        
        Args:
            x: x features array
            y: y features array
            length: desired segment length
            
        Returns:
            Cropped x and y features
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Get time dimension lengths
        len_x = x.shape[0]
        len_y = y.shape[0]

        # Handle shapes
        x_is_1d = (x.ndim == 1)
        y_is_1d = (y.ndim == 1)
        
        if not x_is_1d:
            x_input_format = x.shape  # Store original shape for reference

        if not y_is_1d and self.y_type == 'float':
            y_input_format = y.shape  # Store original shape for reference
        
        # For index features, keep as 1D
        # For float features, ensure they are 2D
        if not x_is_1d and self.x_type == 'index':
            if x.shape[1] == 1:
                x = x.flatten()
                x_is_1d = True
        elif x_is_1d and self.x_type == 'float':
            x = x.reshape(-1, 1)
            x_is_1d = False
            
        if not y_is_1d and self.y_type == 'index':
            if y.shape[1] == 1:
                y = y.flatten()
                y_is_1d = True
        elif y_is_1d and self.y_type == 'float':
            y = y.reshape(-1, 1)
            y_is_1d = False

        # Pad short sequences
        if len_x < length:
            repeats = int(np.ceil(length / len_x))
            if x_is_1d:
                x = np.tile(x, repeats)[:length]  # 1D tiling
            else:
                x = np.tile(x, (repeats, 1))[:length, :]
            len_x = length
            
        if len_y < length:
            repeats = int(np.ceil(length / len_y))
            if y_is_1d:
                y = np.tile(y, repeats)[:length]  # 1D tiling
            else:
                y = np.tile(y, (repeats, 1))[:length, :]
            len_y = length

        # Crop long sequences - prefer truncating from the beginning
        if len_x == len_y:
            # For equal lengths, randomly crop
            max_start = len_x - length
            start = random.randint(0, max_start) if max_start > 0 else 0
            if x_is_1d:
                x_cropped = x[start:start + length]
            else:
                x_cropped = x[start:start + length, :]
                
            if y_is_1d:
                y_cropped = y[start:start + length]
            else:
                y_cropped = y[start:start + length, :]
        else:
            # For different lengths, truncate from beginning
            print(f"Warning: X ({len_x}) and Y ({len_y}) lengths differ before cropping. Truncating from beginning.")
            
            # Start from beginning (no random offset)
            if x_is_1d:
                x_cropped = x[:length]
            else:
                x_cropped = x[:length, :]
                
            if y_is_1d:
                y_cropped = y[:length]
            else:
                y_cropped = y[:length, :]

        return x_cropped, y_cropped

    def get_dims(self):
        """Get feature dimensions and time steps (valid after at least one __getitem__ call)"""
        # If dimensions not set (either through config or detected from data), try to detect them
        if (self.x_type == 'float' and self._x_dim is None) or \
           (self.y_type == 'float' and self._y_dim is None) or \
           self._time_steps is None:
            try:
                self.__getitem__(0)
                print("First call to get_dims, loading first sample to determine dimensions.")
            except Exception as e:
                print(f"Error loading first sample to determine dimensions: {e}")
                return None, None, None
        
        # Return dimensions based on feature types
        x_dim_to_return = self._x_dim if self.x_type == 'float' else None
        y_dim_to_return = self._y_dim if self.y_type == 'float' else None
        
        return x_dim_to_return, y_dim_to_return, self._time_steps

# --- Function to create DataLoaders ---
def create_data_loaders(config_path):
    """
    Create data loaders from configuration file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing 'train_loader', 'valid_loader', 'test_loader',
        'x_dim', 'y_dim', 'seq_len' (x_dim and y_dim may be None for index features)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")

    data_config = config.get('data', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})

    segment_length = data_config.get('segment_length', 100)
    normalization = data_config.get('normalization', 'standard')
    
    # Get feature types and repeat factors
    x_type = model_config.get('x_type', 'float')
    y_type = model_config.get('y_type', 'float')
    
    # Get configured dimensions (if provided)
    x_dim = model_config.get('x_dim')
    y_dim = model_config.get('y_dim')
    
    x_repeat = data_config.get('x_repeat', 1)
    y_repeat = data_config.get('y_repeat', 1)
    max_length_diff = data_config.get('max_length_diff', 5)

    batch_size = train_config.get('batch_size', 64)
    num_workers = train_config.get('num_workers', 4)
    seed = train_config.get('seed', 42)

    # Validate paths
    for key in ['features_dir', 'train_csv', 'valid_csv', 'test_csv']:
        path = data_config.get(key)
        if not path:
            raise ValueError(f"Missing data path in config file: data.{key}")
        if key == 'features_dir' and not os.path.isdir(path):
            print(f"Warning: Features directory '{path}' does not exist.")

    # Create datasets with configured dimensions
    print(f"Creating datasets (x_type: {x_type}, y_type: {y_type}, x_repeat: {x_repeat}, y_repeat: {y_repeat})...")
    print(f"Configured dimensions: x_dim={x_dim}, y_dim={y_dim}")
    
    train_dataset = MIDataset(
        data_config['train_csv'], data_config['features_dir'],
        segment_length=segment_length, normalization=normalization, seed=seed,
        x_type=x_type, y_type=y_type, x_repeat=x_repeat, y_repeat=y_repeat,
        max_length_diff=max_length_diff, x_dim=x_dim, y_dim=y_dim
    )
    valid_dataset = MIDataset(
        data_config['valid_csv'], data_config['features_dir'],
        segment_length=segment_length, normalization=normalization, seed=seed,
        x_type=x_type, y_type=y_type, x_repeat=x_repeat, y_repeat=y_repeat,
        max_length_diff=max_length_diff, x_dim=x_dim, y_dim=y_dim
    )
    test_dataset = MIDataset(
        data_config['test_csv'], data_config['features_dir'],
        segment_length=segment_length, normalization=normalization, seed=seed,
        x_type=x_type, y_type=y_type, x_repeat=x_repeat, y_repeat=y_repeat,
        max_length_diff=max_length_diff, x_dim=x_dim, y_dim=y_dim
    )

    # Get dimension information (configured or detected)
    detected_x_dim, detected_y_dim, seq_len = train_dataset.get_dims()
    
    # Use configured dimensions if provided, otherwise use detected ones
    final_x_dim = x_dim if x_dim is not None else detected_x_dim
    final_y_dim = y_dim if y_dim is not None else detected_y_dim
    
    # Validate dimensions based on feature types
    if x_type == 'float' and final_x_dim is None:
        raise RuntimeError("Could not determine x_dim for float features, check data files and paths or specify in config.")
    if y_type == 'float' and final_y_dim is None:
        raise RuntimeError("Could not determine y_dim for float features, check data files and paths or specify in config.")

    print("-" * 30)
    print("Dataset and dimension information:")
    if x_type == 'index' and y_type == 'index':
        print(f"  Feature types: X=Index (Long), Y=Index (Long)")
        print(f"  Feature dimensions: Both are indices")
    elif x_type == 'index':
        print(f"  Feature types: X=Index (Long), Y=Float")
        y_dim_source = "config" if y_dim is not None else "data"
        print(f"  Feature dimensions: Y={final_y_dim} (from {y_dim_source})")
    elif y_type == 'index':
        print(f"  Feature types: X=Float, Y=Index (Long)")
        x_dim_source = "config" if x_dim is not None else "data"
        print(f"  Feature dimensions: X={final_x_dim} (from {x_dim_source})")
    else:
        print(f"  Feature types: X=Float, Y=Float")
        x_dim_source = "config" if x_dim is not None else "data"
        y_dim_source = "config" if y_dim is not None else "data"
        print(f"  Feature dimensions: X={final_x_dim} (from {x_dim_source}), Y={final_y_dim} (from {y_dim_source})")
    
    print(f"  Cropped sequence length: {seq_len}")
    print(f"  Repeat factors: X={x_repeat}, Y={y_repeat}")
    print(f"  Max length difference: {max_length_diff}")
    print(f"  Normalization method (Float): {normalization}")
    print(f"  Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # Report filtered samples if any
    if train_dataset.filtered_samples:
        print(f"  Filtered {len(train_dataset.filtered_samples)} training samples due to excessive length difference")
    if valid_dataset.filtered_samples:
        print(f"  Filtered {len(valid_dataset.filtered_samples)} validation samples due to excessive length difference")
    if test_dataset.filtered_samples:
        print(f"  Filtered {len(test_dataset.filtered_samples)} test samples due to excessive length difference")
    
    print("-" * 30)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'x_dim': final_x_dim,
        'y_dim': final_y_dim,
        'seq_len': seq_len
    }