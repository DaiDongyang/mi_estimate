# Audio Feature Mutual Information (MINE) Calculator

This tool allows you to calculate the mutual information between different audio feature representations using MINE (Mutual Information Neural Estimation) with advanced features including context windowing and projection layers.

## Pipeline Overview

1. Split audio data into training, validation, and test sets
2. Extract features (EnCodec tokens or spectrograms)
3. Generate CSV files mapping feature pairs
4. Configure and run training with advanced options

## Step-by-Step Guide

### 1. Data Preparation

Split your audio data into train, validation, and test sets:

```bash
find $TRAIN_DATASET_DIR -name "*.wav" > filelist.train
find $VAL_DATASET_DIR -name "*.wav" > filelist.val
find $TEST_DATASET_DIR -name "*.wav" > filelist.test
```

### 2. Feature Extraction

Use the scripts in the `scripts` directory to extract different types of features:

> **Note:** When using models like Wav2vec2.0, MOSHI, etc. for feature extraction, additional environment setup is required.

#### EnCodec Features

```bash
python ./scripts/extract_encodec.py \
    --model_type ${model_type} \
    --flist ${flist} \
    --bandwidth ${bandwidth} \
    --output_dir ${output_dir} \
    --feature_type ${feature_type}
```

#### Spectrogram Features

```bash
python ./scripts/extract_complex_spec.py \
    --flist ${flist} \
    --output_dir ${output_dir}
```

### 3. Generate CSV Files

Create CSV files that map extracted feature pairs for training, validation, and testing:

```bash
python ./scripts/generate_csv.py \
    --x_dir ${train_or_dev_or_test_encodec_feature_dir} \
    --y_dir ${train_or_dev_or_test_spectrogram_feature_dir} \
    --output_csv ${output_csv_path}
```

Repeat this for train, validation, and test sets.

### 4. Configure the Model

The configuration file supports many advanced features. Here's a comprehensive example:

```yaml
# config.yaml - MINE Training Configuration with Advanced Features
data:
  # Data paths
  train_csv: "/path/to/train.csv"
  valid_csv: "/path/to/valid.csv"
  test_csv: "/path/to/test.csv"
  features_dir: "/path/to/features"  # Base directory for features
  
  # Feature parameters
  segment_length: 300       # Cropped sequence length
  normalization: 'standard' # Float feature normalization: none, minmax, standard, robust
  
  # Context window configuration (NEW)
  context_frame_numbers: 3  # Number of consecutive frames to use as context
  # Set to 1 to disable context windowing (original behavior)
  # Higher values capture more temporal dependencies but increase memory usage
  
  # Upsampling factors - supports different sampling rates between x and y
  x_repeat: 4               # Repeat factor for x features (e.g., 12.5Hz -> 50Hz)
  y_repeat: 1               # Repeat factor for y features
  
  # Maximum allowed difference between x and y lengths after repeat
  max_length_diff: 5        # Samples exceeding this will be filtered out

model:
  # Feature types - supports four combinations:
  x_type: 'index'           # 'float' or 'index'
  y_type: 'float'           # 'float' or 'index'
  
  # Feature dimensions (for float type)
  x_dim: 256                # Base dimension of x features when x_type='float'
  y_dim: 2050               # Base dimension of y features when y_type='float'
  
  # Projection dimensions (NEW) - for dimensionality reduction
  # Reduces overfitting and computational cost
  x_proj_dim: 64            # Project x features to this dimension (null to disable)
  y_proj_dim: 128           # Project y features to this dimension (null to disable)
  
  # Required parameters for index features
  x_vocab_size: 6561        # Size of vocabulary for x when x_type='index'
  x_embedding_dim: 256      # Embedding dimension for x
  
  y_vocab_size: 2000        # Size of vocabulary for y when y_type='index'
  y_embedding_dim: 128      # Embedding dimension for y
  
  # Neural network architecture
  hidden_dims: [128, 64]    # MLP hidden layer dimensions
  activation: 'relu'        # Activation: relu, leaky_relu, elu
  batch_norm: True          # Whether to use BatchNorm
  dropout: 0.2              # Dropout rate (0 = no dropout)
  bias_correction_method: 'none'  # 'none' or 'default'

training:
  epochs: 100               # Number of training epochs
  batch_size: 128           # Batch size (reduce if OOM)
  lr: 0.0001                # Learning rate
  device: 'cuda'            # 'cuda' or 'cpu'
  num_workers: 16           # Data loading processes
  seed: 42                  # Random seed for reproducibility

# Validation configuration (NEW) - for large datasets
validation:
  validate_every_n_steps: 1000   # Validate every N training steps
  # Set to null to validate once per epoch (default)
  early_stopping_patience: 10    # Stop if no improvement for N validations
  # Set to null to disable early stopping

checkpoint:
  dir: './checkpoints'      # Directory to save checkpoints
  save_interval: 10         # Save checkpoint every N epochs
  resume_from: null         # Path to checkpoint to resume from

logging:
  log_interval: 5           # Print detailed statistics every N epochs
```

### 5. Advanced Features

#### Context Windowing

Context windowing captures temporal dependencies by using multiple consecutive frames:

- **How it works**: Instead of processing single frames, the model processes windows of consecutive frames
- **Configuration**: Set `context_frame_numbers` > 1 to enable
- **Example**: With `context_frame_numbers: 3`:
  - Original: `[x1, x2, x3, x4, x5]`
  - Windowed: `[x1,x2,x3]`, `[x2,x3,x4]`, `[x3,x4,x5]`
- **Memory impact**: Increases effective feature dimension by `context_frame_numbers`×

#### Projection Layers

Projection layers reduce dimensionality before the MLP, helping with:
- **Overfitting reduction**: Fewer parameters in the main network
- **Memory efficiency**: Smaller intermediate representations
- **Computational speed**: Faster training and inference

Example dimension flow with projection:
```
x: [batch, 768] -> projection -> [batch, 64] ⎤
                                              ├-> concat -> [batch, 192] -> MLP
y: [batch, 512] -> projection -> [batch, 128] ⎦
```

#### Feature Type Combinations

The system supports four combinations:

1. **Float-Float**: Both features are continuous
   - Best for: Spectrograms, mel-features, continuous representations
   - Supports: Normalization, projection layers, context windowing

2. **Index-Float**: Discrete source, continuous target
   - Best for: Token-to-spectrogram mapping
   - Supports: Embeddings for source, projection for target

3. **Float-Index**: Continuous source, discrete target
   - Best for: Spectrogram-to-token mapping
   - Supports: Projection for source, embeddings for target

4. **Index-Index**: Both features are discrete
   - Best for: Token-to-token mapping
   - Supports: Embeddings for both, no projection needed

### 6. Training

Start training with your configuration:

```bash
python train.py -c path/to/config.yaml
```

The training script now supports:
- **Periodic validation**: Validate every N steps for large datasets
- **Early stopping**: Stop training when validation MI stops improving
- **Checkpoint resuming**: Continue training from a saved checkpoint
- **Progress tracking**: Real-time MI estimates during training

#### Training Output

The training process will display:
```
Epoch 10/100 | Time: 45.2s | Steps: 1500 | Train MI: 2.3456 | Val MI: 2.2134 | Best Val MI: 2.2567
```

And periodically show detailed statistics:
```
-------------------- Epoch 10 Stats --------------------
  Recent Batches: 50
  Avg Joint T(x,y): 2.4521
  Std Joint T(x,y): 0.1234
  Avg Marginal T(x,y'): 0.0123
  Std Marginal T(x,y'): 0.0456
  Avg Gradient Norm: 0.8765
  Current LR: 0.0001
-------------------------------------------------------
```

### 7. Performance Optimization

#### Memory Optimization
- Reduce `batch_size` if running out of GPU memory
- Use smaller `context_frame_numbers` for high-dimensional features
- Enable projection layers to reduce intermediate dimensions
- Reduce `num_workers` if CPU memory is limited

#### Model Complexity Control
- Always use projection layers when `context_frame_numbers` > 1
- Start with smaller `hidden_dims` and increase if underfitting
- Increase `dropout` for larger models to prevent overfitting

#### Training Stability
- Start with smaller learning rates when using context windows
- Monitor gradient norms (shown in periodic stats)
- Use `bias_correction_method: 'default'` if MI estimates are unstable

#### Recommended Settings by Feature Dimension

For low-dimensional features (<100 dims):
```yaml
context_frame_numbers: 5-10
x_proj_dim: null  # No projection needed
hidden_dims: [256, 128]
```

For medium-dimensional features (100-500 dims):
```yaml
context_frame_numbers: 3-5
x_proj_dim: 64-128  # Moderate projection
hidden_dims: [128, 64]
```

For high-dimensional features (>500 dims):
```yaml
context_frame_numbers: 2-3
x_proj_dim: 32-64  # Strong projection
hidden_dims: [64, 32]
```

### 8. Interpreting Results

The mutual information values represent the amount of shared information between your feature pairs:
- **Higher MI**: Features are more related/dependent
- **Lower MI**: Features are more independent
- **Typical ranges**: 0.1-5.0 nats (depending on feature types)

The upper bound of MI is min(H(X), H(Y)), where H represents entropy.

### 9. Troubleshooting

**Out of Memory (OOM)**:
- Reduce `batch_size`
- Reduce `context_frame_numbers`
- Enable/increase projection (smaller `x_proj_dim`, `y_proj_dim`)
- Reduce `segment_length`

**Unstable Training**:
- Reduce learning rate
- Enable `bias_correction_method: 'default'`
- Check for NaN values in features
- Ensure proper normalization for float features

**Slow Training**:
- Enable `validate_every_n_steps` for large datasets
- Reduce `num_workers` if I/O bound
- Use projection layers to reduce computation

**Poor Convergence**:
- Increase model capacity (`hidden_dims`)
- Try different `activation` functions
- Adjust `context_frame_numbers`
- Check feature alignment and quality