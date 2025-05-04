# Audio Feature Mutual Information (MINE) Calculator

This tool allows you to calculate the mutual information between different audio feature representations using MINE (Mutual Information Neural Estimation).

## Pipeline Overview

1. Split audio data into training, validation, and test sets
2. Extract features (EnCodec tokens or spectrograms)
3. Generate CSV files mapping feature pairs
4. Configure and run training

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

Modify the config.yaml file based on your feature types. Below is an example configuration for calculating mutual information between Cosyvoice S3 tokens and complex spectrograms:

```yaml
# config.yaml - MINE Training Configuration Example
data:
  # Data paths
  train_csv: "/path/to/datasets/features/s3token_complex_spec/train.csv"
  valid_csv: "/path/to/datasets/features/s3token_complex_spec/dev.csv"
  test_csv: "/path/to/datasets/features/s3token_complex_spec/test.csv"
  features_dir: "/path/to/datasets/features/csv" # Not used if absolute paths are in CSVs
  
  # Feature parameters
  segment_length: 300       # Cropped sequence length
  normalization: 'standard' # Float feature normalization: none, minmax, standard, robust
  
  # Upsampling factors - supports different sampling rates between x and y
  x_repeat: 4               # Repeat factor for x features (upsampling), e.g., 10ms -> 20ms
  y_repeat: 1               # Repeat factor for y features, leave at 1 if no upsampling needed
  
  # Maximum allowed difference between x and y lengths after repeat
  # Samples exceeding this difference will be filtered out
  max_length_diff: 5        # Max absolute difference between x_length and y_length

model:
  # Feature type options (select appropriate pair):
  # Option 1: Both floating point features
  # x_type: 'float'          # Source features are continuous (float)
  # y_type: 'float'          # Target features are continuous (float)
  
  # Option 2: Source is index, target is float
  x_type: 'index'        # Source features are discrete indices (long)
  y_type: 'float'        # Target features are continuous (float)
  
  # Option 3: Source is float, target is index
  # x_type: 'float'        # Source features are continuous (float)
  # y_type: 'index'        # Target features are discrete indices (long)
  
  # Option 4: Both index features
  # x_type: 'index'        # Source features are discrete indices (long)
  # y_type: 'index'        # Target features are discrete indices (long)
  
  # --- Feature dimensions (for float type) ---
  x_dim: 256            # Dimension of x features when x_type='float'
  y_dim: 2050           # Dimension of y features when y_type='float'
  
  # --- Required parameters for index features ---
  # Vocabulary size and embedding dimension for x (required when x_type='index')
  x_vocab_size: 6561    # Size of vocabulary for x 
  x_embedding_dim: 256  # Embedding dimension for x
  
  # Vocabulary size and embedding dimension for y (required when y_type='index')
  y_vocab_size: 2000    # Size of vocabulary for y
  y_embedding_dim: 2050 # Embedding dimension for y
  
  # --- Neural network architecture ---
  hidden_dims: [64]     # Network hidden layer dimensions
  activation: 'relu'    # Activation function: relu, leaky_relu, elu
  batch_norm: True      # Whether to use BatchNorm
  dropout: 0.5          # Dropout rate (0 means none)

training:
  epochs: 500           # Number of training epochs
  batch_size: 64        # Batch size
  lr: 0.0001            # Learning rate
  device: 'cuda'        # Use 'cuda' or 'cpu'
  num_workers: 16       # Number of data loading processes
  seed: 42              # Random seed

checkpoint:
  dir: './s3token_complex_spec/checkpoints'  # Directory to save checkpoints
  save_interval: 10     # Save checkpoint every N epochs
  resume_from: null     # Path to checkpoint to resume from (null to start from scratch)

logging:
  log_interval: 5       # Print detailed statistics every this many epochs
```

Important configuration notes:
- Set correct paths for your CSV files
- Specify appropriate feature types (`x_type` and `y_type`)
- Configure dimensions and vocabulary sizes based on your features
- Adjust upsampling factors (`x_repeat` and `y_repeat`) if needed
- Set neural network architecture as appropriate

### 5. Training

Start the training process:

```bash
python train.py -c path/to/config.yaml
```

## Feature Types

The system supports various combinations of feature types:
- Float-Float: Both features are continuous
- Index-Float: Source is discrete (e.g., EnCodec tokens), target is continuous
- Float-Index: Source is continuous, target is discrete
- Index-Index: Both features are discrete

## Theoretical Background

Mutual information measures how much information one variable provides about another. The upper bound of mutual information I(X;Y) is min(H(X), H(Y)), where H(X) and H(Y) are the entropy values of X and Y respectively. This tool uses neural estimation techniques (MINE) for calculating mutual information between complex audio features.