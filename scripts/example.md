## 特征提取和数据准备脚本使用示例

### 1. 提取复数谱特征

```bash
# 从音频目录提取复数谱特征
python extract_complex_spec.py --input_dir /path/to/audio_files --output_dir /path/to/complex_features --n_fft 2048 --hop_length 512
```

### 2. 提取梅尔谱特征

```bash
# 从音频目录提取梅尔谱特征
python extract_mel_spec.py --input_dir /path/to/audio_files --output_dir /path/to/mel_features --sr 22050 --n_fft 2048 --hop_length 512 --n_mels 128
```

### 3. 生成数据集CSV文件

```bash
# 基于特征目录生成CSV文件
python generate_csv.py --mel_dir /path/to/mel_features --complex_dir /path/to/complex_features --output_dir /path/to/csv_files --train_ratio 0.8 --valid_ratio 0.1 --seed 42

# 如果需要反转源和目标（使复数谱为源，梅尔谱为目标）
python generate_csv.py --mel_dir /path/to/mel_features --complex_dir /path/to/complex_features --output_dir /path/to/csv_files --reverse
```

### 4. 完整流程示例

```bash
# 1. 提取复数谱特征
python extract_complex_spec.py --input_dir /data/audio --output_dir /data/features/complex

# 2. 提取梅尔谱特征
python extract_mel_spec.py --input_dir /data/audio --output_dir /data/features/mel

# 3. 生成CSV文件
python generate_csv.py --mel_dir /data/features/mel --complex_dir /data/features/complex --output_dir /data/csv

# 4. 配置MINE训练
# 编辑config.yaml设置路径:
# data:
#   train_csv: "/data/csv/train.csv"
#   valid_csv: "/data/csv/valid.csv"
#   test_csv: "/data/csv/test.csv"
#   features_dir: "/data/features"  # 用于相对路径

# 5. 训练MINE模型
python train.py --config config.yaml
```