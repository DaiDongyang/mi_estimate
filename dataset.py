# dataset.py
"""
数据集模块 - 加载预处理好的特征文件，进行对齐裁剪和归一化
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import yaml # 用于加载config

class MIDataset(Dataset):
    """互信息数据集，加载预处理好的特征文件"""

    def __init__(self, csv_path, features_dir, segment_length=100,
                 normalization='standard', seed=42, x_dtype=torch.float32): # 新增x_dtype
        """
        初始化数据集

        参数:
            csv_path: CSV文件路径 (包含id, source_feature, target_feature列)
            features_dir: 特征文件目录
            segment_length: 随机裁剪的片段长度
            normalization: 特征归一化方式 ('none', 'minmax', 'standard', 'robust')
            seed: 随机种子
            x_dtype: 源特征的数据类型 (torch.float32 或 torch.long)
        """
        self.features_dir = features_dir
        self.segment_length = segment_length
        self.normalization = normalization
        self.x_dtype = x_dtype # 存储期望的x类型

        random.seed(seed)
        np.random.seed(seed)

        self.data = pd.read_csv(csv_path)
        required_cols = ['id', 'source_feature', 'target_feature']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"CSV必须包含 {required_cols} 列")

        self._source_dim = None
        self._target_dim = None
        self._time_steps = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据样本，返回对齐裁剪后的张量。
        source_tensor 的类型由 self.x_dtype 控制。
        target_tensor 始终为 float32。
        """
        overall_start_time = time.time()
        row = self.data.iloc[idx]

        source_path = row['source_feature']
        if not os.path.isabs(source_path):
            source_path = os.path.join(self.features_dir, source_path)

        target_path = row['target_feature']
        if not os.path.isabs(target_path):
            target_path = os.path.join(self.features_dir, target_path)

        io_start_time = time.time()
        try:
            # --- 修改: 根据x_dtype加载source ---
            if self.x_dtype == torch.long:
                 # 假设 .npy 文件存储的是整数索引
                 source_feature = np.load(source_path).astype(np.int64)
            else:
                 source_feature = np.load(source_path).astype(np.float32)
            # target 始终是 float
            target_feature = np.load(target_path).astype(np.float32)

        except FileNotFoundError:
             print(f"错误: 文件未找到 - Source: {source_path} 或 Target: {target_path}")
             # 返回占位符数据
             dummy_dim_s = self._source_dim if self._source_dim else 10
             dummy_dim_t = self._target_dim if self._target_dim else 10
             feature_shape = (self.segment_length, dummy_dim_s) if self.x_dtype == torch.float32 else (self.segment_length,)
             source_feature = np.random.randn(*feature_shape)
             if self.x_dtype == torch.long: source_feature = np.random.randint(0, 100, size=feature_shape).astype(np.int64) # 假设词汇量100
             else: source_feature = source_feature.astype(np.float32)
             target_feature = np.random.randn(self.segment_length, dummy_dim_t).astype(np.float32)
        except Exception as e:
            print(f"加载特征文件时出错 (idx: {idx}): {e}")
            print(f"  Source: {source_path}")
            print(f"  Target: {target_path}")
            # 返回占位符
            dummy_dim_s = self._source_dim if self._source_dim else 10
            dummy_dim_t = self._target_dim if self._target_dim else 10
            feature_shape = (self.segment_length, dummy_dim_s) if self.x_dtype == torch.float32 else (self.segment_length,)
            source_feature = np.random.randn(*feature_shape)
            if self.x_dtype == torch.long: source_feature = np.random.randint(0, 100, size=feature_shape).astype(np.int64)
            else: source_feature = source_feature.astype(np.float32)
            target_feature = np.random.randn(self.segment_length, dummy_dim_t).astype(np.float32)
        io_end_time = time.time()

        # --- 确保形状 ---
        # 如果 x 是 index (long)，期望是 [time] 或 [time, 1]，裁剪后应为 [time]
        # 如果 x 是 float，期望是 [time, features]
        if self.x_dtype == torch.long:
            if source_feature.ndim == 2 and source_feature.shape[1] == 1:
                source_feature = source_feature.flatten() # [time, 1] -> [time]
            elif source_feature.ndim != 1:
                 raise ValueError(f"Index feature (x) should be 1D or 2D with dim 1, but got shape {source_feature.shape}")
        elif source_feature.ndim == 1: # Float feature but 1D
            source_feature = source_feature.reshape(-1, 1)

        # target 始终是 float，确保是 [time, features]
        if target_feature.ndim == 1:
             target_feature = target_feature.reshape(-1, 1)

        # --- 记录维度 (只记录一次) ---
        if self._source_dim is None and self.x_dtype == torch.float32:
             self._source_dim = source_feature.shape[1]
        if self._target_dim is None:
             self._target_dim = target_feature.shape[1]

        # --- 对齐裁剪 ---
        crop_start_time = time.time()
        source_cropped, target_cropped = self._aligned_crop(
            source_feature, target_feature, self.segment_length
        )
        crop_end_time = time.time()
        # 记录裁剪后的时间步长
        if self._time_steps is None: self._time_steps = source_cropped.shape[0]

        # --- 归一化 (只对 float 特征) ---
        norm_start_time = time.time()
        if self.x_dtype == torch.float32:
             source_normalized = self._normalize_feature(source_cropped)
        else:
             source_normalized = source_cropped # Index 不需要归一化
        target_normalized = self._normalize_feature(target_cropped)
        norm_end_time = time.time()

        # --- 转换为张量 ---
        source_tensor = torch.tensor(source_normalized, dtype=self.x_dtype) # 使用指定的 dtype
        target_tensor = torch.tensor(target_normalized, dtype=torch.float32) # target 始终是 float

        overall_end_time = time.time() # 总结束时间
        io_time_ms = (io_end_time - io_start_time) * 1000
        crop_time_ms = (crop_end_time - crop_start_time) * 1000
        norm_time_ms = (norm_end_time - norm_start_time) * 1000
        # tensor_time_ms = (tensor_end_time - tensor_start_time) * 1000
        overall_time_ms = (overall_end_time - overall_start_time) * 1000
        
            # 打印耗时细分 (可以设置一个阈值，比如总时间 > 50ms 才打印)
        # if overall_time_ms > 50:
            # print(f"Sample {idx} | Total: {overall_time_ms:.1f}ms | IO: {io_time_ms:.1f}ms | Crop: {crop_time_ms:.1f}ms | Norm: {norm_time_ms:.1f}ms")

        return source_tensor, target_tensor

    def _normalize_feature(self, feature):
        """
        向量化实现的特征归一化 (只对 float 特征有效)

        参数:
            feature: 特征数组，形状为 [time, feature_dim]

        返回:
            归一化后的特征
        """
        if self.normalization == 'none' or feature.shape[1] == 0 or feature.dtype != np.float32:
            return feature

        # 设置一个小的 epsilon 防止除以零
        eps = 1e-6
        normalized = np.zeros_like(feature, dtype=np.float32)

        if self.normalization == 'minmax':
            # 计算每列的 min 和 max (axis=0 表示沿时间轴计算)
            f_min = np.min(feature, axis=0, keepdims=True)  # [1, feature_dim]
            f_max = np.max(feature, axis=0, keepdims=True)  # [1, feature_dim]
            denominator = f_max - f_min
            # 处理分母为零的情况
            valid_mask = denominator > eps
            # 对有效列进行归一化
            normalized[:, valid_mask[0]] = (feature[:, valid_mask[0]] - f_min[:, valid_mask[0]]) / denominator[:, valid_mask[0]]
            # 分母为零的列设为 0.5 (或其他默认值)
            normalized[:, ~valid_mask[0]] = 0.5

        elif self.normalization == 'standard':
            f_mean = np.mean(feature, axis=0, keepdims=True) # [1, feature_dim]
            f_std = np.std(feature, axis=0, keepdims=True)   # [1, feature_dim]
            # 处理标准差为零的情况
            valid_mask = f_std > eps
            # 对有效列进行标准化
            normalized[:, valid_mask[0]] = (feature[:, valid_mask[0]] - f_mean[:, valid_mask[0]]) / f_std[:, valid_mask[0]]
            # 标准差为零的列设为 0
            normalized[:, ~valid_mask[0]] = 0.0

        elif self.normalization == 'robust':
            # percentile 也可以指定 axis
            q25, q50, q75 = np.percentile(feature, [25, 50, 75], axis=0, keepdims=True) # [3, 1, feature_dim] -> squeeze
            q25 = q25.squeeze(axis=0) # [1, feature_dim]
            q50 = q50.squeeze(axis=0) # [1, feature_dim]
            q75 = q75.squeeze(axis=0) # [1, feature_dim]

            iqr = q75 - q25 # [1, feature_dim]
            # 处理 IQR 为零的情况
            valid_mask = iqr > eps
            # 对有效列进行标准化
            normalized[:, valid_mask[0]] = (feature[:, valid_mask[0]] - q50[:, valid_mask[0]]) / iqr[:, valid_mask[0]]
            # IQR 为零的列设为 0
            normalized[:, ~valid_mask[0]] = 0.0

        else: # 'none' or unexpected
            return feature # 直接返回原始特征

        # 最后再检查一次 NaN (理论上向量化并处理零分母后不应出现)
        if np.isnan(normalized).any():
            print(f"警告: 向量化归一化后仍出现 NaN 值。")
            normalized = np.nan_to_num(normalized, nan=0.0) # 使用 nan_to_num 处理可能存在的 NaN

        return normalized


    def _aligned_crop(self, source, target, length):
        """对齐地随机裁剪或填充到指定长度"""
        source = np.asarray(source)
        target = np.asarray(target)

        # 获取时间维度长度
        len_s = source.shape[0]
        len_t = target.shape[0]

        # 处理形状（确保 target 是 2D）
        if target.ndim == 1: target = target.reshape(-1, 1)
        # source 根据类型可能是 1D 或 2D
        source_is_1d = (source.ndim == 1)

        # 填充短序列
        if len_s < length:
            repeats = int(np.ceil(length / len_s))
            pad_axis = 0 if source_is_1d else (repeats, 1)
            if not source_is_1d:
                source = np.tile(source, pad_axis)[:length, :]
            else:
                source = np.tile(source, repeats)[:length] # 1D tiling
            len_s = length
        if len_t < length:
            repeats = int(np.ceil(length / len_t))
            target = np.tile(target, (repeats, 1))[:length, :]
            len_t = length

        # 裁剪长序列
        if len_s == len_t:
            max_start = len_s - length
            start = random.randint(0, max_start) if max_start > 0 else 0
            source_cropped = source[start : start + length] # Works for 1D and 2D first axis
            target_cropped = target[start : start + length, :]
        else:
            print(f"警告: 源 ({len_s}) 和目标 ({len_t}) 裁剪前长度不一致。将尝试相对位置裁剪。")
            max_start_s = len_s - length
            max_start_t = len_t - length
            rel_pos = random.random()
            start_s = int(rel_pos * max_start_s) if max_start_s > 0 else 0
            start_t = int(rel_pos * max_start_t) if max_start_t > 0 else 0
            source_cropped = source[start_s : start_s + length]
            target_cropped = target[start_t : start_t + length, :]

        return source_cropped, target_cropped


    def get_dims(self):
        """获取特征维度和时间步长 (在至少一次 __getitem__ 调用后有效)"""
        if self._target_dim is None or self._time_steps is None or (self.x_dtype==torch.float32 and self._source_dim is None):
            try:
                self.__getitem__(0)
                print("首次调用 get_dims，加载第一个样本以确定维度。")
            except Exception as e:
                 print(f"加载第一个样本以确定维度时出错: {e}")
                 return None, None, None
        # 返回 None for x_dim if x is index type
        x_dim_to_return = self._source_dim if self.x_dtype == torch.float32 else None
        return x_dim_to_return, self._target_dim, self._time_steps

# --- 创建 DataLoaders 的函数 ---
def create_data_loaders(config_path):
    """
    从配置文件创建数据加载器

    参数:
        config_path: YAML 配置文件的路径

    返回:
        包含 'train_loader', 'valid_loader', 'test_loader',
        'x_dim', 'y_dim', 'seq_len' 的字典 (x_dim 可能为 None)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"解析配置文件时出错: {e}")

    data_config = config.get('data', {})
    model_config = config.get('model', {}) # 需要 model 配置来确定 x_dtype
    train_config = config.get('training', {})

    segment_length = data_config.get('segment_length', 100)
    normalization = data_config.get('normalization', 'standard')
    network_type = model_config.get('network_type', 'float') # 从模型配置获取类型
    x_dtype = torch.long if network_type == 'index' else torch.float32 # 确定 x 的数据类型

    batch_size = train_config.get('batch_size', 64)
    num_workers = train_config.get('num_workers', 4)
    seed = train_config.get('seed', 42)

    # 验证路径
    for key in ['features_dir', 'train_csv', 'valid_csv', 'test_csv']:
         path = data_config.get(key)
         if not path: raise ValueError(f"配置文件中缺少数据路径: data.{key}")
         if key == 'features_dir' and not os.path.isdir(path): print(f"警告: 特征目录 '{path}' 不存在。")


    # --- 创建数据集 (传入 x_dtype) ---
    print(f"创建数据集 (x 类型: {x_dtype})...")
    train_dataset = MIDataset(
        data_config['train_csv'], data_config['features_dir'],
        segment_length=segment_length, normalization=normalization, seed=seed, x_dtype=x_dtype
    )
    valid_dataset = MIDataset(
        data_config['valid_csv'], data_config['features_dir'],
        segment_length=segment_length, normalization=normalization, seed=seed, x_dtype=x_dtype
    )
    test_dataset = MIDataset(
        data_config['test_csv'], data_config['features_dir'],
        segment_length=segment_length, normalization=normalization, seed=seed, x_dtype=x_dtype
    )

    # --- 获取维度信息 ---
    x_dim, y_dim, seq_len = train_dataset.get_dims()
    if y_dim is None: # y_dim is always needed
         raise RuntimeError("无法确定数据集的维度 (y_dim)，请检查数据文件和路径。")
    # x_dim is only needed for float type network
    if network_type == 'float' and x_dim is None:
        raise RuntimeError("无法确定数据集的 x_dim，对于 float 网络类型是必需的。")


    print("-" * 30)
    print("数据集和维度信息:")
    if network_type == 'index':
        print(f"  特征类型: X=Index (Long), Y=Float")
        print(f"  特征维度: Y={y_dim}")
    else:
        print(f"  特征类型: X=Float, Y=Float")
        print(f"  特征维度: X={x_dim}, Y={y_dim}")
    print(f"  裁剪后的序列长度 (Seq Len): {seq_len}")
    print(f"  归一化方法 (Float): {normalization}")
    print(f"  数据集大小 - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    print("-" * 30)


    # --- 创建数据加载器 ---
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
        'x_dim': x_dim, # Might be None if network_type is 'index'
        'y_dim': y_dim,
        'seq_len': seq_len
    }