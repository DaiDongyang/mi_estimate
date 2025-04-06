# train.py
"""
训练 MINE 模型的主脚本
"""
import torch
import yaml
import os
import time
import argparse  # 导入 argparse
from tqdm import tqdm # 用于进度条

# 假设 mine_estimator 和 dataset 在同一目录下或 Python 路径中
from mine_estimator import MINE
from dataset import create_data_loaders

def main(config_path):
    """主训练函数，接收配置文件路径作为参数"""
    # --- 1. 加载配置 ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"成功加载配置文件: {config_path}")
    except FileNotFoundError:
        # argparse 应该在调用 main 之前处理文件不存在的情况，但这里保留以防万一
        print(f"错误: 配置文件未找到 {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"错误: 解析配置文件失败 {config_path} - {e}")
        return
    except Exception as e:
        print(f"加载或解析配置文件时发生未知错误: {e}")
        return

    model_config = config.get('model', {})
    train_config = config.get('training', {})
    log_config = config.get('logging', {})

    # --- 2. 创建数据加载器并获取维度 ---
    try:
        print("正在创建数据加载器...")
        data_info = create_data_loaders(config_path)
        train_loader = data_info['train_loader']
        valid_loader = data_info['valid_loader']
        test_loader = data_info['test_loader']
        x_dim = data_info['x_dim'] # 可能为 None
        y_dim = data_info['y_dim']
        print("数据加载器创建完成。")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
         print(f"错误: 创建数据加载器失败 - {e}")
         print("请检查配置文件中的 data 部分路径和参数是否正确。")
         return

    # --- 3. 初始化 MINE 模型 ---
    network_type = model_config.get('network_type', 'float')
    vocab_size = model_config.get('vocab_size', None) # 仅用于 'index'
    embedding_dim = model_config.get('embedding_dim', None) # 仅用于 'index'

    print(f"准备初始化 MINE 模型 (类型: {network_type})...")
    # 检查 'index' 类型所需的参数
    if network_type == 'index' and (vocab_size is None or embedding_dim is None):
        print("错误: 当 network_type='index' 时，必须在 config.yaml 中指定 model.vocab_size 和 model.embedding_dim")
        return
    # 检查 'float' 类型所需的参数 (x_dim 在 data_info 中获取)
    if network_type == 'float' and x_dim is None:
         print("错误: 无法从数据确定 x_dim，这对于 network_type='float' 是必需的。")
         print("请确保数据加载正常并且 CSV 文件指向有效的 float 特征。")
         return

    try:
        mine_model = MINE(
            x_dim=x_dim, # 传递 x_dim (如果可用)
            y_dim=y_dim, # y_dim 始终需要
            network_type=network_type,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dims=model_config.get('hidden_dims', [512, 512, 256]),
            activation=model_config.get('activation', 'relu'),
            batch_norm=model_config.get('batch_norm', True),
            dropout=model_config.get('dropout', 0.1),
            lr=train_config.get('lr', 1e-4),
            device=train_config.get('device', 'cuda')
        )
        print("MINE 模型初始化成功。")
    except Exception as e:
        print(f"错误: 初始化 MINE 模型失败 - {e}")
        return

    # --- 4. 训练循环 ---
    epochs = train_config.get('epochs', 100)
    log_interval = log_config.get('log_interval', 10) # 调整打印间隔

    print("\n" + "="*30)
    print(f"开始训练 MINE 模型 (Type: {network_type})")
    print(f"总 Epochs: {epochs}")
    print(f"使用设备: {mine_model.device}")
    print("="*30 + "\n")

    start_time = time.time()
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # --- 训练 ---
        try:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False, ncols=100)
            avg_train_mi, avg_train_loss = mine_model.train_epoch(pbar)
            pbar.close()
            if len(pbar) == 0: # 检查 train_loader 是否为空
                print(f"警告: Epoch {epoch} 的训练加载器为空，无法进行训练。")
                continue # 跳过此 epoch 的剩余部分
        except Exception as e:
            print(f"\n错误: Epoch {epoch} 训练期间发生错误 - {e}")
            # 可以选择继续下一个 epoch 或停止训练
            print("跳过此 Epoch 的训练。")
            continue # 跳到下一个 epoch

        # --- 验证 ---
        try:
            avg_val_mi = mine_model.validate(valid_loader)
            # 检查是否有新的最佳验证 MI
            if mine_model.best_val_mi == avg_val_mi and avg_val_mi != float('-inf'):
                best_epoch = epoch
        except Exception as e:
            print(f"\n错误: Epoch {epoch} 验证期间发生错误 - {e}")
            print("将跳过此 Epoch 的验证和日志记录。")
            continue # 跳到下一个 epoch


        epoch_time = time.time() - epoch_start_time

        # --- 打印日志 ---
        print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s | "
              f"Train MI: {avg_train_mi:.6f} | Train Loss: {avg_train_loss:.6f} | "
              f"Val MI: {avg_val_mi:.6f} | Best Val MI: {mine_model.best_val_mi:.6f} (Epoch {best_epoch if best_epoch > 0 else 'N/A'})")


        # 定期打印更详细的网络统计信息
        if epoch % log_interval == 0 or epoch == epochs:
             print("-" * 20 + f" Epoch {epoch} Stats " + "-" * 20)
             stats_info = mine_model.get_network_stats()
             print(stats_info if isinstance(stats_info, str) else "\n".join([f"  {k}: {v}" for k, v in stats_info.items()]))
             print("-" * (40 + len(f" Epoch {epoch} Stats ")))


    total_training_time = time.time() - start_time
    print("\n" + "="*30)
    print("训练完成")
    print(f"总训练时间: {total_training_time:.2f}s ({total_training_time/60:.2f} 分钟)")
    print(f"最佳验证 MI: {mine_model.best_val_mi:.6f} (在 Epoch {best_epoch} 达到)")
    print("="*30 + "\n")

    # --- 5. 测试 ---
    # 通常使用最佳验证模型进行测试，如果保存了的话需要加载
    # 这里我们直接用训练结束时的模型进行评估
    print("在测试集上评估最终模型...")
    try:
        test_mi = mine_model.evaluate(test_loader)
        print(f"测试集 MI: {test_mi:.6f}")
    except Exception as e:
        print(f"错误: 在测试集上评估时发生错误 - {e}")

    print("="*30)


if __name__ == "__main__":
    # --- 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description='使用指定的配置文件训练 MINE 模型。')

    # 添加 --config 参数
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,  # 使配置文件参数成为必需
        help='指向 YAML 配置文件的路径 (必需)'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取配置文件路径
    config_file_path = args.config

    # --- 检查配置文件是否存在 ---
    if not os.path.exists(config_file_path):
        print(f"错误: 配置文件未找到 '{config_file_path}'")
        print("请提供一个有效的配置文件路径。")
        # 打印示例配置帮助用户创建
        print("\n你可以创建一个类似下面内容的 config.yaml 文件:")
        example_config = """
# config.yaml - MINE 训练配置 (示例)
data:
  features_dir: './features'  # <<<--- 修改为特征 .npy 目录
  train_csv: './train_list.csv' # <<<--- 修改为训练列表 CSV
  valid_csv: './valid_list.csv' # <<<--- 修改为验证列表 CSV
  test_csv: './test_list.csv'   # <<<--- 修改为测试列表 CSV
  segment_length: 100       # 裁剪的序列长度
  normalization: 'standard' # Float 特征归一化: none, minmax, standard, robust

model:
  # --- 选择网络类型 ---
  network_type: 'float'     # 'float' 或 'index'

  # --- 如果 network_type is 'index', 需要下面两个参数 ---
  # vocab_size: 1000          # 示例: X 的词汇表大小 (根据你的数据修改)
  # embedding_dim: 128        # 示例: X 的 Embedding 维度

  # --- 通用 MLP 参数 ---
  hidden_dims: [256, 256]   # 网络隐藏层维度
  activation: 'relu'        # 激活函数: relu, leaky_relu, elu
  batch_norm: True          # 是否使用 BatchNorm
  dropout: 0.1              # Dropout 率 (0 表示不用)

training:
  epochs: 50                # 训练轮数
  batch_size: 128           # 批次大小
  lr: 0.0002                # 学习率
  device: 'cuda'            # 使用 'cuda' 或 'cpu'
  num_workers: 4            # 数据加载进程数 (根据你的机器调整)
  seed: 42                  # 随机种子

logging:
  log_interval: 10          # 每隔多少 epoch 打印一次详细统计信息
"""
        print(example_config)
        exit(1) # 退出程序

    # --- 如果文件存在，调用 main 函数开始训练 ---
    main(config_file_path)