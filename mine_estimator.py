# mine_estimator.py
"""
MINE 互信息估计器
"""
import torch
import torch.optim as optim
import numpy as np
# 导入两个网络类
from mine_network import MINENetworkFloat, MINENetworkIndex

class MINE:
    """使用神经网络估计互信息 I(X;Y)"""

    def __init__(self, x_dim=None, y_dim=None, network_type='float',
                 vocab_size=None, embedding_dim=None,
                 hidden_dims=[128,64],
                 activation="relu", batch_norm=True, dropout=0.1,
                 lr=1e-4, device="cuda"):
        """
        初始化MINE估计器

        参数:
            x_dim: X 的特征维度 (当 network_type='float' 时需要)
            y_dim: Y 的特征维度 (始终需要)
            network_type: 网络类型 ('float' 或 'index')
            vocab_size: X 的词汇表大小 (当 network_type='index' 时需要)
            embedding_dim: X 的 Embedding 维度 (当 network_type='index' 时需要)
            hidden_dims: MINE网络隐藏层维度列表
            activation: 激活函数
            batch_norm: 是否使用批归一化
            dropout: Dropout率
            lr: 学习率
            device: 设备 ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"MINE Estimator using device: {self.device}")
        self.network_type = network_type

        # --- 创建 MINE 网络 ---
        if self.network_type == 'float':
            if x_dim is None or y_dim is None:
                raise ValueError("x_dim and y_dim must be provided for network_type 'float'")
            print(f"Initializing MINENetworkFloat (x_dim={x_dim}, y_dim={y_dim})")
            self.mine_net = MINENetworkFloat(
                x_dim, y_dim, hidden_dims, activation, batch_norm, dropout
            ).to(self.device)
        elif self.network_type == 'index':
            if vocab_size is None or embedding_dim is None or y_dim is None:
                raise ValueError("vocab_size, embedding_dim, and y_dim must be provided for network_type 'index'")
            print(f"Initializing MINENetworkIndex (vocab={vocab_size}, embed_dim={embedding_dim}, y_dim={y_dim})")
            self.mine_net = MINENetworkIndex(
                vocab_size, embedding_dim, y_dim, hidden_dims, activation, batch_norm, dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown network_type: {self.network_type}. Choose 'float' or 'index'.")

        # 优化器
        self.optimizer = optim.Adam(
            self.mine_net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        # 学习率调度器 (可选, 但推荐)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr / 10
        )

        # 训练历史记录
        self.train_mi_history = []
        self.val_mi_history = []
        self.best_val_mi = float('-inf')

        # 诊断信息
        self.recent_t_values = []
        self.gradient_norms = []

    def compute_mutual_info(self, x_joint, y_joint):
        """
        计算互信息估计 (Donsker-Varadhan lower bound)

        参数:
            x_joint: 联合分布 X 样本, 形状 [B, S, Dx] (float) 或 [B, S] (long)
            y_joint: 联合分布 Y 样本, 形状 [B, S, Dy] (float)

        返回:
            互信息估计值 (标量 Tensor)
        """
        # --- 1. 生成 y_marginal ---
        # 通过在 N=B*S 维度上 shuffle y_joint 来生成边缘样本 y'
        if y_joint.ndim == 3: # [B, S, Dy]
            B, S, Dy = y_joint.shape
            N = B * S
            y_joint_flat = y_joint.reshape(N, Dy)
            shuffle_idx = torch.randperm(N).to(self.device)
            y_marginal_flat = y_joint_flat[shuffle_idx]
            # 将 y_marginal reshape 回原始形状，因为 self.mine_net.forward 期望原始形状
            y_marginal = y_marginal_flat.reshape(B, S, Dy)
        elif y_joint.ndim == 2: # [B, Dy] - 非序列情况
            N = y_joint.shape[0]
            shuffle_idx = torch.randperm(N).to(self.device)
            y_marginal = y_joint[shuffle_idx]
            # 检查 x 的维度是否匹配
            if x_joint.ndim != (1 if self.network_type == 'index' else 2) or x_joint.shape[0] != N:
                 raise ValueError("Shape mismatch between x_joint and y_joint for non-sequential case")
        else:
             raise ValueError(f"Unsupported y_joint dimension: {y_joint.ndim}")


        # --- 2. 计算网络输出 ---
        # 网络 forward 方法内部会处理 reshape，返回 [N, 1]
        t_joint = self.mine_net(x_joint, y_joint)
        t_marginal = self.mine_net(x_joint, y_marginal) # 使用原始 x 和 shuffle 后的 y

        # --- 诊断信息 ---
        self.recent_t_values.append({
            'joint_mean': t_joint.mean().item(),
            'joint_std': t_joint.std().item(),
            'marginal_mean': t_marginal.mean().item(),
            'marginal_std': t_marginal.std().item()
        })
        if len(self.recent_t_values) > 50:
            self.recent_t_values.pop(0)
        # --- 诊断信息结束 ---

        # --- 3. 计算 MI lower bound ---
        # E_P[T] - log(E_Q[e^T])，其中 P 是联合分布，Q 是边缘分布的乘积
        e_joint = torch.mean(t_joint)

        # 使用 logsumexp 技巧提高数值稳定性计算 log(E_Q[e^T])
        max_t = torch.max(t_marginal).detach()
        log_e_marginal = max_t + torch.log(
            torch.mean(torch.exp(t_marginal - max_t)) + 1e-8 # 加小 epsilon 防 log(0)
        )

        mi_estimate = e_joint - log_e_marginal
        return mi_estimate

    def train_batch(self, x_joint_orig, y_joint_orig):
        """
        用单个批次训练 MINE 网络

        参数:
            x_joint_orig: 原始联合 X 样本 (float or long)
            y_joint_orig: 原始联合 Y 样本 (float)

        返回:
            该批次的 MI 估计值 (float), 损失值 (float)
        """
        self.mine_net.train()

        # 移动数据到设备并确保类型正确
        # y 始终是 float
        y_joint = y_joint_orig.float().to(self.device)
        # x 根据网络类型确定类型
        if self.network_type == 'index':
            x_joint = x_joint_orig.long().to(self.device) # 确保是 LongTensor
        else:
            x_joint = x_joint_orig.float().to(self.device)

        # 计算 MI (内部会生成 y_marginal 并调用 forward)
        mi_estimate = self.compute_mutual_info(x_joint, y_joint)

        # 损失函数: 最大化 MI 等价于最小化 -MI
        loss = -mi_estimate

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.mine_net.parameters(), max_norm=1.0
        )
        self.gradient_norms.append(grad_norm.item())
        if len(self.gradient_norms) > 100: # 只保留最近的梯度范数
            self.gradient_norms.pop(0)

        self.optimizer.step()

        # 检查是否有 NaN 或 Inf (有助于调试)
        if torch.isnan(loss) or torch.isinf(loss):
             print(f"警告: 损失值出现 NaN 或 Inf!")

        return mi_estimate.item(), loss.item()

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.mine_net.train()
        epoch_mi = []
        total_loss = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            mi_value, loss_value = self.train_batch(batch_x, batch_y)

            if not np.isnan(mi_value) and not np.isinf(mi_value):
                epoch_mi.append(mi_value)
                total_loss += loss_value
            else:
                print(f"警告: Batch {batch_idx} 产生无效 MI/Loss 值，已跳过。")

            if abs(mi_value) > 100 and not np.isnan(mi_value):
                 print(f"警告: Batch {batch_idx} MI 值异常: {mi_value:.4f}")

        # 更新学习率
        self.scheduler.step()

        avg_mi = np.mean(epoch_mi) if epoch_mi else 0.0
        avg_loss = total_loss / len(epoch_mi) if epoch_mi else 0.0
        self.train_mi_history.append(avg_mi)

        return avg_mi, avg_loss

    def evaluate(self, data_loader):
        """在验证集或测试集上评估模型"""
        self.mine_net.eval()
        eval_mi = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                # 移动数据到设备并确保类型正确
                y_joint = batch_y.float().to(self.device)
                if self.network_type == 'index':
                    x_joint = batch_x.long().to(self.device)
                else:
                    x_joint = batch_x.float().to(self.device)

                # 计算 MI
                mi_estimate = self.compute_mutual_info(x_joint, y_joint)

                if not np.isnan(mi_estimate.item()) and not np.isinf(mi_estimate.item()):
                    eval_mi.append(mi_estimate.item())

        avg_mi = np.mean(eval_mi) if eval_mi else 0.0
        return avg_mi

    def validate(self, val_loader):
        """执行验证步骤并更新最佳 MI"""
        avg_val_mi = self.evaluate(val_loader)
        self.val_mi_history.append(avg_val_mi)

        if avg_val_mi > self.best_val_mi:
            self.best_val_mi = avg_val_mi
            print(f"  * New best validation MI: {self.best_val_mi:.4f}")
            # 保存模型
            # torch.save(self.mine_net.state_dict(), 'best_mine_model.pth')

        return avg_val_mi

    def get_network_stats(self):
        """返回网络状态的诊断信息"""
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