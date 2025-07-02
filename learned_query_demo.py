import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import time

class LearnedTokenSelector(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int, selection_ratio: float = 0.5):
        super().__init__()
        self.learned_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.selection_ratio = selection_ratio
        self.num_selected = max(1, int(num_tokens * selection_ratio))
        
        # 注意力机制组件
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [batch_size, num_tokens, embed_dim]
        返回: 选择后的token和选择掩码
        """
        batch_size, num_tokens, embed_dim = x.shape
        
        # 计算注意力得分
        q = self.q_proj(self.learned_query).expand(batch_size, -1, -1)  # [B, 1, D]
        k = self.k_proj(x)  # [B, N, D]
        scores = torch.bmm(q, k.transpose(1, 2)) / (embed_dim ** 0.5)  # [B, 1, N]
        
        # 计算选择概率
        probs = F.softmax(scores, dim=-1)  # [B, 1, N]
        
        # 基于概率选择token
        if self.training:
            # 训练时使用gumble-softmax进行可微分选择
            gumbel_samples = F.gumbel_softmax(probs, tau=0.5, hard=True)
            selected_mask = gumbel_samples.squeeze(1)  # [B, N]
        else:
            # 推理时直接选择top-k
            _, indices = torch.topk(probs, self.num_selected, dim=-1)
            selected_mask = torch.zeros_like(probs).scatter_(2, indices, 1.0).squeeze(1)
        
        # 应用选择掩码
        selected_tokens = x * selected_mask.unsqueeze(-1)  # [B, N, D]
        
        return selected_tokens, selected_mask

class PrunedViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        selection_ratio: float = 0.5,
        num_classes: int = 1000
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim))
        
        # Token选择器
        self.token_selector = LearnedTokenSelector(
            embed_dim=embed_dim,
            num_tokens=(image_size // patch_size) ** 2,
            selection_ratio=selection_ratio
        )
        
        # 编码器块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation=F.gelu,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # 添加分类token和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # 分离cls token和patch tokens
        cls_token = x[:, 0:1]
        patch_tokens = x[:, 1:]
        
        # 应用token选择
        selected_tokens, selection_mask = self.token_selector(patch_tokens)
        
        # 合并cls token和选择后的patch tokens
        x = torch.cat([cls_token, selected_tokens], dim=1)
        
        # 通过Transformer编码器
        x = self.encoder(x)
        x = self.norm(x)
        
        # 使用cls token进行分类
        x = x[:, 0]
        x = self.head(x)
        
        return x, selection_mask

def count_parameters(model: nn.Module) -> int:
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型实例
    # 原始模型
    original_vit = PrunedViT(selection_ratio=1.0).to(device)
    # 压缩模型 (保留50%的tokens)
    pruned_vit = PrunedViT(selection_ratio=0.5).to(device)
    
    # 打印模型参数数量
    print(f"Original ViT parameters: {count_parameters(original_vit):,}")
    print(f"Pruned ViT parameters: {count_parameters(pruned_vit):,}")
    
    # 创建测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 测试前向传播
    print("\nTesting forward pass...")
    
    # 原始模型
    original_vit.eval()
    start_time = time.time()
    with torch.no_grad():
        original_output, original_mask = original_vit(test_input)
    original_time = time.time() - start_time
    print(f"Original ViT inference time: {original_time:.4f} seconds")
    print(f"Original output shape: {original_output.shape}")
    print(f"Average tokens selected: {original_mask.float().mean().item() * 100:.2f}%")
    
    # 压缩模型
    pruned_vit.eval()
    start_time = time.time()
    with torch.no_grad():
        pruned_output, pruned_mask = pruned_vit(test_input)
    pruned_time = time.time() - start_time
    print(f"Pruned ViT inference time: {pruned_time:.4f} seconds")
    print(f"Pruned output shape: {pruned_output.shape}")
    print(f"Average tokens selected: {pruned_mask.float().mean().item() * 100:.2f}%")
    
    # 计算加速比
    speedup = original_time / pruned_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # 测试训练模式
    print("\nTesting training mode...")
    original_vit.train()
    pruned_vit.train()
    
    with torch.no_grad():
        original_output, original_mask = original_vit(test_input)
        pruned_output, pruned_mask = pruned_vit(test_input)
    
    print(f"Training mode - Original: {original_mask.float().mean().item() * 100:.2f}% tokens selected")
    print(f"Training mode - Pruned: {pruned_mask.float().mean().item() * 100:.2f}% tokens selected")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()