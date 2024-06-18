#@title Trying with Conv of nodes also
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#@markdown Need to figure out best context scale size (must be a factor od 2000)
context_scale = 10 #@param

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = int(num_heads)  # Ensure num_heads is an integer
        self.num_groups = num_groups
        self.head_dim = embed_dim // self.num_heads  # Ensure head_dim is an integer

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Input embedding dimension must match model embedding dimension"

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        assert seq_len % self.num_groups == 0, "seq_len must be divisible by num_groups"
        group_size = seq_len // self.num_groups

        # Group queries
        q_groups = q.view(batch_size, self.num_heads, self.num_groups, group_size, self.head_dim)  # (batch_size, num_heads, num_groups, group_size, head_dim)
        k_groups = k.view(batch_size, self.num_heads, self.num_groups, group_size, self.head_dim)
        v_groups = v.view(batch_size, self.num_heads, self.num_groups, group_size, self.head_dim)

        # Compute attention for each group
        attn_scores = torch.einsum('bhgqd,bhgkd->bhgqk', q_groups, k_groups)  # (batch_size, num_heads, num_groups, group_size, group_size)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.einsum('bhgqk,bhgvd->bhgqd', attn_probs, v_groups)  # (batch_size, num_heads, num_groups, group_size, head_dim)
        attn_output = attn_output.contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)

def modulate(normed_x, shift, scale):
    return normed_x * (1 + scale) + shift

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, feedforward_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_groups)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.attention(modulate(self.norm1(x), shift_msa, scale_msa))
        x = self.norm1(x)
        x = x + gate_mlp * self.feedforward(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.norm2(x)
        return x

class GeneticDiffusionModuleBlock(nn.Module):
    def __init__(self, channels: int, num_diffusion_steps: int = 100, training: bool = False, depth: int = 3):
        super(GeneticDiffusionModuleBlock, self).__init__()
        self.channels = channels
        assert channels % 8 == 0, "channels must be divisible by 64"
        num_heads = channels // 8
        self.num_diffusion_steps = num_diffusion_steps
        self.time_embeddings = nn.Parameter(torch.randn(num_diffusion_steps, 2000//context_scale, channels))
        self.training = training
        self.depth = depth
        self.noise_scale = nn.Parameter(torch.linspace(1.0, 0.01, num_diffusion_steps))

        # Custom transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim=channels, num_heads=num_heads, num_groups=num_heads//2, feedforward_dim=channels * 4) for _ in range(3)
        ])

    def forward(self, x: Tensor = None, ground_truth: Tensor = None):
        batch_size, num_nodes, k = x.size()
        x_1 = x.clone()

        # Simulate the multi-step diffusion process
        for step in range(self.num_diffusion_steps):
            noise_level = self.noise_scale[step]  # Get the noise level for the current step
            noise = torch.randn_like(x) * noise_level  # Generate noise scaled by the noise level
            x_1 = x_1 + noise  # Add noise to the input

            c = self.time_embeddings[step] # Get the time embedding for the current step
            c = c.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, 1000, channels]

            # Apply custom transformer layers
            for transformer_layer in self.transformer_layers:
                x_1 = transformer_layer(x_1, c)


        if self.training and ground_truth is not None:
            loss = F.mse_loss(x_1, ground_truth)
            return x_1, loss

        return x_1

class GeneticDiffusion(nn.Module):
    def __init__(self, channels: int, num_diffusion_steps: int = 10, k: int = 64, embeddings: int = 384, training: bool = False, depth: int = 3):
        super(GeneticDiffusion, self).__init__()
        self.channels = channels
        self.num_diffusion_steps = num_diffusion_steps
        self.training = training
        self.depth = depth

        # 2D convolutional layer to transform the input tensor from [10, 2000, 24576] to [10, 200, 128]
        self.conv = nn.Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=(context_scale, 192),
                                stride=(context_scale, 192),
                                padding=0)
        # Layers
        self.layers = nn.ModuleList([
            GeneticDiffusionModuleBlock(channels, num_diffusion_steps, training, depth) for _ in range(depth)
        ])

    def forward(self, x: Tensor = None, ground_truth: Tensor = None):
        # Assuming input x shape is [batch_size, nodes, k, embeddings]
        batch_size, nodes, k, embeddings = x.size()
        # Flatten the k neighbors' embeddings per node into a 1D vector
        x = x.view(batch_size, nodes, -1)  # [batch_size, nodes, k * embeddings]
        # Pad nodes to ensure a total of context_size nodes per batch
        padding = torch.zeros(batch_size, 2000 - nodes, k * embeddings).to(x.device)
        if nodes < 2000:
            x = torch.cat([x, padding], dim=1)  # [batch_size, context_size, k * embeddings]
            if self.training and ground_truth is not None:
              ground_truth = ground_truth.view(batch_size, nodes, -1)
              ground_truth = torch.cat([ground_truth, padding], dim=1)
              ground_truth = ground_truth.unsqueeze(1)  # Add channel dimension
              ground_truth = self.conv(ground_truth)
              # Reshape back to desired shape: [batch_size, height, width]
              ground_truth = ground_truth.squeeze(1)


        # Reshape to fit Conv2d input: [batch_size, channels, height, width]
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        # Reshape back to desired shape: [batch_size, height, width]
        x = x.squeeze(1)  # Remove channel dimension

        # Apply GeneticDiffusionModuleBlock
        loss = None
        if self.training and ground_truth is not None:
            for layer in self.layers:
                x, loss = layer(x, ground_truth)
            return x, loss
        else:
            for layer in self.layers:
                x = layer(x)
            return x

# Example usage with fixed number of nodes per batch
batch_size = 10
nodes = 1929  # Fixed number of nodes
k = 64
embeddings = 384
channels = 128  # Number of channels for the diffusion

# Creating a dataset with a fixed number of nodes
dummy_inputs = [torch.randn(batch_size, nodes, k, embeddings)]
dummy_ground_truths = [torch.randn(batch_size, nodes, k, embeddings)]
num_diffusion_steps = 100 #@param
model = GeneticDiffusion(channels=channels, num_diffusion_steps=num_diffusion_steps, training=True, depth=3)

for dummy_input, dummy_ground_truth in zip(dummy_inputs, dummy_ground_truths):
    output, loss = model(dummy_input, dummy_ground_truth)
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item()}")

