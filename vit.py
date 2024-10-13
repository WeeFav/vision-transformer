import torch
import torch.nn as nn
import math

class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(self.num_channels, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, patch_size, patch_size, embedding_dim)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, embedding_dim)
        return x

class Embeddings(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, embedding_dim, batch_size):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, num_channels, embedding_dim)
        self.cls_tokens = nn.Parameter(torch.randn(batch_size, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(torch.randn(batch_size, self.patch_embeddings.num_patches + 1, embedding_dim))
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        embeddings = self.patch_embeddings(x)
        embeddings = torch.cat((self.cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, d_k, bias):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.d_k = d_k
        self.query = nn.Linear(embedding_dim, self.d_k, bias=bias)
        self.key = nn.Linear(embedding_dim, self.d_k, bias=bias)
        self.value = nn.Linear(embedding_dim, self.d_k, bias=bias)
        self.qkv_dropout = nn.Dropout(0)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # print("query.shape", query.shape)
        # print("key.shape", key.shape)
        # print("value.shape", value.shape)
        # print("key.transpose(-1, -2).shape", key.transpose(-1, -2).shape)

        attention = torch.matmul(query, key.transpose(-1, -2))
        attention = attention / math.sqrt(self.d_k)
        attention = nn.functional.softmax(attention, dim=-1)
        # print("attention.shape", attention.shape)
        attention = torch.matmul(attention, value)
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, bias):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.d_k = embedding_dim // n_heads # this is usually how d_k is calculated, since we need to split number of embeddedings across number of heads
        self.all_head_size = n_heads * self.d_k
        self.bias = bias

        self.heads = nn.ModuleList([])
        for n in range(n_heads):
            head = AttentionHead(embedding_dim, self.d_k, bias=bias)
            self.heads.append(head)

        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        attention_list = [head(x) for head in self.heads]
        attention_cat = torch.cat([attention for attention in attention_list], dim=-1)
        attention_cat = self.linear(attention_cat)
        attention_cat = self.dropout(attention_cat)
        return attention_cat

class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.acctivation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.acctivation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, hidden_dim):
        super().__init__()
        self.LN1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.MHA = MultiHeadAttention(embedding_dim, n_heads, bias=True)
        self.LN2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.MLP = MLP(embedding_dim, hidden_dim)

    def forward(self, x):
        input = x
        x = self.LN1(x)
        x = self.MHA(x)
        x = x + input
        input2 = x
        x = self.LN2(x)
        x = self.MLP(x)
        x = x + input2
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, embedding_dim, batch_size, n_heads, hidden_dim, num_blocks, num_classes):
        super().__init__()
        self.embedding_model = Embeddings(image_size, patch_size, num_channels, embedding_dim, batch_size)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            block = TransformerBlock(embedding_dim, n_heads, hidden_dim)
            self.blocks.append(block)
        self.mlp_head = nn.Linear(embedding_dim, num_classes) # only the cls token will get passed into the mlp head

    def forward(self, x):
        x = self.embedding_model(x)
        for block in self.blocks:
            x = block(x)
        logits = self.mlp_head(x[:,0,:])

        return logits