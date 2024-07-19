import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Scaled Dot Production Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask = None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype = torch.float32))
       #scores = [batch size, n heads, query len, key len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim = -1)
        #attention = [batch size, n heads, query len, key len]
        
        output = torch.matmul(attn, V)
        #x = [batch size, n heads, query len, head dim]

        return output, attn

# Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask = None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        attn_output, attn = ScaledDotProductAttention()(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
        #x = [batch size, query len, n heads, head dim]
        #x = [batch size, query len, hid dim]

        output = self.fc(attn_output)
        #x = [batch size, query len, hid dim]

        return output, attn
    
# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
        #x = [batch size, seq len, hid dim]

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask = None):
        src1, _ = self.attention(src, src, src, mask)
        src = src + self.dropout1(src1)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, dropout = 0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, mask = None):
        src = self.embedding(src)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, mask)
        #src = [batch size, src len, hid dim]

        return self.norm(src)

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask = None, tgt_mask = None):
        tgt2, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn = self.encoder_attention(tgt, memory, memory, src_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn
    
# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, dropout = 0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask = None, tgt_mask = None):
        tgt = self.embedding(tgt)
        tgt = self.dropout(tgt)

        for layer in self.layers:
            tgt, attn = layer(tgt, memory, src_mask, tgt_mask)
        
        tgt = self.fc(tgt)
        tgt = F.softmax(tgt, dim=-1)

        return tgt, attn
    