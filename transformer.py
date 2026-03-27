import torch
import torch.nn as nn


class PreNormTransformerEncoderLayer(nn.Module):
    """
    One Pre-Norm Transformer encoder layer.

    Constructor parameters:
        d_model: int
            Hidden dimension of the model.
            Each token representation has shape [d_model].

        nhead: int
            Number of attention heads in multi-head self-attention.

        dim_feedforward: int, default=2048
            Hidden dimension of the feedforward network.

        dropout: float, default=0.1
            Dropout probability used in attention and feedforward sublayers.

    Forward input:
        x: Tensor of shape [batch_size, seq_len, d_model]
            Batched token representations.

        source_attn_mask: optional Tensor
            Attention mask used inside self-attention.
            Typical shape: [seq_len, seq_len].
            It blocks some query-key pairs from attending to each other.

        source_key_padding_mask: optional Tensor
            Padding mask of shape [batch_size, seq_len].
            True means that position is a padding token and should be ignored.

    Forward output:
        out: Tensor of shape [batch_size, seq_len, d_model]
            Output token representations after one Pre-Norm encoder layer.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Save model hyperparameters to self
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Multi-head self-attention module
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm before attention and before feedforward network
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network: d_model -> dim_feedforward -> d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropout layers
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn_hidden = nn.Dropout(dropout)
        self.dropout_ffn_out = nn.Dropout(dropout)

    def forward(self, x, source_attn_mask=None, source_key_padding_mask=None):
        # Pre-Norm attention block:
        # x = x + Attention(LayerNorm(x))
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=source_attn_mask,
            key_padding_mask=source_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout_attn(attn_out)

        # Pre-Norm feedforward block:
        # x = x + FFN(LayerNorm(x))
        x_norm = self.norm2(x)
        ffn_out = self.linear2(
            self.dropout_ffn_hidden(
                self.activation(self.linear1(x_norm))
            )
        )
        x = x + self.dropout_ffn_out(ffn_out)

        return x


class PreNormTransformerEncoder(nn.Module):
    """
    Multi-layer Pre-Norm Transformer encoder.

    Constructor parameters:
        d_model: int
            Hidden dimension of the model.
            Each token embedding / hidden state has size d_model.

        nhead: int
            Number of attention heads in each encoder layer.

        num_layers: int
            Number of stacked Pre-Norm encoder layers.

        dim_feedforward: int, default=2048
            Hidden dimension of the feedforward network in each layer.

        dropout: float, default=0.1
            Dropout probability used in each layer.

    Forward input:
        source: Tensor of shape [batch_size, seq_len, d_model]
            Batched input token representations.

        source_attn_mask: optional Tensor
            Attention mask used in self-attention.
            Typical shape: [seq_len, seq_len].

        source_key_padding_mask: optional Tensor
            Padding mask of shape [batch_size, seq_len].
            True means the corresponding token is padding.

    Forward output:
        output: Tensor of shape [batch_size, seq_len, d_model]
            Encoded sequence representations after all encoder layers.
    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()

        # Save model hyperparameters to self
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Stack multiple Pre-Norm encoder layers
        self.layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, source, source_attn_mask=None, source_key_padding_mask=None):
        output = source

        # Pass input through each encoder layer sequentially
        for layer in self.layers:
            output = layer(
                output,
                source_attn_mask=source_attn_mask,
                source_key_padding_mask=source_key_padding_mask
            )

        return output
    
class ParallelTransformerEncoderLayer(nn.Module):
    """
    One Parallel Layer Transformer encoder layer.

    Constructor parameters:
        d_model: int
            Hidden dimension of the model.
            Each token representation has shape [d_model].

        nhead: int
            Number of attention heads in multi-head self-attention.

        dim_feedforward: int, default=2048
            Hidden dimension of the feedforward network.

        dropout: float, default=0.1
            Dropout probability used in attention and feedforward sublayers.

    Forward input:
        x: Tensor of shape [batch_size, seq_len, d_model]
            Batched token representations.

        source_attn_mask: optional Tensor
            Attention mask used inside self-attention.
            Typical shape: [seq_len, seq_len].

        source_key_padding_mask: optional Tensor
            Padding mask of shape [batch_size, seq_len].
            True means that position is padding and should be ignored.

    Forward output:
        out: Tensor of shape [batch_size, seq_len, d_model]
            Output token representations after one Parallel Layer encoder block.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Save model hyperparameters to self
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # One LayerNorm shared by attention branch and MLP branch
        self.norm = nn.LayerNorm(d_model)

        # Multi-head self-attention branch
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward branch: d_model -> dim_feedforward -> d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropout layers
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn_hidden = nn.Dropout(dropout)
        self.dropout_ffn_out = nn.Dropout(dropout)

    def forward(self, x, source_attn_mask=None, source_key_padding_mask=None):
        # Parallel Layer:
        # out = x + Attention(LN(x)) + MLP(LN(x))
        x_norm = self.norm(x)

        # Attention branch
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=source_attn_mask,
            key_padding_mask=source_key_padding_mask,
            need_weights=False
        )
        attn_out = self.dropout_attn(attn_out)

        # Feedforward branch
        ffn_out = self.linear2(
            self.dropout_ffn_hidden(
                self.activation(self.linear1(x_norm))
            )
        )
        ffn_out = self.dropout_ffn_out(ffn_out)

        # Add both branches back to the same residual input
        out = x + attn_out + ffn_out
        return out


class ParallelTransformerEncoder(nn.Module):
    """
    Multi-layer Parallel Layer Transformer encoder.

    Constructor parameters:
        d_model: int
            Hidden dimension of the model.
            Each token embedding / hidden state has size d_model.

        nhead: int
            Number of attention heads in each encoder layer.

        num_layers: int
            Number of stacked Parallel Layer encoder layers.

        dim_feedforward: int, default=2048
            Hidden dimension of the feedforward network in each layer.

        dropout: float, default=0.1
            Dropout probability used in each layer.

    Forward input:
        source: Tensor of shape [batch_size, seq_len, d_model]
            Batched input token representations.

        source_attn_mask: optional Tensor
            Attention mask used in self-attention.
            Typical shape: [seq_len, seq_len].

        source_key_padding_mask: optional Tensor
            Padding mask of shape [batch_size, seq_len].
            True means the corresponding token is padding.

    Forward output:
        output: Tensor of shape [batch_size, seq_len, d_model]
            Encoded sequence representations after all encoder layers.
    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()

        # Save model hyperparameters to self
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Stack multiple Parallel Layer encoder blocks
        self.layers = nn.ModuleList([
            ParallelTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, source, source_attn_mask=None, source_key_padding_mask=None):
        output = source

        # Pass input through each parallel encoder layer
        for layer in self.layers:
            output = layer(
                output,
                source_attn_mask=source_attn_mask,
                source_key_padding_mask=source_key_padding_mask
            )

        return output