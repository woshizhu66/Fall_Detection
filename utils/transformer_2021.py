"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):

    def __init__(self, n_inputs, n_outputs, window_size, encode_position, transformer_dim, dropout):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = transformer_dim

        self.input_proj = nn.Sequential(nn.Conv1d(n_inputs, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = window_size
        self.encode_position = encode_position
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim,
                                                nhead=8,
                                                dim_feedforward=128,
                                                dropout=dropout,
                                                activation="gelu")

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=6,
                                                      norm=nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        num_classes = n_outputs
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 4, num_classes)
        )

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src = self.input_proj(data.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]

        # Class probability
        target = self.imu_head(target)

        return target


def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))
