##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDN(nn.Module):
    """
    Deep Deterministic Normalization (DDN)

    A learnable normalization method that uses deep neural networks to learn
    optimal normalization and denormalization transformations for time series data.

    Architecture:
    - Statistics Encoder: Learns to extract normalization statistics
    - Normalization: Applies learned normalization
    - Denormalization Decoder: Learns inverse transformation
    """

    def __init__(self, cfg):
        """
        Initialize DDN module

        Args:
            cfg: Configuration object containing DDN parameters
        """
        super(DDN, self).__init__()
        self.cfg = cfg
        self.num_features = cfg.DATA.N_VAR
        self.seq_len = cfg.DATA.SEQ_LEN

        # DDN Configuration
        self.hidden_dim = getattr(cfg.DDN, 'HIDDEN_DIM', 64)
        self.num_layers = getattr(cfg.DDN, 'NUM_LAYERS', 2)
        self.dropout = getattr(cfg.DDN, 'DROPOUT', 0.1)
        self.eps = getattr(cfg.DDN, 'EPS', 1e-5)
        self.use_residual = getattr(cfg.DDN, 'USE_RESIDUAL', True)

        # Statistics Encoder: Extract normalization parameters
        self.stats_encoder = nn.Sequential()
        input_dim = self.num_features * self.seq_len

        # First layer
        self.stats_encoder.add_module('linear_0', nn.Linear(input_dim, self.hidden_dim))
        self.stats_encoder.add_module('relu_0', nn.ReLU())
        self.stats_encoder.add_module('dropout_0', nn.Dropout(self.dropout))

        # Hidden layers
        for i in range(1, self.num_layers):
            self.stats_encoder.add_module(f'linear_{i}', nn.Linear(self.hidden_dim, self.hidden_dim))
            self.stats_encoder.add_module(f'relu_{i}', nn.ReLU())
            self.stats_encoder.add_module(f'dropout_{i}', nn.Dropout(self.dropout))

        # Output layers for mean and scale
        self.mean_head = nn.Linear(self.hidden_dim, self.num_features)
        self.scale_head = nn.Linear(self.hidden_dim, self.num_features)

        # Denormalization Decoder: Learn inverse transformation
        self.denorm_decoder = nn.Sequential()

        # First layer
        self.denorm_decoder.add_module('denorm_linear_0', nn.Linear(self.num_features, self.hidden_dim))
        self.denorm_decoder.add_module('denorm_relu_0', nn.ReLU())
        self.denorm_decoder.add_module('denorm_dropout_0', nn.Dropout(self.dropout))

        # Hidden layers
        for i in range(1, self.num_layers):
            self.denorm_decoder.add_module(f'denorm_linear_{i}', nn.Linear(self.hidden_dim, self.hidden_dim))
            self.denorm_decoder.add_module(f'denorm_relu_{i}', nn.ReLU())
            self.denorm_decoder.add_module(f'denorm_dropout_{i}', nn.Dropout(self.dropout))

        # Output layer
        self.denorm_decoder.add_module('denorm_output', nn.Linear(self.hidden_dim, self.num_features))

        # Initialize weights
        self._init_weights()

        # Store statistics for denormalization
        self.register_buffer('stored_mean', torch.zeros(1, 1, self.num_features))
        self.register_buffer('stored_scale', torch.ones(1, 1, self.num_features))

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mode: str):
        """
        Forward pass for DDN

        Args:
            x: Input tensor [batch_size, seq_len, num_features]
            mode: 'norm' for normalization, 'denorm' for denormalization

        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")

    def _normalize(self, x):
        """
        Normalize input using learned statistics

        Args:
            x: Input tensor [batch_size, seq_len, num_features]

        Returns:
            Normalized tensor
        """
        batch_size, _, num_features = x.shape

        # Flatten for statistics extraction
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * num_features]

        # Extract statistics using encoder
        stats_features = self.stats_encoder(x_flat)  # [batch_size, hidden_dim]

        # Predict normalization parameters
        learned_mean = self.mean_head(stats_features)  # [batch_size, num_features]
        learned_scale = F.softplus(self.scale_head(stats_features)) + self.eps  # [batch_size, num_features]

        # Reshape for broadcasting
        learned_mean = learned_mean.unsqueeze(1)  # [batch_size, 1, num_features]
        learned_scale = learned_scale.unsqueeze(1)  # [batch_size, 1, num_features]

        # Store for denormalization
        self.stored_mean = learned_mean.detach()
        self.stored_scale = learned_scale.detach()

        # Apply normalization
        x_normalized = (x - learned_mean) / learned_scale

        # Optional: Add residual connection
        if self.use_residual:
            # Simple standardization as residual
            simple_mean = x.mean(dim=1, keepdim=True)
            simple_std = x.std(dim=1, keepdim=True) + self.eps
            simple_norm = (x - simple_mean) / simple_std

            # Weighted combination
            alpha = torch.sigmoid(self.stats_encoder[0].weight.mean())  # Learnable weight
            x_normalized = alpha * x_normalized + (1 - alpha) * simple_norm

        return x_normalized

    def _denormalize(self, x):
        """
        Denormalize output using stored statistics and learned decoder

        Args:
            x: Normalized tensor [batch_size, seq_len, num_features]

        Returns:
            Denormalized tensor
        """
        batch_size, seq_len, num_features = x.shape

        # Apply stored normalization parameters
        x_denorm_basic = x * self.stored_scale + self.stored_mean

        # Apply learned denormalization refinement
        x_reshaped = x.view(batch_size * seq_len, num_features)  # [batch_size * seq_len, num_features]
        denorm_adjustment = self.denorm_decoder(x_reshaped)  # [batch_size * seq_len, num_features]
        denorm_adjustment = denorm_adjustment.view(batch_size, seq_len, num_features)

        # Combine basic denormalization with learned adjustment
        x_denormalized = x_denorm_basic + 0.1 * denorm_adjustment  # Small adjustment weight

        return x_denormalized

    def get_statistics(self):
        """
        Get current normalization statistics

        Returns:
            Dictionary containing mean and scale statistics
        """
        return {
            'mean': self.stored_mean.cpu().numpy(),
            'scale': self.stored_scale.cpu().numpy()
        }

    def reset_statistics(self):
        """Reset stored statistics"""
        self.stored_mean.zero_()
        self.stored_scale.fill_(1.0)


class DDNLoss(nn.Module):
    """
    Loss function for DDN training
    Combines reconstruction loss with regularization terms
    """

    def __init__(self, cfg):
        super(DDNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.regularization_weight = getattr(cfg.DDN, 'REG_WEIGHT', 0.01)

    def forward(self, original, normalized, denormalized, ddn_module):
        """
        Compute DDN loss

        Args:
            original: Original input
            normalized: Normalized output
            denormalized: Denormalized output
            ddn_module: DDN module for regularization

        Returns:
            Total loss
        """
        # Reconstruction loss
        reconstruction_loss = self.mse_loss(denormalized, original)

        # Regularization: Encourage reasonable normalization
        norm_reg = torch.mean(torch.abs(normalized))  # L1 norm of normalized values
        scale_reg = torch.mean(torch.abs(ddn_module.stored_scale - 1.0))  # Encourage scale close to 1

        # Total loss
        total_loss = reconstruction_loss + self.regularization_weight * (norm_reg + scale_reg)

        return total_loss, reconstruction_loss, norm_reg, scale_reg