import torch.nn as nn

# Model Definition
class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(ASRModel, self).__init__()

        # Ensure hidden_dims is a list
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * 3  # Default: 3 layers with the same hidden_dim

        self.num_layers = len(hidden_dims)
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # LSTM Layers with Layer Normalization
        for i in range(self.num_layers):
            input_size = input_dim if i == 0 else hidden_dims[i - 1] * 2  # Bidirectional doubles output size
            self.lstm_layers.append(
                nn.LSTM(input_size, hidden_dims[i], batch_first=True, bidirectional=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i] * 2))  # LayerNorm after bidirectional LSTM

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully Connected Layer (Final Output)
        self.fc = nn.Linear(hidden_dims[-1] * 2, output_dim)

    def forward(self, x):
        # Transpose input to [batch_size, seq_len, n_mels]
        x = x.transpose(1, 2)

        # Pass through LSTM layers with Layer Normalization and Residual Connections
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            residual = x if i > 0 else None  # Residual connection (skip for the first layer)
            x, _ = lstm(x)  # LSTM output: [batch_size, seq_len, hidden_dim * 2]
            x = layer_norm(x)  # Apply LayerNorm
            if residual is not None:
                x += residual  # Add residual connection
            x = self.dropout(x)  # Apply dropout

        # Fully Connected Layer
        x = self.fc(x)  # [batch_size, seq_len, output_dim]
        return x