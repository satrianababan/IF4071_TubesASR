import torch.nn as nn

# Model Definition
class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ASRModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional => 2x hidden_dim

    def forward(self, x):
        # Ensure correct input shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch_size, n_mels, seq_len] -> [batch_size, seq_len, n_mels]

        # LSTM
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]

        # Fully Connected Layer
        x = self.fc(x)  # [batch_size, seq_len, output_dim]
        return x