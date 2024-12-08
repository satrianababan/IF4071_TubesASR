import torch.nn as nn

# Model Definition
class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ASRModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional => 2x hidden_dim

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x