import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
     Encodes sequential data using a standard or bidirectional LSTM.

     :param input_dim: Size of input features
     :param hidden_dim: Size of LSTM hidden state
     :param num_layers: Number of LSTM layers
     :param bidirectional: Whether to use bidirectional LSTM
     """

    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers * num_directions, batch, hidden_dim]
        # Take the last layer's output
        h_last = h_n[-self.num_directions:]  # shape: [num_directions, batch, hidden_dim]
        embedding = h_last.permute(1, 0, 2).reshape(x.size(0), -1)  # shape: [batch, hidden_dim * num_directions]
        return embedding
